#!/usr/bin/env python3
"""Run the engineering simulation suites sequentially and report the first failure."""
import argparse
from datetime import datetime
import json
import math
from importlib.util import find_spec
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

from run_slit_tests import stop_ioc

ROOT = Path(__file__).resolve().parent
CFG = ROOT / 'ecmccfg'
INTEGRATION = CFG / 'tests/integration'
SUITES = ('offline', 'inventory', 'motion', 'slit')
INVENTORY_ERRORS = re.compile(
    r'\bERROR\s*:|Error: File .* does not exist|returned error:|drvUserCreate: (?:Parameter .* not found|Command .* only valid)|'
    r'init_record Error|Substitution file error|Unbalanced quote|Failed to open master device',
)


def run_command(argv, log_path, env, ioc=None, timeout=None, cwd=ROOT):
    with log_path.open('w') as log:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout if timeout else None
        try:
            while process.poll() is None:
                if ioc is not None and ioc.poll() is not None:
                    raise RuntimeError(f'IOC exited during checks ({ioc.returncode})')
                if deadline is not None and time.monotonic() >= deadline:
                    raise RuntimeError('Check timed out')
                time.sleep(.1)
            if process.returncode:
                raise RuntimeError(f'Command exited with status {process.returncode}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def wait_ready(ioc, log_path, markers, timeout):
    pending = set(markers)
    deadline = time.monotonic() + timeout
    with log_path.open() as log:
        while time.monotonic() < deadline:
            if ioc.poll() is not None:
                raise RuntimeError(f'IOC exited before readiness ({ioc.returncode})')
            for line in log:
                pending.difference_update(marker for marker in tuple(pending) if marker in line)
                if not pending:
                    return
            time.sleep(.1)
    raise RuntimeError('IOC did not publish readiness markers: ' + ', '.join(sorted(pending)))


def run_fixture(suite, args, directory, env):
    motion = suite == 'motion'
    prefix = 'SIMMOTION' if motion else 'SIMSLAVES'
    relative = 'motion/ioc/startup.cmd' if motion else 'slave_inventory/startup.cmd'
    name = f'ecmc-{suite}-{uuid.uuid4().hex[:12]}'
    launch = [str(ROOT / 'ecmc_podman/run-sim.sh'), '--image', args.image,
              '--container-name', name, '--ioc-name', prefix]
    if motion:
        launch.append('--hardware-sim')
    launch.append(str(INTEGRATION / relative))
    ioc_log = directory / f'{suite}-ioc.log'
    with ioc_log.open('w') as log:
        ioc = subprocess.Popen(launch, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                               stdout=log, stderr=subprocess.STDOUT, text=True,
                               start_new_session=True)
        try:
            markers = ['All initialization complete']
            if motion:
                markers.append('run-sim: hardware models started for')
            timeout = args.startup_timeout if motion else args.inventory_startup_timeout
            print(f'Waiting up to {timeout:g}s for {prefix} startup: {ioc_log}', flush=True)
            wait_ready(ioc, ioc_log, markers, timeout)
            if motion:
                argv = [sys.executable, str(INTEGRATION / 'motion/check.py')]
                for case in args.motion_case:
                    argv.extend(('--case', case))
                run_command(argv, directory / 'motion-check.log', env, ioc, timeout=600)
            else:
                # EPICS can finish initialization even when individual records fail.
                plain = re.sub(r'\x1b\[[0-9;]*m', '', ioc_log.read_text(errors='replace'))
                failures = [line.strip() for line in plain.splitlines()
                            if not line.lstrip().startswith('#') and INVENTORY_ERRORS.search(line)]
                if failures:
                    raise RuntimeError(f'Inventory IOC reported {len(failures)} startup errors. '
                                       f'First: {failures[0]}')
            if ioc.poll() is not None:
                raise RuntimeError(f'IOC exited during checks ({ioc.returncode})')
        finally:
            stop_ioc(ioc, name)
            if not ioc.stdin.closed:
                ioc.stdin.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, action='append',
                        help='Repeat to select suites. Default: all in listed order.')
    parser.add_argument('--image', default='ecmc-dev:local')
    parser.add_argument('--prebuilt', action='store_true', help='Reuse the specified image')
    parser.add_argument('--startup-timeout', type=float, default=120,
                        help='Motion and slit IOC startup timeout in seconds (default: 120)')
    parser.add_argument('--inventory-startup-timeout', type=float, default=900,
                        help='Inventory IOC startup timeout in seconds (default: 900)')
    parser.add_argument('--slit-loop', choices=('open', 'closed', 'both'), default='both')
    parser.add_argument('--motion-case', action='append', default=[],
                        help='Motion checker case, repeatable')
    parser.add_argument('--slit-case', action='append', default=[],
                        help='Slit case group, repeatable')
    parser.add_argument('--slit-test', action='append', default=[],
                        help='Slit test name substring, repeatable')
    parser.add_argument('--startat', type=int, help='Start at this numbered slit case')
    args = parser.parse_args()
    for option, value in (('--startup-timeout', args.startup_timeout),
                          ('--inventory-startup-timeout', args.inventory_startup_timeout)):
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{option} must be finite and positive')
    selected = tuple(dict.fromkeys(args.suite or SUITES))
    if args.motion_case and 'motion' not in selected:
        parser.error('--motion-case requires the motion suite')
    if (args.slit_case or args.slit_test) and 'slit' not in selected:
        parser.error('--slit-case and --slit-test require the slit suite')
    if args.startat is not None and ('slit' not in selected or args.startat < 1):
        parser.error('--startat requires the slit suite and a positive case number')
    slit_filters = [part for option, values in (('--case', args.slit_case), ('--test', args.slit_test))
                    for value in values for part in (option, value)]
    if args.startat is not None:
        slit_filters.extend(('--startat', str(args.startat)))
    directory = ROOT / 'test-results/simulation' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    directory.mkdir(parents=True)
    report_path = directory / 'report.json'
    print(f'Report and logs: {directory}', flush=True)
    report = {'status': 'running', 'selected': selected, 'stages': [],
              'filters': {'motion_case': args.motion_case, 'slit_loop': args.slit_loop,
                          'slit_case': args.slit_case, 'slit_test': args.slit_test,
                          'startat': args.startat}}
    env = dict(os.environ, EPICS_CA_AUTO_ADDR_LIST='NO', EPICS_CA_ADDR_LIST='127.0.0.1',
               EPICS_CA_SERVER_PORT='5064', EPICS_CA_REPEATER_PORT='5065',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    def stage(name, operation):
        print(f'RUN  {name}', flush=True)
        entry = {'name': name, 'status': 'running'}
        report['stages'].append(entry)
        started = time.monotonic()
        save()
        try:
            operation()
        except KeyboardInterrupt:
            entry['status'] = 'interrupted'
            raise
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            entry.update(status='failed', error=str(error))
            raise
        else:
            entry['status'] = 'passed'
            print(f'PASS {name}', flush=True)
        finally:
            entry['seconds'] = round(time.monotonic() - started, 3)
            save()

    def check_dependencies():
        packages = {}
        if 'offline' in selected:
            packages.update(jinja2='jinja2', yaml='pyyaml')
        if {'motion', 'slit'}.intersection(selected):
            packages['epics'] = 'pyepics'
        missing = [package for module, package in packages.items() if find_spec(module) is None]
        if missing:
            raise RuntimeError('Missing Python dependencies. Install with: ' +
                               shlex.join([sys.executable, '-m', 'pip', 'install', *missing]))

    image_ready = args.prebuilt
    try:
        stage('dependencies', check_dependencies)
        for suite in selected:
            if suite != 'offline' and not image_ready:
                stage('build', lambda: run_command(
                    [sys.executable, str(ROOT / 'ecmc_podman/build-local.py'), '--image', args.image],
                    directory / 'build.log', env))
                image_ready = True
            if suite == 'offline':
                for test in sorted((CFG / 'tests/simulation').glob('test_*.py')):
                    stage(f'offline/{test.stem}', lambda: run_command(
                        [sys.executable, str(test)], directory / f'{test.stem}.log', env, timeout=120, cwd=CFG))
            elif suite == 'slit':
                stage('slit', lambda: run_command(
                    [sys.executable, str(ROOT / 'run_slit_tests.py'), '--prebuilt', '--image', args.image,
                     '--loop', args.slit_loop, '--startup-timeout', str(args.startup_timeout),
                     '--ioc-log', str(directory / 'slit-ioc.log'),
                     '--report', str(directory / 'slit.json'), *slit_filters], directory / 'slit-check.log', env))
            else:
                stage(suite, lambda: run_fixture(suite, args, directory, env))
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        return 130
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        report['status'] = 'failed'
        print(f'FAIL {report["stages"][-1]["name"]}: {error}\nSee {directory}', file=sys.stderr)
        return 1
    else:
        report['status'] = 'passed'
        return 0
    finally:
        save()


if __name__ == '__main__':
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    sys.exit(main())
