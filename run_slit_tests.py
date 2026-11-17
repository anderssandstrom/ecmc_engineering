#!/usr/bin/env python3
"""Build local sources, start the dedicated slit SIM IOC, run tests, then stop it."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / 'ecmccfg/tests/integration/slit/run.py'


def command(args):
    result = [sys.executable, str(RUNNER), '--prefix', 'SIMSLIT:']
    for option in ('case', 'test'):
        for value in getattr(args, option):
            result += ['--' + option, value]
    for option in ('loop', 'report', 'startat'):
        value = getattr(args, option)
        if value is not None:
            result += ['--' + option, str(value)]
    if args.keep_going:
        result += ['--keep-going']
    if not args.list:
        result += ['--plant-only' if args.plant_only else '--run']
    return result


def stop_ioc(process, name):
    if process.poll() is None:
        try:
            process.stdin.write('exit\n')
            process.stdin.flush()
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            process.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass
    # Also handle a launcher that died while leaving its container running.
    subprocess.run(['podman', 'stop', '--time', '5', name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=10)


def run_ioc(args, test_command, environment):
    name = 'ecmc-slit-' + uuid.uuid4().hex[:12]
    launch = [str(ROOT / 'ecmc_podman/run-sim.sh'), '--image', args.image,
              '--container-name', name, '--ioc-name', 'SIMSLIT',
              str(ROOT / 'ecmccfg/tests/integration/slit/ioc/startup.cmd')]
    trace_axis = getattr(args, 'motor_status_trace_axis', None)
    if trace_axis is not None:
        launch[1:1] = ['--env', f'ECMC_MOTOR_STATUS_TRACE_AXIS={trace_axis}']
    requested_log = getattr(args, 'ioc_log', None)
    log_path = (Path(requested_log).expanduser() if requested_log else
                ROOT / 'test-results/slit' /
                f'ioc-{datetime.now():%Y%m%d-%H%M%S}-{name[-12:]}.log').resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Append when a path is explicitly reused, preserving previous diagnostics.
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\nStarting slit IOC: ' + shlex.join(launch) + '\n')
    print(f'IOC log: {log_path}', flush=True)
    print('Follow in another terminal: tail -f ' + shlex.quote(str(log_path)), flush=True)
    print('Starting slit IOC…', flush=True)
    process = subprocess.Popen(launch, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               env=environment, start_new_session=True)
    ready = threading.Event()

    def output():
        with log_path.open('a', encoding='utf-8', buffering=1) as log:
            for line in process.stdout:
                log.write(line)
                if getattr(args, 'show_ioc_log', False):
                    print(line, end='', flush=True)
                if ('iocRun: All initialization complete' in line
                        or 'iocInit: All initialization complete' in line):
                    ready.set()

    reader = threading.Thread(target=output, daemon=True)
    reader.start()
    try:
        deadline = time.monotonic() + args.startup_timeout
        while not ready.wait(.1):
            if process.poll() is not None:
                raise RuntimeError(f'IOC exited before initialization (status {process.returncode})')
            if time.monotonic() >= deadline:
                raise RuntimeError('Timed out waiting for IOC initialization; see IOC log')
        if process.poll() is not None:
            raise RuntimeError('IOC exited after initialization')
        print('Running slit tests: ' + shlex.join(test_command), flush=True)
        # The runner independently validates the SIM mode and fixture schema
        # through Channel Access before starting its coupled plant or writing PVs.
        tests = subprocess.Popen(test_command, env=environment, start_new_session=True)
        try:
            while tests.poll() is None:
                if process.poll() is not None:
                    raise RuntimeError('IOC exited while slit tests were running')
                time.sleep(.1)
            return tests.returncode
        finally:
            if tests.poll() is None:
                tests.send_signal(signal.SIGINT)
                try:
                    tests.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    tests.kill()
                    tests.wait()
    finally:
        print(f'Stopping slit IOC… (log: {log_path})', flush=True)
        stop_ioc(process, name)
        if not process.stdin.closed:
            process.stdin.close()
        reader.join(timeout=2)
        process.stdout.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ioc-log', help='IOC log file (appends; default: timestamped file in test-results/slit/)')
    parser.add_argument('--show-ioc-log', action='store_true', help='also print IOC output in the test terminal')
    parser.add_argument('--motor-status-trace-axis', type=int, help='buffer motor status diagnostics for this ECMC axis; dump on fault reset into IOC log')
    parser.add_argument('--image', default='ecmc-dev:local')
    parser.add_argument('--prebuilt', action='store_true', help='use the image as-is instead of checking/building local sources')
    parser.add_argument('--startup-timeout', type=float, default=120, help='IOC initialization timeout in seconds, excluding build time')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--list', action='store_true', help='list tests without building, starting an IOC or requiring PyEpics')
    mode.add_argument('--plant-only', action='store_true', help='start the IOC and coupled plant for manual testing; Ctrl+C stops both')
    parser.add_argument('--case', action='append', default=[], help='test group (repeatable; see --list)')
    parser.add_argument('--test', action='append', default=[], help='test name substring (repeatable)')
    parser.add_argument('--loop', choices=('open', 'closed', 'both'))
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--report', help='JSON report path; also writes matching JUnit XML')
    parser.add_argument('--startat', type=int)
    args = parser.parse_args(argv)
    if args.startup_timeout <= 0:
        parser.error('--startup-timeout must be positive')
    selected = command(args)
    if args.list:
        return subprocess.call(selected)
    # Validate test selection and Python dependencies before a potentially long build.
    plan = [part for part in selected if part not in ('--run', '--plant-only')]
    subprocess.run(plan, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, '-c', 'import epics; epics.ca.initialize_libca()'], check=True)
    environment = dict(os.environ, EPICS_CA_AUTO_ADDR_LIST='NO',
                       EPICS_CA_ADDR_LIST='127.0.0.1', EPICS_CA_SERVER_PORT='5064',
                       EPICS_CA_REPEATER_PORT='5065', PYTHONUNBUFFERED='1')
    if not args.prebuilt:
        subprocess.run([sys.executable, str(ROOT / 'ecmc_podman/build-local.py'),
                        '--image', args.image], check=True)
    return run_ioc(args, selected, environment)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'Slit simulation: {error}', file=sys.stderr)
        sys.exit(1)
