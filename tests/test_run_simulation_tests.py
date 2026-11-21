"""Unified runner failure handling with subprocess fixtures and no IOC or Podman."""
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_simulation_tests as runner


class SimulationRunnerFailureTests(unittest.TestCase):
    def exercise(self, ready):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'ecmc_podman'
            runtime.mkdir()
            fixture = runtime / 'fake_ioc.py'
            fixture.write_text(
                'import pathlib, sys\n'
                'print("fake IOC startup", flush=True)\n'
                + ('print("iocRun: All initialization complete", flush=True)\n'
                   'print("run-sim: hardware models started for SIMMOTION", flush=True)\n'
                   if ready else '')
                + 'if sys.stdin.readline().strip() == "exit":\n'
                  '    pathlib.Path("ioc-exited").write_text("clean shutdown")\n'
            )
            launcher = runtime / 'run-sim.sh'
            launcher.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(fixture))}\n')
            launcher.chmod(0o755)
            motion = root / 'motion'
            motion.mkdir()
            (motion / 'check.py').write_text('print("fake checker failure")\nraise SystemExit(7)\n')
            argv = ['runner', '--prebuilt', '--suite', 'motion', '--suite', 'inventory',
                    '--startup-timeout', '5' if ready else '.5']
            with (patch.object(runner, 'ROOT', root),
                  patch.object(runner, 'INTEGRATION', root),
                  patch.object(runner, 'find_spec', return_value=object()),
                  patch.object(sys, 'argv', argv)):
                self.assertEqual(runner.main(), 1)

            run, = (root / 'test-results/simulation').iterdir()
            report = json.loads((run / 'report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual([(s['name'], s['status']) for s in report['stages']],
                             [('dependencies', 'passed'), ('motion', 'failed')])
            self.assertIn('fake IOC startup', (run / 'motion-ioc.log').read_text())
            self.assertTrue((root / 'ioc-exited').is_file())
            self.assertFalse((run / 'inventory-ioc.log').exists())
            error = report['stages'][-1]['error']
            if ready:
                self.assertIn('status 7', error)
                self.assertIn('fake checker failure', (run / 'motion-check.log').read_text())
            else:
                self.assertIn('readiness markers', error)
                self.assertFalse((run / 'motion-check.log').exists())

    def test_startup_timeout_stops_ioc_and_preserves_report(self):
        self.exercise(ready=False)

    def test_checker_failure_stops_ioc_and_skips_remaining_suites(self):
        self.exercise(ready=True)


if __name__ == '__main__':
    unittest.main()
