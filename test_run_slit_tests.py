import argparse
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_slit_tests as wrapper


class SlitWrapperTests(unittest.TestCase):
    def fixture(self, source):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        directory = root / 'ecmc_podman'
        directory.mkdir()
        launcher = directory / 'run-sim.sh'
        launcher.write_text(f'#!{sys.executable}\n' + source)
        launcher.chmod(0o755)
        return root

    def test_runs_tests_and_returns_status_after_stopping_ioc(self):
        root = self.fixture('import sys\nprint("iocRun: All initialization complete", flush=True)\nassert sys.stdin.readline() == "exit\\n"\n')
        args = argparse.Namespace(image='test', startup_timeout=2)
        terminal = io.StringIO()
        with patch.object(wrapper, 'ROOT', root), contextlib.redirect_stdout(terminal):
            result = wrapper.run_ioc(args, [sys.executable, '-c', 'raise SystemExit(7)'], os.environ.copy())
        self.assertEqual(result, 7)
        logs = list((root / 'test-results/slit').glob('*.log'))
        self.assertEqual(len(logs), 1)
        self.assertIn('iocRun: All initialization complete', logs[0].read_text())
        self.assertNotIn('iocRun: All initialization complete', terminal.getvalue())
        self.assertIn(str(logs[0]), terminal.getvalue())

    def test_timeout_does_not_start_tests(self):
        root = self.fixture('import sys\nsys.stdin.readline()\n')
        args = argparse.Namespace(image='test', startup_timeout=.1)
        with patch.object(wrapper, 'ROOT', root):
            with self.assertRaisesRegex(RuntimeError, 'Timed out'):
                wrapper.run_ioc(args, ['must-not-execute'], os.environ.copy())

    def test_failed_start_does_not_start_tests(self):
        root = self.fixture('raise SystemExit(3)\n')
        args = argparse.Namespace(image='test', startup_timeout=2)
        with patch.object(wrapper, 'ROOT', root), patch.object(wrapper, 'stop_ioc'):
            with self.assertRaisesRegex(RuntimeError, 'before initialization'):
                wrapper.run_ioc(args, ['must-not-execute'], os.environ.copy())

    def test_list_does_not_build_or_start_ioc(self):
        with patch.object(wrapper.subprocess, 'call', return_value=0) as call, patch.object(wrapper.subprocess, 'run') as run, patch.object(wrapper, 'run_ioc') as ioc:
            self.assertEqual(wrapper.main(['--list', '--case', 'feedback', '--loop', 'closed']), 0)
            self.assertNotIn('--run', call.call_args.args[0])
            run.assert_not_called()
            ioc.assert_not_called()

    def test_build_failure_prevents_ioc_start(self):
        with patch.object(wrapper.subprocess, 'run', side_effect=[None, None, subprocess.CalledProcessError(1, 'build')]), patch.object(wrapper, 'run_ioc') as ioc:
            with self.assertRaises(subprocess.CalledProcessError):
                wrapper.main([])
            ioc.assert_not_called()

    def test_prebuilt_skips_build_and_sets_local_ca(self):
        with patch.object(wrapper.subprocess, 'run') as run, patch.object(wrapper, 'run_ioc', return_value=0) as ioc:
            self.assertEqual(wrapper.main(['--prebuilt', '--plant-only']), 0)
            self.assertEqual(run.call_count, 2)
            self.assertIn('--plant-only', ioc.call_args.args[1])
            self.assertEqual(ioc.call_args.args[2]['EPICS_CA_ADDR_LIST'], '127.0.0.1')


if __name__ == '__main__':
    unittest.main()
