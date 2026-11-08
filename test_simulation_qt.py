"""Exercise simulation routing and console I/O without starting a container."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from ecmc_engineering_qt import EngineeringWindow, QtWidgets, QtCore
from ecmc_simulation_qt import SimulationConsole, SimulationOptionsDialog, simulation_command, simulation_online_environment


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.window = EngineeringWindow()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.startup = self.root / "test_startup.cmd"
        self.startup.write_text('# test\n')
        self.window.load_project(self.startup, confirm=False)

    def wait_until(self, predicate):
        deadline = time.monotonic() + 4
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate())

    def test_launcher_arguments_and_real_script_dry_run(self):
        command = simulation_command(self.startup, Path(__file__).resolve().parent)
        self.assertEqual(command[2:], ['--image', 'ecmc-dev:portable', '--ioc-name', 'SIMIOC', '--hardware-sim', str(self.startup)])
        result = subprocess.run(command[:2] + ['--dry-run'] + command[2:], text=True, capture_output=True, check=True)
        self.assertIn('ecmc-dev:portable', result.stdout)
        self.assertIn('RUN_SIM_IOC_NAME=SIMIOC', result.stdout)
        self.assertIn('RUN_SIM_HARDWARE=1', result.stdout)
        self.assertIn(':/config:ro', result.stdout)
        self.assertIn('./test_startup.cmd', result.stdout)

    def test_missing_runtime_is_reported(self):
        with self.assertRaisesRegex(ValueError, 'runtime is missing'):
            simulation_command(self.startup, self.root)

    def test_button_routes_current_project_and_saves_unsaved_files(self):
        w = self.window
        w.editor.setPlainText('# current unsaved project\n')
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Save), patch.object(SimulationOptionsDialog, 'exec_', return_value=QtWidgets.QDialog.Accepted), patch.object(SimulationConsole, 'start') as start:
            w.simulate_project()
            start.assert_called_once()
        console = w.simulation_console
        self.assertEqual(console.command[-1], str(self.startup))
        self.assertEqual(console.working_directory, str(self.startup.parent))
        self.assertEqual(self.startup.read_text(), '# current unsaved project\n')
        self.assertIs(w.tabs.currentWidget(), console)
        with patch.object(console, 'is_running', return_value=True), patch.object(SimulationConsole, 'start') as start:
            w.simulate_project()
            start.assert_not_called()

    def test_options_route_image_and_disable_models(self):
        def accept_options(dialog):
            dialog.image.setText('custom:simulation')
            dialog.hardware_sim.setChecked(False)
            return QtWidgets.QDialog.Accepted
        with patch.object(SimulationOptionsDialog, 'exec_', accept_options), patch.object(SimulationConsole, 'start'):
            self.window.simulate_project()
        console = self.window.simulation_console
        self.assertIn('custom:simulation', console.command)
        self.assertNotIn('--hardware-sim', console.command)
        button = self.window.tabs.tabBar().tabButton(self.window.tabs.indexOf(console), QtWidgets.QTabBar.RightSide)
        button.click()
        self.assertIsNone(self.window.simulation_console)
        self.assertEqual(self.window.tabs.indexOf(console), -1)

    def test_cancel_options_does_not_launch(self):
        with patch.object(SimulationOptionsDialog, 'exec_', return_value=QtWidgets.QDialog.Rejected), patch.object(SimulationConsole, 'start') as start:
            self.window.simulate_project()
            start.assert_not_called()

    def test_close_running_tab_waits_for_ioc_exit(self):
        script = self.root / 'fake.sh'
        script.write_text('while IFS= read -r line; do [ "$line" = exit ] && exit 0; done\n')
        console = SimulationConsole(['bash', str(script)], self.root)
        self.window.simulation_console = console
        self.window.tabs.addTab(console, 'Simulation test')
        console.start()
        self.wait_until(lambda: console.process.state() == QtCore.QProcess.Running)
        self.window.close_simulation_tab()
        self.assertIs(self.window.simulation_console, console)
        self.wait_until(lambda: self.window.simulation_console is None)

    def test_cancel_does_not_launch_or_save(self):
        w = self.window
        w.editor.setPlainText('# unsaved\n')
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Cancel), patch.object(SimulationConsole, 'start') as start:
            w.simulate_project()
            start.assert_not_called()
        self.assertEqual(self.startup.read_text(), '# test\n')

    def test_console_output_commands_and_graceful_stop(self):
        script = self.root / 'fake runner.sh'
        script.write_text('printf "READY\\n"\nwhile IFS= read -r input_line; do\n  printf "GOT:%s\\n" "$input_line"\n  if [ "$input_line" = exit ]; then exit 0; fi\ndone\n')
        console = SimulationConsole(['bash', str(script)], self.root)
        self.addCleanup(lambda: console.process.kill() if console.is_running() else None)
        console.start()
        self.wait_until(lambda: 'READY' in console.output.toPlainText())
        self.assertTrue(console.online_button.isEnabled())
        console.input.setText('dbl')
        console.send_command()
        self.wait_until(lambda: 'GOT:dbl' in console.output.toPlainText())
        # Closing a running console sends exit and waits, rather than destroying QProcess.
        self.assertFalse(console.close())
        self.wait_until(lambda: not console.is_running())
        self.app.processEvents()
        self.assertIn('GOT:exit', console.output.toPlainText())
        self.assertEqual(console.status.text(), 'Simulation stopped')
        self.assertFalse(console.online_button.isEnabled())
        self.assertTrue(console.close())

    def test_failed_process_is_visible(self):
        console = SimulationConsole([str(self.root / 'does-not-exist')], self.root)
        console.start()
        self.wait_until(lambda: console.status.text() == 'Could not start simulation')
        self.assertIn('Launcher error:', console.output.toPlainText())
        self.assertFalse(console.send_button.isEnabled())

    def test_ansi_colours_are_removed_across_process_reads(self):
        source = ('\x1b[34;1m# EtherCAT bus startup delay [seconds->cycles]:\x1b[0m\n'
                  '\x1b[1mecmcEpicsEnvSetCalc("EC_DELAY_CYCLES" ,"1000*10")\x1b[0m\n')
        expected = ('# EtherCAT bus startup delay [seconds->cycles]:\n'
                    'ecmcEpicsEnvSetCalc("EC_DELAY_CYCLES" ,"1000*10")\n')
        console = SimulationConsole(['bash', 'unused'], self.root)
        # QProcess can split output anywhere, even inside an escape sequence.
        for split in range(len(source) + 1):
            console.output.clear()
            console.append_process_output(source[:split])
            console.append_process_output(source[split:], final=True)
            self.assertEqual(console.output.toPlainText(), expected)
        console.append_process_output('plain [text]\n\x1b[34;', final=True)
        console.append_process_output('next run\n')
        self.assertTrue(console.output.toPlainText().endswith('plain [text]\nnext run\n'))

    def test_online_tools_are_launched_with_local_ca_environment(self):
        with patch.dict(os.environ, {'EPICS_CA_ADDR_LIST': 'physical-controller', 'ETHERCAT_HOST': 'physical-controller'}):
            environment = simulation_online_environment()
            self.assertEqual(environment.value('EPICS_CA_ADDR_LIST'), '127.0.0.1')
            self.assertEqual(environment.value('EPICS_CA_AUTO_ADDR_LIST'), 'NO')
            self.assertEqual(environment.value('EPICS_CA_SERVER_PORT'), '5064')
            self.assertFalse(environment.contains('ETHERCAT_HOST'))
            self.assertEqual(os.environ['EPICS_CA_ADDR_LIST'], 'physical-controller')
            console = SimulationConsole(['bash', 'unused'], self.root)
            launched = []
            def detach(process):
                launched.append((process.program(), process.arguments(), process.processEnvironment()))
                return True, 1234
            with patch.object(QtCore.QProcess, 'startDetached', detach):
                console.open_online_tools()
            program, arguments, environment = launched[0]
            self.assertEqual(program, 'bash')
            self.assertTrue(arguments[0].endswith('/ecmc_cfg_tool/start_ioc.sh'))
            self.assertEqual(arguments[1:], ['SIMIOC'])
            self.assertEqual(environment.value('EPICS_CA_ADDR_LIST'), '127.0.0.1')


if __name__ == '__main__':
    unittest.main()
