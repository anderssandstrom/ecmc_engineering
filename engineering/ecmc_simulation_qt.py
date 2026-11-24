"""Run the project's Podman simulation launcher in a Qt console."""
import codecs
import json
import os
from pathlib import Path
import re
import signal
import shlex
import sys

from qt_compat import QtCore, QtGui, QtWidgets


def simulation_command(startup, repository, image="ecmc-dev:local", hardware_sim=True, local_sources=True):
    startup = Path(startup).resolve()
    launcher = Path(repository).resolve() / "ecmc_podman" / "run-sim.sh"
    if not launcher.is_file():
        raise ValueError("Simulation runtime is missing. Run git submodule update --init --recursive.")
    if not startup.is_file():
        raise ValueError("The startup file must be saved before simulation.")
    if not image.strip():
        raise ValueError("Enter a simulation image.")
    return (["bash", str(launcher), "--image", image.strip(), "--container-name",
             "ecmc-engineering-simulation", "--ioc-name", "SIMIOC"]
            + (["--local-sources"] if local_sources else [])
            + (["--hardware-sim"] if hardware_sim else []) + [str(startup)])


def project_simulation_launcher(startup):
    """Find a launcher following the standard project simulation layout."""
    launcher = Path(startup).resolve().parent / "simulation" / "run.py"
    return launcher if launcher.is_file() else None


def project_simulation_ioc_name(startup):
    """Read the optional IOC name used by a project-owned simulation."""
    return project_simulation_config(startup).get("ioc_name")


def project_simulation_config(startup):
    """Read and validate optional project simulation capabilities."""
    config = Path(startup).resolve().parent / "simulation" / "config.json"
    if not config.is_file():
        return {}
    try:
        values = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid simulation/config.json: " + str(exc))
    if not isinstance(values, dict):
        raise ValueError("simulation/config.json must contain a JSON object.")
    name = str(values.get("ioc_name", "")).strip()
    if name and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", name):
        raise ValueError("simulation/config.json contains an invalid ioc_name.")
    ioc_shell = values.get("ioc_shell", False)
    if not isinstance(ioc_shell, bool):
        raise ValueError("simulation/config.json ioc_shell must be true or false.")
    return {"ioc_name": name or None, "ioc_shell": ioc_shell}


def project_simulation_command(startup, repository, image="ecmc-dev:local", local_sources=True):
    """Build the command for a project-owned IOC and plant simulation."""
    launcher = project_simulation_launcher(startup)
    if launcher is None:
        return None
    if not image.strip():
        raise ValueError("Enter a simulation image.")
    command = [sys.executable, "-u", str(launcher), "--engineering",
               str(Path(repository).resolve()), "--image", image.strip()]
    if local_sources:
        command.append("--local-sources")
    return command


class SimulationOptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, project_launcher=None, project_ioc_name=None):
        super().__init__(parent)
        self.has_project_launcher = project_launcher is not None
        self.setWindowTitle("Start simulation")
        layout = QtWidgets.QFormLayout(self)
        self.image = QtWidgets.QLineEdit("ecmc-dev:local")
        layout.addRow("Container image", self.image)
        ioc_label = project_ioc_name or "SIMIOC"
        self.project_ioc_name = project_ioc_name
        self.ioc_name_label = QtWidgets.QLabel(ioc_label)
        layout.addRow("IOC name", self.ioc_name_label)
        self.local_sources = QtWidgets.QCheckBox("Use local ecmc, ecmccfg and ecmccomp submodules")
        self.local_sources.setChecked(True)
        layout.addRow(self.local_sources)
        self.plant_source = QtWidgets.QComboBox()
        self.plant_source.addItem("No simulation plant", "none")
        self.plant_source.addItem("Generic hardware simulation", "generic")
        if project_launcher is not None:
            self.plant_source.addItem("Project plant (simulation/run.py)", "project")
            self.plant_source.setCurrentIndex(2)
        else:
            self.plant_source.setCurrentIndex(1)
        layout.addRow("Simulation plant", self.plant_source)
        self.plant_source.currentIndexChanged.connect(self.update_ioc_name)
        if project_launcher is not None:
            detected = QtWidgets.QLabel(
                "Project simulation detected: {}\nThe project launcher will start the IOC and coupled plant."
                .format(project_launcher))
            detected.setWordWrap(True)
            layout.addRow(detected)
        self.detached_online_tools = QtWidgets.QCheckBox("Open online tools in a separate window")
        self.detached_online_tools.setChecked(False)
        layout.addRow(self.detached_online_tools)
        hint = QtWidgets.QLabel(
            "The first local-source launch may take several minutes while the modules are compiled. "
            "Unchanged sources reuse the cached image. Disable local sources to use a prebuilt image.")
        hint.setWordWrap(True)
        layout.addRow(hint)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.image.textChanged.connect(lambda text: buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(bool(text.strip())))
        layout.addRow(buttons)

    def uses_project_plant(self):
        return self.has_project_launcher and self.plant_source.currentData() == "project"

    def uses_generic_plant(self):
        return self.plant_source.currentData() == "generic"

    def update_ioc_name(self, *_):
        self.ioc_name_label.setText(
            self.project_ioc_name if self.uses_project_plant() else "SIMIOC")


def simulation_online_environment():
    environment = QtCore.QProcessEnvironment.systemEnvironment()
    environment.insert("EPICS_CA_AUTO_ADDR_LIST", "NO")
    environment.insert("EPICS_CA_ADDR_LIST", "127.0.0.1")
    environment.insert("EPICS_CA_SERVER_PORT", "5064")
    environment.insert("EPICS_CA_REPEATER_PORT", "5065")
    environment.remove("ETHERCAT_HOST")
    return environment


class SimulationConsole(QtWidgets.QWidget):
    def __init__(self, command, working_directory, parent=None, detached_online_tools=False,
                 stop_with_interrupt=False, ioc_name="SIMIOC", ioc_shell=True):
        super().__init__(parent)
        self.command = list(command)
        self.working_directory = str(working_directory)
        self.stop_requested = False
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending_escape = ""
        self.detached_online_tools = detached_online_tools
        self.stop_with_interrupt = stop_with_interrupt
        self.ioc_name = ioc_name
        self.ioc_shell = ioc_shell
        self.online_navigator = None
        self.saved_ca_environment = None
        layout = QtWidgets.QVBoxLayout(self)
        self.launch_header = QtWidgets.QFrame()
        self.launch_header.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.launch_header.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        header_layout = QtWidgets.QVBoxLayout(self.launch_header)
        image = self.command[self.command.index("--image") + 1] if "--image" in self.command else "custom launcher"
        models = "enabled" if "--hardware-sim" in self.command else "disabled"
        if stop_with_interrupt:
            title_text = f"Project simulation — {image} — coupled plant"
        else:
            title_text = f"Simulation: SIMIOC — {image} — Python hardware models {models}"
        self.launch_title = QtWidgets.QLabel(title_text)
        header_layout.addWidget(self.launch_title)
        command_text = shlex.join(self.command)
        self.command_display = QtWidgets.QLineEdit(command_text)
        self.command_display.setReadOnly(True)
        self.command_display.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.command_display.setToolTip(command_text)
        self.command_display.setCursorPosition(0)
        header_layout.addWidget(self.command_display)
        self.status = QtWidgets.QLabel("Ready")
        header_layout.addWidget(self.status)
        self.launch_header.setFixedHeight(self.launch_header.sizeHint().height())
        layout.addWidget(self.launch_header)
        self.console_tabs = QtWidgets.QTabWidget()
        ioc_page = QtWidgets.QWidget()
        ioc_layout = QtWidgets.QVBoxLayout(ioc_page)
        ioc_layout.setContentsMargins(0, 0, 0, 0)
        output_title = ("Project launcher — IOC and plant output" if stop_with_interrupt
                        else "IOC shell — output and simulation logs")
        ioc_layout.addWidget(QtWidgets.QLabel(output_title))
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.output.setMaximumBlockCount(10000)
        ioc_layout.addWidget(self.output, 1)
        row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("IOC shell command (for example: dbl) — press Enter to send")
        self.input.setAccessibleName("IOC shell command")
        self.input.setVisible(ioc_shell)
        self.input.returnPressed.connect(self.send_command)
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.send_command)
        self.send_button.setVisible(ioc_shell)
        self.stop_button = QtWidgets.QPushButton("Stop Simulation")
        self.stop_button.clicked.connect(self.request_stop)
        self.online_button = QtWidgets.QPushButton("Open Online Tools")
        self.online_button.clicked.connect(self.open_online_tools)
        self.online_button.setVisible(bool(ioc_name))
        row.addWidget(self.input, 1)
        row.addWidget(self.send_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.online_button)
        ioc_layout.addLayout(row)
        self.console_tabs.addTab(ioc_page, "Launcher output" if stop_with_interrupt else "IOC shell")
        self.terminal = None
        if not stop_with_interrupt:
            self.terminal = ContainerTerminal("ecmc-engineering-simulation", self)
            self.console_tabs.addTab(self.terminal, "Terminal")
        self.online_page = QtWidgets.QWidget()
        online_layout = QtWidgets.QVBoxLayout(self.online_page)
        if detached_online_tools:
            note = QtWidgets.QLabel(
                "Online tools are configured to open in a separate window. Start the IOC, then click Open Online Tools.")
            note.setWordWrap(True)
            online_layout.addWidget(note)
            online_layout.addStretch(1)
        else:
            self.online_placeholder = QtWidgets.QLabel(
                "Start the simulation, then open this tab to load the IOC navigator.")
            self.online_placeholder.setAlignment(QtCore.Qt.AlignCenter)
            online_layout.addWidget(self.online_placeholder, 1)
        if ioc_name:
            self.console_tabs.addTab(self.online_page, "Online tools")
        if self.console_tabs.count() == 1:
            self.console_tabs.tabBar().hide()
        self.console_tabs.currentChanged.connect(self.console_tab_changed)
        layout.addWidget(self.console_tabs, 1)
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.setWorkingDirectory(self.working_directory)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.started.connect(self.started)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.failed)
        self.set_controls(False)

    def is_running(self):
        return self.process.state() != QtCore.QProcess.NotRunning

    def set_controls(self, running):
        self.input.setEnabled(running and not self.stop_requested)
        self.send_button.setEnabled(running and not self.stop_requested)
        self.stop_button.setEnabled(running and not self.stop_requested)
        self.online_button.setEnabled(running and not self.stop_requested)

    def open_online_tools(self):
        if not self.detached_online_tools:
            self.console_tabs.setCurrentWidget(self.online_page)
            self.load_online_navigator()
            if self.online_navigator is not None:
                self.online_navigator.refresh()
            return
        launcher = Path(__file__).resolve().parents[1] / "ecmc_cfg_tool" / "start_ioc.sh"
        if not launcher.is_file():
            self.append_output("\nOnline tools are missing. Run git submodule update --init --recursive.\n")
            return
        process = QtCore.QProcess()
        process.setProgram("bash")
        process.setArguments([str(launcher), self.ioc_name])
        process.setWorkingDirectory(str(launcher.parent))
        process.setProcessEnvironment(simulation_online_environment())
        result = process.startDetached()
        started = result[0] if isinstance(result, tuple) else result
        if started:
            self.append_output("\nLaunched IOC navigator for {} on localhost.\n".format(self.ioc_name))
        else:
            self.append_output("\nCould not launch online tools: " + process.errorString() + "\n")

    def console_tab_changed(self, _index):
        if (self.console_tabs.currentWidget() is self.online_page
                and not self.detached_online_tools):
            self.load_online_navigator()

    def load_online_navigator(self):
        if self.online_navigator is not None:
            return
        if not hasattr(self, "process") or self.process.state() == QtCore.QProcess.NotRunning:
            return
        self.enable_simulation_ca_environment()
        try:
            from ecmc_ioc_navigator import IocNavigator
            self.online_navigator = IocNavigator(self.ioc_name)
            self.online_navigator.setWindowFlags(QtCore.Qt.Widget)
            layout = self.online_page.layout()
            layout.removeWidget(self.online_placeholder)
            self.online_placeholder.deleteLater()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.online_navigator)
        except (ImportError, RuntimeError) as error:
            self.online_placeholder.setText("Could not load embedded online tools: " + str(error))
            self.restore_ca_environment()

    def enable_simulation_ca_environment(self):
        names = ("EPICS_CA_AUTO_ADDR_LIST", "EPICS_CA_ADDR_LIST",
                 "EPICS_CA_SERVER_PORT", "EPICS_CA_REPEATER_PORT", "ETHERCAT_HOST")
        self.saved_ca_environment = {name: os.environ.get(name) for name in names}
        os.environ.update({
            "EPICS_CA_AUTO_ADDR_LIST": "NO", "EPICS_CA_ADDR_LIST": "127.0.0.1",
            "EPICS_CA_SERVER_PORT": "5064", "EPICS_CA_REPEATER_PORT": "5065",
        })
        os.environ.pop("ETHERCAT_HOST", None)

    def restore_ca_environment(self):
        if self.saved_ca_environment is None:
            return
        for name, value in self.saved_ca_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.saved_ca_environment = None

    def start(self):
        if self.is_running():
            return
        self.stop_requested = False
        self.decoder.reset()
        self.pending_escape = ""
        self.status.setText("Starting simulation…")
        self.process.start(self.command[0], self.command[1:])

    def started(self):
        self.status.setText("Launcher running — see output for IOC startup status")
        if self.stop_requested:
            self.stop_requested = False
            self.request_stop()
        else:
            self.set_controls(True)

    def append_output(self, text):
        cursor = self.output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def read_output(self):
        self.append_process_output(self.decoder.decode(bytes(self.process.readAllStandardOutput())))

    def append_process_output(self, text, final=False):
        """Remove ANSI CSI formatting, including sequences split across reads."""
        text = self.pending_escape + text
        self.pending_escape = ""
        incomplete = re.search(r"\x1b(?:\[[0-?]*[ -/]*)?$", text)
        if incomplete:
            if not final:
                self.pending_escape = incomplete.group()
            text = text[:incomplete.start()]
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        self.append_output(text)

    def send_command(self):
        command = self.input.text().strip()
        if command and self.process.state() == QtCore.QProcess.Running and not self.stop_requested:
            self.process.write((command + "\n").encode("utf-8"))
            self.append_output("\n> " + command + "\n")
            self.input.clear()

    def request_stop(self):
        if not self.is_running() or self.stop_requested:
            return
        self.stop_requested = True
        if self.process.state() == QtCore.QProcess.Running and self.stop_with_interrupt:
            try:
                os.kill(self.process.processId(), signal.SIGINT)
            except (OSError, TypeError, ValueError):
                self.process.terminate()
        elif self.process.state() == QtCore.QProcess.Running:
            self.process.write(b"exit\n")
            self.process.closeWriteChannel()
        self.set_controls(False)
        self.status.setText("Stopping simulation — waiting for the IOC and hardware models to exit…")

    def finished(self, code, exit_status):
        self.read_output()
        self.append_process_output(self.decoder.decode(b"", final=True), final=True)
        self.set_controls(False)
        if self.terminal is not None:
            self.terminal.disconnect_terminal()
        self.restore_ca_environment()
        if exit_status == QtCore.QProcess.NormalExit and code == 0:
            self.status.setText("Simulation stopped")
        else:
            self.status.setText("Simulation exited with error {} — see output".format(code))

    def failed(self, error):
        self.append_output("\nLauncher error: " + self.process.errorString() + "\n")
        if error == QtCore.QProcess.FailedToStart:
            self.set_controls(False)
            self.status.setText("Could not start simulation")
            self.restore_ca_environment()

    def closeEvent(self, event):
        if self.is_running():
            self.request_stop()
            event.ignore()
        else:
            self.restore_ca_environment()
            event.accept()


class ContainerTerminal(QtWidgets.QWidget):
    """A simple command shell in the running simulation container."""
    def __init__(self, container_name, parent=None):
        super().__init__(parent)
        self.container_name = container_name
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.note = QtWidgets.QLabel(
            "Connect after the simulation container has started. During the first build, wait for module compilation to finish.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.output.setMaximumBlockCount(10000)
        layout.addWidget(self.output, 1)
        row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Shell command — press Enter to send")
        self.input.returnPressed.connect(self.send_command)
        self.connect_button = QtWidgets.QPushButton("Connect Terminal")
        self.connect_button.clicked.connect(self.connect_terminal)
        row.addWidget(self.input, 1)
        row.addWidget(self.connect_button)
        layout.addLayout(row)
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.started.connect(lambda: self.set_connected(True))
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.failed)
        self.set_connected(False)

    def set_connected(self, connected):
        self.input.setEnabled(connected)
        self.connect_button.setText("Disconnect" if connected else "Connect Terminal")

    def append_output(self, text):
        cursor = self.output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def connect_terminal(self):
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.disconnect_terminal()
            return
        self.append_output("Connecting to the simulation container...\n")
        self.process.start("podman", ["exec", "-i", self.container_name,
                                      "/bin/bash", "--noprofile", "--norc"])

    def disconnect_terminal(self):
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.process.write(b"exit\n")
            self.process.closeWriteChannel()

    def send_command(self):
        command = self.input.text().strip()
        if command and self.process.state() == QtCore.QProcess.Running:
            self.process.write((command + "\n").encode("utf-8"))
            self.append_output("$ " + command + "\n")
            self.input.clear()

    def read_output(self):
        self.append_output(bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace"))

    def finished(self, code, _exit_status):
        self.read_output()
        self.set_connected(False)
        if code:
            self.append_output(
                "Terminal could not connect or exited. Wait until the IOC container is running, then try again.\n")

    def failed(self, _error):
        self.append_output("Could not start terminal: " + self.process.errorString() + "\n")
        self.set_connected(False)
