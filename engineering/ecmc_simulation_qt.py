"""Run the project's Podman simulation launcher in a Qt console."""
import codecs
import os
from pathlib import Path
import re
import shlex

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


class SimulationOptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start simulation")
        layout = QtWidgets.QFormLayout(self)
        self.image = QtWidgets.QLineEdit("ecmc-dev:local")
        layout.addRow("Container image", self.image)
        layout.addRow("IOC name", QtWidgets.QLabel("SIMIOC"))
        self.local_sources = QtWidgets.QCheckBox("Use local ecmc, ecmccfg and ecmccomp submodules")
        self.local_sources.setChecked(True)
        layout.addRow(self.local_sources)
        self.hardware_sim = QtWidgets.QCheckBox("Enable Python hardware simulation")
        self.hardware_sim.setChecked(True)
        layout.addRow(self.hardware_sim)
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


def simulation_online_environment():
    environment = QtCore.QProcessEnvironment.systemEnvironment()
    environment.insert("EPICS_CA_AUTO_ADDR_LIST", "NO")
    environment.insert("EPICS_CA_ADDR_LIST", "127.0.0.1")
    environment.insert("EPICS_CA_SERVER_PORT", "5064")
    environment.insert("EPICS_CA_REPEATER_PORT", "5065")
    environment.remove("ETHERCAT_HOST")
    return environment


class SimulationConsole(QtWidgets.QWidget):
    def __init__(self, command, working_directory, parent=None, detached_online_tools=False):
        super().__init__(parent)
        self.command = list(command)
        self.working_directory = str(working_directory)
        self.stop_requested = False
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending_escape = ""
        self.detached_online_tools = detached_online_tools
        self.online_navigator = None
        self.saved_ca_environment = None
        layout = QtWidgets.QVBoxLayout(self)
        image = self.command[self.command.index("--image") + 1] if "--image" in self.command else "custom launcher"
        models = "enabled" if "--hardware-sim" in self.command else "disabled"
        title = QtWidgets.QLabel(f"Simulation: SIMIOC — {image} — Python hardware models {models}")
        title.setWordWrap(True)
        layout.addWidget(title)
        command_label = QtWidgets.QLabel(shlex.join(self.command))
        command_label.setWordWrap(True)
        command_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(command_label)
        self.status = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status)
        self.console_tabs = QtWidgets.QTabWidget()
        ioc_page = QtWidgets.QWidget()
        ioc_layout = QtWidgets.QVBoxLayout(ioc_page)
        ioc_layout.setContentsMargins(0, 0, 0, 0)
        ioc_layout.addWidget(QtWidgets.QLabel("IOC shell — output and simulation logs"))
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.output.setMaximumBlockCount(10000)
        ioc_layout.addWidget(self.output, 1)
        row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("IOC shell command (for example: dbl) — press Enter to send")
        self.input.setAccessibleName("IOC shell command")
        self.input.returnPressed.connect(self.send_command)
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.send_command)
        self.stop_button = QtWidgets.QPushButton("Stop Simulation")
        self.stop_button.clicked.connect(self.request_stop)
        self.online_button = QtWidgets.QPushButton("Open Online Tools")
        self.online_button.clicked.connect(self.open_online_tools)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.online_button)
        ioc_layout.addLayout(row)
        self.console_tabs.addTab(ioc_page, "IOC shell")
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
        self.console_tabs.addTab(self.online_page, "Online tools")
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
        process.setArguments([str(launcher), "SIMIOC"])
        process.setWorkingDirectory(str(launcher.parent))
        process.setProcessEnvironment(simulation_online_environment())
        result = process.startDetached()
        started = result[0] if isinstance(result, tuple) else result
        if started:
            self.append_output("\nLaunched IOC navigator for SIMIOC on localhost.\n")
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
            self.online_navigator = IocNavigator("SIMIOC")
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
        if self.process.state() == QtCore.QProcess.Running:
            self.process.write(b"exit\n")
            self.process.closeWriteChannel()
        self.set_controls(False)
        self.status.setText("Stopping simulation — waiting for the IOC and hardware models to exit…")

    def finished(self, code, exit_status):
        self.read_output()
        self.append_process_output(self.decoder.decode(b"", final=True), final=True)
        self.set_controls(False)
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
