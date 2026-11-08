"""Run the project's Podman simulation launcher in a Qt console."""
import codecs
from pathlib import Path
import re
import shlex

from qt_compat import QtCore, QtGui, QtWidgets


def simulation_command(startup, repository, image="ecmc-dev:portable", hardware_sim=True):
    startup = Path(startup).resolve()
    launcher = Path(repository).resolve() / "ecmc_podman" / "run-sim.sh"
    if not launcher.is_file():
        raise ValueError("Simulation runtime is missing. Run git submodule update --init --recursive.")
    if not startup.is_file():
        raise ValueError("The startup file must be saved before simulation.")
    if not image.strip():
        raise ValueError("Enter a simulation image.")
    return (["bash", str(launcher), "--image", image.strip(), "--ioc-name", "SIMIOC"]
            + (["--hardware-sim"] if hardware_sim else []) + [str(startup)])


class SimulationOptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start simulation")
        layout = QtWidgets.QFormLayout(self)
        self.image = QtWidgets.QLineEdit("ecmc-dev:portable")
        layout.addRow("Container image", self.image)
        layout.addRow("IOC name", QtWidgets.QLabel("SIMIOC"))
        self.hardware_sim = QtWidgets.QCheckBox("Enable Python hardware simulation")
        self.hardware_sim.setChecked(True)
        layout.addRow(self.hardware_sim)
        hint = QtWidgets.QLabel("Disable hardware simulation to run the IOC without Python hardware models.")
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
    def __init__(self, command, working_directory, parent=None):
        super().__init__(parent)
        self.command = list(command)
        self.working_directory = str(working_directory)
        self.stop_requested = False
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending_escape = ""
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
        layout.addWidget(QtWidgets.QLabel("IOC shell — output and simulation logs"))
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.output.setMaximumBlockCount(10000)
        layout.addWidget(self.output, 1)
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
        layout.addLayout(row)
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
        launcher = Path(__file__).resolve().parent / "ecmc_cfg_tool" / "start_ioc.sh"
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
        if exit_status == QtCore.QProcess.NormalExit and code == 0:
            self.status.setText("Simulation stopped")
        else:
            self.status.setText("Simulation exited with error {} — see output".format(code))

    def failed(self, error):
        self.append_output("\nLauncher error: " + self.process.errorString() + "\n")
        if error == QtCore.QProcess.FailedToStart:
            self.set_controls(False)
            self.status.setText("Could not start simulation")

    def closeEvent(self, event):
        if self.is_running():
            self.request_stop()
            event.ignore()
        else:
            event.accept()
