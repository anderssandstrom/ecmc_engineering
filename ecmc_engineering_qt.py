#!/usr/bin/env python3
"""Qt project workspace, sharing the existing startup parser and validator."""
import argparse
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent / "ecmc_cfg_tool"
sys.path.insert(0, str(TOOLS))
from qt_compat import QtCore, QtGui, QtWidgets
import ecmc_engineering_studio as core
from ecmc_object_editor_qt import ObjectEditingMixin, OBJECT_ROLE, ROW_ROLE
from ecmc_ioc_project import create_ioc_project, parameters_path
from ecmc_simulation_qt import SimulationConsole, SimulationOptionsDialog, simulation_command


class EngineeringWindow(ObjectEditingMixin, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.startup = None
        self.current = None
        self.buffers = {}
        self.saved = {}
        self.model = None
        self.copied_objects = ""
        self.object_items = {}
        self.undo_objects, self.redo_objects = [], []
        self.online_prefix = "IOC:ECMC"
        self.online_host = ""
        self.simulation_console = None
        self.inventory = core._build_repository_inventory(core._find_ecmccfg_root())
        self.setWindowTitle("ecmc Engineering Studio — Qt preview")
        self.resize(1250, 850)
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        project = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(project)
        self.tabs.addTab(project, "Project")
        toolbar = QtWidgets.QHBoxLayout()
        new_button = QtWidgets.QToolButton()
        new_button.setText("New IOC Project…")
        new_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        new_button.clicked.connect(self.new_project)
        new_menu = QtWidgets.QMenu(new_button)
        new_menu.addAction("New IOC Project…", self.new_project)
        new_menu.addAction("New Startup File…", self.new_startup_file)
        new_button.setMenu(new_menu)
        toolbar.addWidget(new_button)
        open_button = QtWidgets.QToolButton()
        open_button.setText("Open Project…")
        open_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        open_button.clicked.connect(self.open_project_dialog)
        open_menu = QtWidgets.QMenu(open_button)
        open_menu.addAction("Open Project Directory…", self.open_project_dialog)
        open_menu.addAction("Open Startup File…", self.open_dialog)
        open_button.setMenu(open_menu)
        toolbar.addWidget(open_button)
        for label, callback in [("Save All", self.save_all), ("Validate", self.validate)]:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        add = QtWidgets.QToolButton()
        add.setText("Add")
        add.setMinimumWidth(70)
        add.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(add)
        self.populate_add_menu(menu)
        add.setMenu(menu)
        toolbar.addWidget(add)
        toolbar.addStretch()
        simulate = QtWidgets.QPushButton("Simulate")
        simulate.setToolTip("Choose simulation options and run the current project as SIMIOC")
        simulate.clicked.connect(self.simulate_project)
        toolbar.addWidget(simulate)
        online = QtWidgets.QPushButton("Connect / Demo…")
        online.clicked.connect(self.open_online)
        toolbar.addWidget(online)
        layout.addLayout(toolbar)
        self.path_label = QtWidgets.QLabel("Create a new IOC project or open an existing startup file to begin.")
        layout.addWidget(self.path_label)
        vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(vertical, 1)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        vertical.addWidget(split)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Project objects"])
        self.tree.currentItemChanged.connect(lambda item, previous: self.select_item(item, 0) if item is not None else None)
        self.tree.itemDoubleClicked.connect(lambda item, column: self.edit_object() if self.row_type(item) == "object" else None)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.context_menu)
        navigation = QtWidgets.QWidget()
        nav_layout = QtWidgets.QVBoxLayout(navigation)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter objects…")
        self.filter.textChanged.connect(self.filter_objects)
        nav_layout.addWidget(self.filter)
        nav_layout.addWidget(self.tree)
        split.addWidget(navigation)
        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.editor.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.editor.setPlaceholderText("New IOC Project → enter IOC name → choose parent directory → Add Require → Add Slave")
        self.editor.setEnabled(False)
        self.editor.textChanged.connect(self.edit_changed)
        split.addWidget(self.editor)
        self.properties = QtWidgets.QTabWidget()
        self.details = QtWidgets.QTreeWidget()
        self.details.setHeaderLabels(["Field", "Value"])
        self.details.setColumnWidth(0, 140)
        self.details.itemDoubleClicked.connect(lambda *_: self.edit_object())
        self.properties.addTab(self.details, "Values")
        self.context = QtWidgets.QPlainTextEdit()
        self.context.setReadOnly(True)
        self.properties.addTab(self.context, "Context / Help")
        split.addWidget(self.properties)
        split.setSizes([280, 630, 320])
        self.problems = QtWidgets.QTreeWidget()
        self.problems.setHeaderLabels(["Severity", "Location", "Message"])
        self.problems.itemClicked.connect(self.select_item)
        self.problems.setColumnWidth(0, 85)
        self.problems.setColumnWidth(1, 240)
        vertical.addWidget(self.problems)
        vertical.setSizes([600, 150])
        self.rebuild_timer = QtCore.QTimer(self)
        self.rebuild_timer.setSingleShot(True)
        self.rebuild_timer.setInterval(400)
        self.rebuild_timer.timeout.connect(self.rebuild)
        self.statusBar().showMessage("Ready — new projects start with Add → Require")
        # Tree shortcuts only: typing in the editor retains normal text shortcuts.
        action_class = getattr(QtGui, "QAction", None) or QtWidgets.QAction
        for title, shortcut, callback in [("Edit Object", "Return", self.edit_object),
                                           ("Copy Objects", QtGui.QKeySequence.Copy, self.copy_objects),
                                           ("Paste Objects", QtGui.QKeySequence.Paste, lambda: self.insert_command(self.copied_objects) if self.copied_objects else None),
                                           ("Remove Objects", "Delete", self.delete_objects),
                                           ("Move Up", "Alt+Up", lambda: self.move_object(-1)),
                                           ("Move Down", "Alt+Down", lambda: self.move_object(1))]:
            action = action_class(title, self.tree)
            action.setShortcut(shortcut)
            action.setShortcutContext(QtCore.Qt.WidgetShortcut)
            action.triggered.connect(lambda checked=False, callback=callback: callback())
            self.tree.addAction(action)

    def error(self, text):
        QtWidgets.QMessageBox.critical(self, "Engineering Studio", str(text))

    def edit_changed(self):
        if self.current:
            self.buffers[self.current] = self.editor.toPlainText()
            dirty = self.buffers[self.current] != self.saved[self.current]
            self.path_label.setText(str(self.current) + (" • unsaved" if dirty else ""))
            self.problems.clear()
            self.rebuild_timer.start()

    def show_file(self, path, line=1):
        path = Path(path).resolve()
        if path not in self.buffers:
            try:
                self.buffers[path] = self.saved[path] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                self.error(exc)
                return
        if self.current != path or self.editor.toPlainText() != self.buffers[path]:
            self.editor.blockSignals(True)
            self.editor.setPlainText(self.buffers[path])
            self.editor.blockSignals(False)
        self.current = path
        self.editor.setEnabled(True)
        self.path_label.setText(str(path) + (" • unsaved" if self.buffers[path] != self.saved[path] else ""))
        block = self.editor.document().findBlockByNumber(max(0, line - 1))
        if block.isValid():
            self.editor.setTextCursor(QtGui.QTextCursor(block))
            self.editor.centerCursor()

    def select_item(self, item, column):
        location = item.data(0, QtCore.Qt.UserRole)
        if location:
            self.show_file(*location)
        self.show_properties(self.object_for_item(item))

    def show_properties(self, obj):
        self.details.clear()
        self.context.clear()
        if not obj:
            return
        try:
            values = self.object_values(obj)
        except (ValueError, OSError):
            values = {}
        for label, pairs in [("Command values", values.items()), ("Resolved", obj.details), ("Linked file macros", obj.linked_file_details)]:
            group = QtWidgets.QTreeWidgetItem(self.details, [label])
            group.setFirstColumnSpanned(True)
            for key, value in pairs:
                row = QtWidgets.QTreeWidgetItem(group, [str(key), str(value)])
                row.setToolTip(1, str(value))
            group.setExpanded(True)
        from ecmc_project_commands import OBJECT_TYPES
        script = OBJECT_TYPES.get(obj.kind, ("", "", ""))[1]
        if obj.kind == "require" and values.get("MODULE") == "ecmccfg":
            script = "startup.cmd"
        paths = self.inventory.module_scripts.get(script, [])
        help_text = "{}\n{}:{}\n\n{}".format(obj.title, obj.source, obj.line, obj.summary)
        if paths:
            try:
                docs = []
                for line in paths[0].read_text(encoding="utf-8").splitlines():
                    if line.strip() and not line.lstrip().startswith("#"):
                        break
                    docs.append(line)
                help_text += "\n\n" + str(paths[0]) + "\n" + "\n".join(docs)
            except OSError:
                pass
        self.context.setPlainText(help_text)

    def select_location(self, path, line):
        for (source, number, _kind), item in self.object_items.items():
            if source == Path(path).resolve() and number == line:
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)
                self.show_properties(self.object_for_item(item))
                return

    def filter_objects(self, *_):
        text = self.filter.text().strip().lower()
        def visit(item, ancestor_matches=False):
            matches = not text or text in (item.text(0) + " " + item.toolTip(0)).lower()
            child_matches = [visit(item.child(i), ancestor_matches or matches) for i in range(item.childCount())]
            visible = ancestor_matches or matches or any(child_matches)
            item.setHidden(not visible)
            if text and any(child_matches):
                item.setExpanded(True)
            return visible
        for index in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(index))

    def confirm_switch(self):
        if not any(text != self.saved[path] for path, text in self.buffers.items()):
            return True
        choice = QtWidgets.QMessageBox.question(
            self, "Unsaved project", "Save changes before leaving this project?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        return self.save_all() if choice == QtWidgets.QMessageBox.Save else choice == QtWidgets.QMessageBox.Discard

    def load_project(self, path, confirm=True):
        path = Path(path).resolve()
        if confirm and not self.confirm_switch():
            return
        try:
            text = path.read_text(encoding="utf-8")
            parameters = parameters_path(path)
            parameter_text = parameters.read_text(encoding="utf-8") if parameters and parameters.exists() else None
        except (OSError, UnicodeError) as exc:
            self.error(exc)
            return
        self.rebuild_timer.stop()
        self.startup = path
        self.undo_objects.clear()
        self.redo_objects.clear()
        self.buffers = {path: text}
        if parameter_text is not None:
            self.buffers[parameters] = parameter_text
        self.saved = dict(self.buffers)
        self.online_host = ""
        self.problems.clear()
        self.details.clear()
        self.context.clear()
        self.show_file(path)
        self.rebuild()

    def open_project_dialog(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open IOC project directory", str(self.startup.parent) if self.startup else str(Path.cwd()),
        )
        if directory:
            self.load_project_directory(directory)

    def load_project_directory(self, directory):
        directory = Path(directory).expanduser().resolve()
        preferred = directory / (directory.name + "_startup.cmd")
        if preferred.is_file():
            self.load_project(preferred)
            return
        candidates = sorted(path for path in directory.glob("*_startup.cmd") if path.is_file())
        conventional = directory / "startup.cmd"
        if conventional.is_file():
            candidates.append(conventional)
        if not candidates:
            self.error("No startup file found in {}.\nExpected {} or startup.cmd. Use Open Startup File for another filename.".format(directory, preferred.name))
            return
        if len(candidates) == 1:
            self.load_project(candidates[0])
            return
        filename, accepted = QtWidgets.QInputDialog.getItem(
            self, "Choose startup file", "This directory contains multiple startup files:",
            [path.name for path in candidates], 0, False,
        )
        if accepted:
            self.load_project(directory / filename)

    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open startup", "", "Startup (*.cmd *.iocsh *.script);;All files (*)")
        if path:
            self.load_project(path)

    def new_project(self):
        name_field = QtWidgets.QLineEdit()
        name_field.setPlaceholderText("MTEST04-MTN-CPP-LOGIC")
        values = self.form("New IOC Project", [("IOC name", name_field)])
        if values is None:
            return
        name = values[0].strip()
        # Validate before asking for the destination directory.
        import re
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", name):
            self.error("Use an IOC name containing letters, digits, underscores or hyphens.")
            return
        parent = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose parent directory for " + name,
            str(self.startup.parent.parent) if self.startup else str(Path.cwd()),
        )
        if not parent or not self.confirm_switch():
            return
        try:
            path = create_ioc_project(parent, name)
        except (OSError, ValueError) as exc:
            self.error("Cannot create IOC project: " + str(exc))
            return
        self.load_project(path, confirm=False)
        self.online_prefix = name
        self.online_host = ""
        self.statusBar().showMessage("Created {}. Edit its parameters file or use Add to build the startup.".format(name))

    def new_startup_file(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "New startup", "startup.cmd", "Startup (*.cmd)")
        if not path:
            return
        if not self.confirm_switch():
            return
        try:
            with Path(path).open("x", encoding="utf-8") as stream:
                stream.write("# ecmc startup configuration\n")
        except OSError as exc:
            self.error(exc)
            return
        self.load_project(path, confirm=False)

    def save_all(self):
        try:
            # Check every destination before writing any of this save operation.
            for path, text in self.buffers.items():
                if text != self.saved[path]:
                    if (self.saved[path] is None and path.exists()) or (self.saved[path] is not None and path.read_text(encoding="utf-8") != self.saved[path]):
                        self.error("File changed outside Studio; resolve before saving:\n" + str(path))
                        return False
            for path, text in self.buffers.items():
                if text != self.saved[path]:
                    if self.saved[path] is None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8")
                    self.saved[path] = text
        except (OSError, UnicodeError) as exc:
            self.error(exc)
            return False
        if self.current:
            self.path_label.setText(str(self.current))
        self.rebuild()
        self.statusBar().showMessage("Project saved", 4000)
        return True

    def rebuild(self):
        if not self.startup:
            return
        try:
            model = core.build_startup_tree(self.startup, self.buffers[self.startup], self.inventory, self.buffers)
        except Exception as exc:
            self.statusBar().showMessage("Cannot parse project: " + str(exc))
            return
        selected = self.object_for_item()
        file_selection = self.tree.currentItem().data(0, QtCore.Qt.UserRole)[0] if self.row_type() in {"file", "parameters"} else None
        selected_key = (selected.source.resolve(), selected.line, selected.kind) if selected else None
        expanded = {key for key, item in self.object_items.items() if item.isExpanded()}
        self.model = model
        self.object_items = {}
        self.tree.blockSignals(True)
        self.tree.clear()
        for node in model.files:
            parent = QtWidgets.QTreeWidgetItem(self.tree, [node.path.name])
            parent.setToolTip(0, str(node.path))
            parent.setData(0, QtCore.Qt.UserRole, (node.path, 1))
            parent.setData(0, ROW_ROLE, "file")
            if node.path == file_selection:
                self.tree.setCurrentItem(parent)
            if node.path in self.buffers and self.buffers[node.path] != self.saved[node.path]:
                parent.setText(0, node.path.name + " *")
            slaves, axes, plcs = {}, {}, {}
            for obj in node.objects:
                owner = parent
                if obj.kind in {"component", "ecsdo", "ecdataitem", "slave_config"}:
                    owner = slaves.get(obj.parent_slave_id, parent)
                elif obj.kind == "encoder":
                    owner = axes.get(obj.parent_axis_line, parent)
                elif obj.kind in {"plcvar_analog", "plcvar_binary"}:
                    owner = plcs.get(obj.parent_plc_id, parent)
                item = QtWidgets.QTreeWidgetItem(owner, [obj.title])
                item.setToolTip(0, obj.summary)
                item.setData(0, QtCore.Qt.UserRole, (obj.source, obj.line))
                item.setData(0, OBJECT_ROLE, obj)
                item.setData(0, ROW_ROLE, "object")
                key = (obj.source.resolve(), obj.line, obj.kind)
                self.object_items[key] = item
                item.setExpanded(key in expanded)
                if obj.kind == "slave":
                    slaves[obj.slave_id] = item
                elif obj.kind == "axis":
                    axes[obj.line] = item
                elif obj.kind == "plc":
                    plcs[core._parse_int_value(dict(obj.details).get("PLC_ID", ""))] = item
                if obj.linked_file:
                    linked = QtWidgets.QTreeWidgetItem(item, [obj.linked_file.name])
                    linked.setData(0, QtCore.Qt.UserRole, (obj.linked_file, 1))
                    linked.setData(0, OBJECT_ROLE, obj)
                    linked.setData(0, ROW_ROLE, "linked")
                    if obj.linked_file.resolve() in self.buffers and self.buffers[obj.linked_file.resolve()] != self.saved[obj.linked_file.resolve()]:
                        linked.setText(0, obj.linked_file.name + " *")
                    elif not obj.linked_file.exists():
                        linked.setText(0, obj.linked_file.name + " (missing)")
            parent.setExpanded(True)
        parameters = parameters_path(self.startup)
        if parameters in self.buffers:
            dirty = self.buffers[parameters] != self.saved[parameters]
            item = QtWidgets.QTreeWidgetItem(self.tree, [parameters.name + (" *" if dirty else "")])
            item.setToolTip(0, str(parameters))
            item.setData(0, QtCore.Qt.UserRole, (parameters, 1))
            item.setData(0, ROW_ROLE, "parameters")
            if file_selection == parameters:
                self.tree.setCurrentItem(item)
        if selected_key in self.object_items:
            self.tree.setCurrentItem(self.object_items[selected_key])
            self.show_properties(self.object_for_item())
        self.tree.blockSignals(False)
        self.filter_objects()

    def validate(self):
        if not self.startup:
            return
        self.problems.clear()
        try:
            result = core.validate_project(self.startup, self.buffers[self.startup], self.inventory, self.buffers)
        except Exception as exc:
            self.error(exc)
            return
        for issue in result.issues:
            item = QtWidgets.QTreeWidgetItem(self.problems, [issue.severity, "{}:{}".format(issue.source.name, issue.line), issue.message])
            item.setData(0, QtCore.Qt.UserRole, (issue.source, issue.line))
        self.statusBar().showMessage("Validation: {} issue(s)".format(len(result.issues)))

    def form(self, title, fields):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        layout = QtWidgets.QFormLayout(dialog)
        for label, widget in fields:
            layout.addRow(label, widget)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        try:
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return None
            # Read children while their owning dialog is still alive. Callers
            # receive plain values, never widgets whose parent may be deleted.
            values = []
            for _label, widget in fields:
                if isinstance(widget, QtWidgets.QCheckBox):
                    values.append(widget.isChecked())
                elif isinstance(widget, QtWidgets.QComboBox):
                    values.append(widget.currentText())
                elif isinstance(widget, QtWidgets.QSpinBox):
                    values.append(widget.value())
                else:
                    values.append(widget.text())
            return values
        finally:
            dialog.deleteLater()

    def append_command(self, command):
        self.insert_command(command)

    def add_require(self):
        self.add_object("require")

    def add_slave(self):
        self.add_object("slave")

    def simulate_project(self):
        if self.simulation_console is not None and self.simulation_console.is_running():
            self.tabs.setCurrentWidget(self.simulation_console)
            self.statusBar().showMessage("A simulation is already running. Stop it before starting another project.")
            return
        if self.startup is None:
            self.error("Open or create an IOC project before starting a simulation.")
            return
        try:
            command = simulation_command(self.startup, Path(__file__).resolve().parent)
        except ValueError as exc:
            self.error(exc)
            return
        if any(text != self.saved[path] for path, text in self.buffers.items()):
            choice = QtWidgets.QMessageBox.question(
                self, "Save before simulation", "Save all project changes before running the simulation?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Save,
            )
            if choice != QtWidgets.QMessageBox.Save or not self.save_all():
                return
        options = SimulationOptionsDialog(self)
        if options.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            command = simulation_command(self.startup, Path(__file__).resolve().parent,
                                         options.image.text(), options.hardware_sim.isChecked())
        except ValueError as exc:
            self.error(exc)
            return
        console = SimulationConsole(command, self.startup.parent, self)
        if self.simulation_console is not None:
            old = self.simulation_console
            self.tabs.removeTab(self.tabs.indexOf(old))
            old.deleteLater()
        self.simulation_console = console
        self.tabs.setCurrentIndex(self.tabs.addTab(console, "Simulation — SIMIOC"))
        close_button = QtWidgets.QToolButton(self.tabs)
        close_button.setText("×")
        close_button.setToolTip("Close simulation tab (stops a running IOC)")
        close_button.setAccessibleName("Close simulation tab")
        close_button.clicked.connect(self.close_simulation_tab)
        self.tabs.tabBar().setTabButton(self.tabs.indexOf(console), QtWidgets.QTabBar.RightSide, close_button)
        console.start()

    def close_simulation_tab(self):
        console = self.simulation_console
        if console is None:
            return
        if console.is_running():
            if not getattr(console, "close_when_stopped", False):
                console.close_when_stopped = True
                console.process.stateChanged.connect(
                    lambda state: QtCore.QTimer.singleShot(0, self.close_simulation_tab)
                    if state == QtCore.QProcess.NotRunning else None)
            console.request_stop()
            return
        self.tabs.removeTab(self.tabs.indexOf(console))
        self.simulation_console = None
        console.deleteLater()

    def project_online_host(self):
        """Prefer the current parameters buffer over the session's manual host."""
        if self.startup:
            parameters = parameters_path(self.startup)
            if parameters and (parameters in self.buffers or parameters.exists()):
                for entry in core._parse_simple_yaml_paths(self.source_text(parameters)):
                    if entry.path == "ioc_host":
                        raw = (entry.value or "").strip()
                        if raw.lower() in {"", "null", "~"}:
                            break
                        host = core._normalize_value(raw).strip()
                        if host:
                            return host
        return self.online_host

    def open_online(self):
        prefix = QtWidgets.QLineEdit(self.online_prefix)
        try:
            default_host = self.project_online_host()
        except (OSError, UnicodeError) as exc:
            self.error(exc)
            return
        host = QtWidgets.QLineEdit(default_host)
        demo = QtWidgets.QCheckBox("Use offline sample objects")
        values = self.form("Open IOC workspace", [("IOC prefix", prefix), ("SSH host (optional)", host), ("Demo", demo)])
        if values is None:
            return
        prefix_text, host_text, use_demo = values
        prefix_text, host_text = prefix_text.strip(), host_text.strip()
        if not prefix_text:
            self.error("Enter an IOC prefix.")
            return
        self.online_prefix, self.online_host = prefix_text, host_text
        try:
            from ecmc_ioc_navigator import IocNavigator
            navigator = IocNavigator(prefix=prefix_text, ssh_host=host_text, demo=use_demo)
        except Exception as exc:
            self.error(exc)
            return
        self.tabs.setCurrentIndex(self.tabs.addTab(navigator, "Demo IOC" if use_demo else prefix_text))

    def closeEvent(self, event):
        if self.confirm_switch():
            # Embedded QMainWindows must receive their own close events (motion
            # and measurement tools perform cleanup in those handlers).
            for index in range(1, self.tabs.count()):
                tab = self.tabs.widget(index)
                for tool in list(getattr(tab, "_embedded_tools", [])):
                    if not tool.close():
                        event.ignore()
                        return
                if not tab.close():
                    self.tabs.setCurrentWidget(tab)
                    event.ignore()
                    return
            event.accept()
        else:
            event.ignore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup", type=Path)
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    window = EngineeringWindow()
    if args.startup:
        window.load_project(args.startup)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
