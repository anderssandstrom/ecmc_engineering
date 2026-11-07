"""Qt structured object editor and project-tree actions."""
from pathlib import Path
import re
from functools import wraps

from qt_compat import QtCore, QtGui, QtWidgets
import ecmc_engineering_studio as core
from ecmc_project_commands import OBJECT_TYPES, CHILD_TYPES, read_values, render_command, replace_line

OBJECT_ROLE = int(QtCore.Qt.UserRole) + 1
ROW_ROLE = int(QtCore.Qt.UserRole) + 2


def project_action(method):
    """Report file/parse failures inside Qt slots instead of aborting the app."""
    @wraps(method)
    def run(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except (OSError, UnicodeError, ValueError) as exc:
            self.error(exc)
    return run


class ObjectDialog(QtWidgets.QDialog):
    def __init__(self, window, kind, initial, editing=False):
        super().__init__(window)
        self.window, self.kind, self.initial = window, kind, initial
        self.command = None
        self.setWindowTitle(("Edit " if editing else "Add ") + OBJECT_TYPES[kind][0])
        self.resize(720, 560)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Edit command values below. Macro references are preserved. Blank optional fields are omitted."))
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Field / macro", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setColumnWidth(0, 180)
        layout.addWidget(self.table, 1)
        keys = list(dict.fromkeys(OBJECT_TYPES[kind][2].split() + list(initial)))
        script = OBJECT_TYPES[kind][1]
        spec = window.inventory.module_macro_specs.get(script)
        if spec:
            keys += sorted(spec.required - set(keys))
        for key in keys:
            self.add_row(key, initial.get(key, ""))
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add field / macro")
        add.clicked.connect(lambda: self.add_row("", ""))
        remove = QtWidgets.QPushButton("Remove field")
        remove.clicked.connect(self.remove_row)
        browse = QtWidgets.QPushButton("Choose file…")
        browse.clicked.connect(self.choose_file)
        row.addWidget(add)
        row.addWidget(remove)
        row.addWidget(browse)
        row.addStretch()
        layout.addLayout(row)
        self.support_label = QtWidgets.QLabel()
        self.support_label.setWordWrap(True)
        layout.addWidget(self.support_label)
        self.feedback = QtWidgets.QLabel()
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(100)
        layout.addWidget(self.preview)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept_values)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.table.itemChanged.connect(self.update_preview)
        self.update_preview()

    def add_row(self, key, value):
        blocked = self.table.blockSignals(True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
        choices = self.window.field_choices(key)
        if choices:
            widget = QtWidgets.QComboBox()
            widget.setEditable(True)
            widget.addItems([""] + choices)
            widget.setCurrentText(value)
            widget.completer().setFilterMode(QtCore.Qt.MatchContains)
            widget.currentTextChanged.connect(self.update_preview)
        else:
            widget = QtWidgets.QLineEdit(value)
            widget.textChanged.connect(self.update_preview)
        widget.setObjectName("field_" + key)
        self.table.setCellWidget(row, 1, widget)
        self.table.blockSignals(blocked)
        self.update_preview()

    def remove_row(self):
        if self.table.currentRow() >= 0:
            self.table.removeRow(self.table.currentRow())
            self.update_preview()

    def values(self):
        values = {}
        for row in range(self.table.rowCount()):
            cell = self.table.item(row, 0)
            key = cell.text().strip() if cell else ""
            widget = self.table.cellWidget(row, 1)
            if widget is None:
                continue
            value = widget.currentText() if isinstance(widget, QtWidgets.QComboBox) else widget.text()
            if not key and not value:
                continue
            if key in values:
                raise ValueError("Duplicate field: " + key)
            if value or key in self.initial:
                values[key] = value
        return values

    def choose_file(self):
        row = self.table.currentRow()
        if row < 0:
            self.feedback.setText("Select a file-valued field first (for example FILE or CONFIG).")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose linked file", str(self.window.startup.parent) if self.window.startup else "")
        if path:
            base = self.window.insertion_path().parent
            try:
                value = "./" + str(Path(path).relative_to(base))
            except ValueError:
                value = path
            widget = self.table.cellWidget(row, 1)
            if isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(value)
            else:
                widget.setText(value)

    def update_preview(self, *_):
        if not hasattr(self, "preview"):
            return
        try:
            values = self.values()
            self.preview.setPlainText(render_command(self.kind, values))
            self.feedback.setText("")
            if self.kind == "component":
                definition = self.window.inventory.component_definitions.get(values.get("COMP"))
                support_map = self.window.inventory.component_support.get(values.get("EC_COMP_TYPE"), {})
                support = support_map.get(definition.comp_type) if definition else None
                text = ""
                if support:
                    text = "Channels: {}. Supported component macros: {}".format(
                        support.channel_count or "not specified", ", ".join(sorted(support.supported_macros)) or "not specified")
                elif definition and support_map:
                    text = "This component is not supported by the selected hardware type."
                self.support_label.setText(text)
        except ValueError as exc:
            self.preview.clear()
            self.feedback.setText(str(exc))

    def accept_values(self):
        try:
            values = self.values()
            self.window.check_object_values(self.kind, values)
            self.command = render_command(self.kind, values)
        except ValueError as exc:
            self.feedback.setText(str(exc))
            return
        self.accept()


class ObjectEditingMixin:
    def object_for_item(self, item=None):
        item = item or self.tree.currentItem()
        return item.data(0, OBJECT_ROLE) if item else None

    def row_type(self, item=None):
        item = item or self.tree.currentItem()
        return item.data(0, ROW_ROLE) if item else ""

    def source_text(self, path):
        path = Path(path).resolve()
        if path not in self.buffers:
            self.buffers[path] = self.saved[path] = path.read_text(encoding="utf-8")
        return self.buffers[path]

    def insertion_path(self):
        item = self.tree.currentItem()
        if item and self.row_type(item) == "file":
            return item.data(0, QtCore.Qt.UserRole)[0]
        obj = self.object_for_item(item)
        return obj.source if obj else self.startup

    def object_values(self, obj):
        lines = self.source_text(obj.source).splitlines(True)
        if not 0 < obj.line <= len(lines):
            raise ValueError("The source changed; select the object again.")
        return read_values(obj.kind, lines[obj.line - 1])

    def ensure_fresh_objects(self):
        if self.rebuild_timer.isActive():
            self.rebuild_timer.stop()
            self.rebuild()

    def apply_text(self, path, text, line=1):
        path = Path(path).resolve()
        previous = self.source_text(path)
        if previous == text:
            return
        self.undo_objects.append((path, previous, text, line))
        self.redo_objects.clear()
        self.buffers[path] = text
        self.problems.clear()
        self.show_file(path, line)
        self.rebuild_timer.stop()
        self.rebuild()
        self.select_location(path, line)
        self.statusBar().showMessage("Changed {} — use Save All to write changes".format(path.name))

    @project_action
    def undo_object_change(self, redo=False):
        source = self.redo_objects if redo else self.undo_objects
        destination = self.undo_objects if redo else self.redo_objects
        if not source:
            return
        path, old, new, line = source[-1]
        expected, replacement = (old, new) if redo else (new, old)
        if self.source_text(path) != expected:
            self.error("The source was edited after this object action. Undo the text edit first before undoing this object action.")
            return
        source.pop()
        destination.append((path, old, new, line))
        self.buffers[path] = replacement
        self.show_file(path, line)
        self.problems.clear()
        self.rebuild()
        self.select_location(path, line)

    def selected_objects(self, item=None):
        item = item or self.tree.currentItem()
        if item is None or self.row_type(item) != "object":
            return []
        result = [self.object_for_item(item)]
        obj = result[0]
        lines = self.source_text(obj.source).splitlines()
        if core._strip_inline_comment(lines[obj.line - 1]).endswith("\\"):
            raise ValueError("Use the source editor to copy, move or remove continued commands.")
        for index in range(item.childCount()):
            result.extend(self.selected_objects(item.child(index)))
        return result

    @project_action
    def insert_command(self, command, before=False, axis_parent=None):
        self.ensure_fresh_objects()
        if not self.startup:
            self.new_project()
        if not self.startup:
            return
        path = axis_parent.source if axis_parent is not None else self.insertion_path()
        text = self.source_text(path)
        lines = text.splitlines(True)
        objects = self.selected_objects()
        position = (min(obj.line for obj in objects) - 1 if before else max(obj.line for obj in objects)) if objects else len(lines)
        if axis_parent is not None:
            # Linked-file rows are not object subtrees. Use the actual axis
            # command as the anchor, and stop before the next unrelated object.
            position = axis_parent.line
            following = sorted((obj for obj in self.all_objects()
                                if obj.source == path and obj.line > axis_parent.line), key=lambda obj: obj.line)
            for obj in following:
                if obj.kind != "encoder" or obj.parent_axis_line != axis_parent.line:
                    break
                position = obj.line
        if position and not lines[position - 1].endswith("\n"):
            lines[position - 1] += "\n"
        lines.insert(position, command.rstrip("\n") + "\n")
        self.apply_text(path, "".join(lines), position + 1)
        return path

    def all_objects(self):
        return [obj for node in self.model.files for obj in node.objects] if self.model else []

    def field_choices(self, key):
        inventory = self.inventory
        fixed = {"HW_DESC": sorted(inventory.hardware_descs),
                 "COMP": sorted(inventory.component_definitions),
                 "EC_COMP_TYPE": sorted(inventory.component_support),
                 "MODULE": ["ecmccfg", "ecmc", "ecmccomp"],
                 "WRAPPER": ["ecmcConfigOrDie", "ecmcConfig"],
                 "DT": ["BIT", "U8", "S8", "U16", "S16", "U32", "S32", "U64", "S64", "F32", "F64"]}
        if key in fixed:
            return fixed[key]
        id_key = "SLAVE_ID" if key in {"DRV_SID", "ENC_SID", "COMP_S_ID", "STRT_ENTRY_S_ID"} else key
        if id_key.endswith("_ID"):
            return sorted({dict(obj.details)[id_key] for obj in self.all_objects() if id_key in dict(obj.details)})
        return []

    def initial_values(self, kind):
        initial = {"require": {"MODULE": "ecmccfg", "MACROS": "MASTER_ID=0"}, "master": {"MASTER_ID": "0"},
                   "component": {"CH_ID": "1"}, "ecmc_command": {"WRAPPER": "ecmcConfigOrDie"},
                   "datastorage": {"DS_SIZE": "1000"},
                   "plcvar_analog": {"P": "$(IOC):", "PORT": "MC_CPU1", "REC_NAME": "-Var", "T_SMP_MS": "1000"},
                   "plcvar_binary": {"P": "$(IOC):", "PORT": "MC_CPU1", "REC_NAME": "-Var", "T_SMP_MS": "1000"}}.get(kind, {}).copy()
        id_key = {"slave": "SLAVE_ID", "axis": "AXIS_ID", "plc": "PLC_ID", "plugin": "PLUGIN_ID", "datastorage": "DS_ID", "lut": "LUT_ID"}.get(kind)
        if id_key:
            used = [core._parse_int_value(dict(obj.details).get(id_key, "")) for obj in self.all_objects()]
            initial[id_key] = str(max([value for value in used if value is not None] + [0 if kind == "axis" else -1]) + 1)
        if kind in {"axis", "encoder", "plc"}:
            filename = {"axis": "axis.yaml", "encoder": "encoder.yaml", "plc": "main.plc"}[kind]
            base = self.insertion_path().parent if self.startup else Path.cwd()
            candidate = Path("cfg") / filename
            reserved = {obj.linked_file.resolve() for obj in self.all_objects() if obj.linked_file}
            number = 2
            while (base / candidate).exists() or (base / candidate).resolve() in self.buffers or (base / candidate).resolve() in reserved:
                candidate = Path("cfg") / (Path(filename).stem + "_" + str(number) + Path(filename).suffix)
                number += 1
            initial["FILE"] = "./" + str(candidate)
        obj = self.object_for_item()
        if obj and obj.kind == "slave" and kind in CHILD_TYPES["slave"]:
            details = dict(obj.details)
            sid = str(obj.slave_id) if obj.slave_id is not None else ""
            initial[{"component": "COMP_S_ID", "ecdataitem": "STRT_ENTRY_S_ID"}.get(kind, "SLAVE_ID")] = sid
            if kind == "component":
                hw = details.get("HW_DESC", "")
                initial["EC_COMP_TYPE"] = self.inventory.hardware_component_types.get(hw, hw)
            if kind == "slave_config":
                initial["HW_DESC"] = details.get("HW_DESC", "")
        if obj and obj.kind == "plc" and kind.startswith("plcvar_"):
            initial["ASYN_NAME"] = "plcs.plc{}.static.var".format(dict(obj.details).get("PLC_ID", "0"))
        return initial

    def check_object_values(self, kind, values):
        primary = {"slave": ["HW_DESC"], "axis": ["FILE"], "encoder": ["FILE"], "plc": ["FILE"],
                   "component": ["COMP"], "configured_axis": ["CONFIG"], "plugin": ["FILE"],
                   "lut": ["FILE"], "subst_config": ["FILE"]}.get(kind, [])
        missing = [key for key in primary if not values.get(key, "").strip()]
        if missing:
            raise ValueError("Enter " + ", ".join(missing))
        for key in ("MACROS", "PLC_MACROS", "CFG_MACROS"):
            payload = values.get(key, "")
            if payload and not payload.startswith(("${", "$(")):
                _, malformed = core._parse_macro_payload(payload)
                if malformed:
                    raise ValueError("Invalid {}: {}".format(key, ", ".join(malformed)))
        if kind == "component":
            ch = values.get("CH_ID", "1")
            if "$" not in ch:
                parsed = core._parse_int_value(ch)
                if parsed is None or parsed < 1:
                    raise ValueError("CH_ID must be an integer greater than or equal to 1.")
                definition = self.inventory.component_definitions.get(values.get("COMP"))
                support_map = self.inventory.component_support.get(values.get("EC_COMP_TYPE"), {})
                support = support_map.get(definition.comp_type) if definition else None
                if definition and support_map and not support:
                    raise ValueError("The selected component is not supported by this hardware type.")
                if support and support.channel_count is not None and parsed > support.channel_count:
                    raise ValueError("This component supports channels 1..{}.".format(support.channel_count))
                if support and support.supported_macros:
                    pairs, _ = core._parse_macro_payload(values.get("MACROS", ""))
                    invalid = set(dict(pairs)) - support.supported_macros
                    if invalid:
                        raise ValueError("Unsupported component macros: " + ", ".join(sorted(invalid)))

    def prompt_object(self, kind, initial, editing=False):
        dialog = ObjectDialog(self, kind, initial, editing)
        try:
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                return dialog.command
        finally:
            dialog.deleteLater()
        return None

    @project_action
    def add_object(self, kind, before=False):
        self.ensure_fresh_objects()
        selected = self.object_for_item()
        axis_parent = selected if kind == "encoder" and not before and selected and selected.kind == "axis" else None
        command = self.prompt_object(kind, self.initial_values(kind))
        if command is not None:
            target = None
            if kind in {"axis", "encoder", "plc"}:
                if not self.startup:
                    self.new_project()
                if not self.startup:
                    return
                values = read_values(kind, command)
                file_value = values.get("FILE", "")
                if file_value and "$" not in file_value:
                    base = self.insertion_path().parent
                    relative = Path(file_value)
                    # Bare new filenames use cfg as well as the suggested names.
                    if relative.parent == Path(".") and not (base / relative).exists() and (base / relative).resolve() not in self.buffers:
                        relative = Path("cfg") / relative
                        values["FILE"] = "./" + str(relative)
                        command = render_command(kind, values)
                    target = (base / relative).resolve()
                    if not target.exists() and target not in self.buffers:
                        try:
                            target.relative_to((base / "cfg").resolve())
                        except ValueError:
                            raise ValueError("Create new axis, encoder and PLC files inside cfg/, for example ./cfg/{}.".format(relative.name))
            if axis_parent is not None:
                inserted = self.insert_command(command, axis_parent=axis_parent)
            else:
                inserted = self.insert_command(command, before)
            if inserted is not None and target and not target.exists() and target not in self.buffers:
                self.buffers[target] = core.ValidatorApp._starter_file_content(None, kind)
                self.saved[target] = None
                self.rebuild()
                self.statusBar().showMessage("Added {} and prepared {}. Use Save All to write both files.".format(kind, target.name))

    def edit_object(self):
        self.ensure_fresh_objects()
        obj = self.object_for_item()
        if not obj or obj.kind not in OBJECT_TYPES:
            return
        try:
            original_values = self.object_values(obj)
            command = self.prompt_object(obj.kind, original_values, editing=True)
            if command is None or read_values(obj.kind, command) == original_values:
                return
            lines = self.source_text(obj.source).splitlines(True)
            lines[obj.line - 1] = replace_line(lines[obj.line - 1], command)
            self.apply_text(obj.source, "".join(lines), obj.line)
        except (OSError, ValueError) as exc:
            self.error(exc)

    @project_action
    def copy_objects(self):
        self.ensure_fresh_objects()
        objects = self.selected_objects()
        if objects:
            lines = self.source_text(objects[0].source).splitlines(True)
            self.copied_objects = "".join(lines[line - 1].rstrip("\n") + "\n" for line in sorted({obj.line for obj in objects}))
            QtWidgets.QApplication.clipboard().setText(self.copied_objects)
            self.statusBar().showMessage("Copied {} object(s), including children".format(len(objects)))

    @project_action
    def copy_macros(self):
        self.ensure_fresh_objects()
        obj = self.object_for_item()
        if obj:
            try:
                values = self.object_values(obj)
                if obj.kind == "require":
                    pairs, _ = core._parse_macro_payload(values.get("MACROS", ""))
                    values = dict(pairs)
                elif obj.kind == "macro":
                    values = {values["NAME"]: values["VALUE"]}
                QtWidgets.QApplication.clipboard().setText(
                    ", ".join("{}={}".format(k, core._quote_startup_value(v)) for k, v in values.items()))
            except ValueError as exc:
                self.error(exc)

    @project_action
    def delete_objects(self):
        self.ensure_fresh_objects()
        objects = self.selected_objects()
        if not objects:
            return
        if QtWidgets.QMessageBox.question(self, "Remove objects", "Remove {} object(s), including children? Linked files are retained.".format(len(objects)),
                                          QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        path = objects[0].source
        numbers = {obj.line for obj in objects}
        text = "".join(line for number, line in enumerate(self.source_text(path).splitlines(True), 1) if number not in numbers)
        self.apply_text(path, text, min(numbers))

    @project_action
    def move_object(self, direction):
        self.ensure_fresh_objects()
        item = self.tree.currentItem()
        if self.row_type(item) != "object":
            return
        parent = item.parent()
        siblings = [parent.child(i) for i in range(parent.childCount()) if self.row_type(parent.child(i)) == "object"]
        index = siblings.index(item)
        target = index + direction
        if not 0 <= target < len(siblings):
            return
        own, other = self.selected_objects(item), self.selected_objects(siblings[target])
        a, b = sorted([own, other], key=lambda group: min(obj.line for obj in group))
        a0, a1 = min(obj.line for obj in a) - 1, max(obj.line for obj in a)
        b0, b1 = min(obj.line for obj in b) - 1, max(obj.line for obj in b)
        path = own[0].source
        included = {obj.line for obj in a + b}
        if a1 > b0 or any(obj.source == path and a0 < obj.line <= b1 and obj.line not in included for obj in self.all_objects()):
            self.error("These object groups interleave with other commands. Move them in the source editor.")
            return
        lines = self.source_text(path).splitlines(True)
        # Each moved command needs its own line even when it was originally EOF.
        first = [line if line.endswith("\n") else line + "\n" for line in lines[b0:b1]]
        second = [line if line.endswith("\n") else line + "\n" for line in lines[a0:a1]]
        text = "".join(lines[:a0] + first + lines[a1:b0] + second + lines[b1:])
        new_line = a0 + 1 if direction < 0 else a0 + len(first) + (b0 - a1) + 1
        self.apply_text(path, text, new_line)

    def add_inline_macro(self):
        self.ensure_fresh_objects()
        obj = self.object_for_item()
        if not obj:
            return
        try:
            values = self.object_values(obj)
            name = QtWidgets.QLineEdit()
            value = QtWidgets.QLineEdit()
            entered = self.form("Add macro to command", [("Name", name), ("Value", value)])
            if entered is None:
                return
            key, val = entered
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError("Enter a valid macro name.")
            if obj.kind == "require":
                pairs, _ = core._parse_macro_payload(values.get("MACROS", ""))
                if key in dict(pairs):
                    raise ValueError("That macro already exists; use Edit Object.")
                pairs.append((key, val))
                values["MACROS"] = ", ".join("{}={}".format(k, core._quote_startup_value(v)) for k, v in pairs)
            else:
                if key in values:
                    raise ValueError("That macro already exists; use Edit Object.")
                values[key] = val
            command = render_command(obj.kind, values)
            lines = self.source_text(obj.source).splitlines(True)
            lines[obj.line - 1] = replace_line(lines[obj.line - 1], command)
            self.apply_text(obj.source, "".join(lines), obj.line)
        except (OSError, ValueError) as exc:
            self.error(exc)

    @project_action
    def create_linked_file(self):
        obj = self.object_for_item()
        if not obj or not obj.linked_file:
            return
        path = obj.linked_file.resolve()
        if path.exists() or path in self.buffers:
            self.show_file(path)
            return
        text = core.ValidatorApp._starter_file_content(None, obj.kind)
        if not text:
            self.error("Create this file in the source editor or select an existing file in Edit Object.")
            return
        self.buffers[path] = text
        self.saved[path] = None
        self.show_file(path)
        self.rebuild()
        self.statusBar().showMessage("Created a starter file in memory. Complete its configuration, then Save All.")

    def open_object_online(self, tool):
        obj = self.object_for_item()
        if not obj:
            return
        details = dict(obj.details)
        is_sdo = tool == "sdo"
        try:
            default_host = self.project_online_host()
        except (OSError, UnicodeError) as exc:
            self.error(exc)
            return
        connection = QtWidgets.QLineEdit(default_host if is_sdo else self.online_prefix)
        identifier = QtWidgets.QLineEdit(str(obj.slave_id) if is_sdo and obj.slave_id is not None else details.get("AXIS_ID", ""))
        fields = [("SSH host" if is_sdo else "IOC prefix", connection),
                  ("Slave position" if is_sdo else "Axis ID", identifier)]
        if is_sdo:
            fields.append(("EtherCAT master ID", QtWidgets.QLineEdit(details.get("MASTER_ID", ""))))
        values = self.form("Open online tool — check target", fields)
        if values is None:
            return
        values = [value.strip() for value in values]
        if not values[0] or values[0].startswith("-") or any(not re.fullmatch(r"[0-9]+", value) for value in values[1:]):
            self.error("Enter a connection target and numeric IDs. Resolve project macros before connecting.")
            return
        try:
            if is_sdo:
                from ecmc_sdo_qt import SdoBrowserWindow
                self.online_host = values[0]
                widget = SdoBrowserWindow(values[0], values[2], values[1])
                title = "SDO {}:{}".format(values[2], values[1])
            else:
                from ecmc_ioc_navigator import IocNavigator
                self.online_prefix = values[0]
                widget = IocNavigator(prefix=values[0], ssh_host=default_host)
                action = {"axis": "_open_axis_config", "controller": "_open_controller", "motion": "_open_motion", "iso230": "_open_iso230"}[tool]
                getattr(widget, action)({"id": values[1]})
                title = "{} / axis {}".format(values[0], values[1])
        except Exception as exc:
            self.error(exc)
            return
        self.tabs.setCurrentIndex(self.tabs.addTab(widget, title))

    def build_context_menu(self):
        menu = QtWidgets.QMenu(self.tree)
        obj = self.object_for_item()
        row_type = self.row_type()
        if row_type == "parameters":
            path = self.tree.currentItem().data(0, QtCore.Qt.UserRole)[0]
            menu.addAction("Open Parameters File", lambda: self.show_file(path))
            return menu
        if obj and row_type == "object":
            if obj.kind in OBJECT_TYPES:
                menu.addAction("Edit Object…", self.edit_object)
            menu.addAction("Copy Object and Children", self.copy_objects)
            menu.addAction("Copy Macros", self.copy_macros)
            if obj.kind in OBJECT_TYPES and obj.kind not in {"macro", "ecmc_command"}:
                menu.addAction("Add Macro…", self.add_inline_macro)
            menu.addAction("Move Up", lambda: self.move_object(-1))
            menu.addAction("Move Down", lambda: self.move_object(1))
            menu.addAction("Remove Object and Children…", self.delete_objects)
            if obj.linked_file:
                menu.addAction("Open Linked File", lambda: self.show_file(obj.linked_file))
                if not obj.linked_file.exists() and obj.linked_file.resolve() not in self.buffers:
                    menu.addAction("Create Linked File…", self.create_linked_file)
            if obj.kind in CHILD_TYPES:
                children = menu.addMenu("Add Child")
                for kind in CHILD_TYPES[obj.kind]:
                    children.addAction(OBJECT_TYPES[kind][0] + "…", lambda checked=False, kind=kind: self.add_object(kind))
            if obj.kind in {"axis", "configured_axis", "slave"}:
                online = menu.addMenu("Online Tools")
                actions = [("SDO Browser", "sdo")] if obj.kind == "slave" else [
                    ("Axis Configuration", "axis"), ("Controller Tuning", "controller"),
                    ("Motion", "motion"), ("ISO230", "iso230")]
                for label, tool in actions:
                    online.addAction(label + "…", lambda checked=False, tool=tool: self.open_object_online(tool))
            menu.addSeparator()
        if row_type != "linked":
            for title, before in ([("Insert Before", True), ("Insert After", False)] if obj else [("Add", False)]):
                submenu = menu.addMenu(title)
                self.populate_add_menu(submenu, before)
            if self.copied_objects:
                menu.addAction("Paste Objects", lambda: self.insert_command(self.copied_objects))
        else:
            menu.addAction("Open Linked File", lambda: self.show_file(obj.linked_file))
            if not obj.linked_file.exists() and obj.linked_file.resolve() not in self.buffers:
                menu.addAction("Create Linked File…", self.create_linked_file)
        menu.addSeparator()
        undo = menu.addAction("Undo Object Action", self.undo_object_change)
        undo.setEnabled(bool(self.undo_objects))
        redo = menu.addAction("Redo Object Action", lambda: self.undo_object_change(True))
        redo.setEnabled(bool(self.redo_objects))
        return menu

    def populate_add_menu(self, menu, before=False):
        for kind, (label, _script, _fields) in OBJECT_TYPES.items():
            if kind == "master":
                continue
            menu.addAction(label + "…", lambda checked=False, kind=kind: self.add_object(kind, before))

    def context_menu(self, point):
        self.ensure_fresh_objects()
        item = self.tree.itemAt(point)
        self.tree.setCurrentItem(item)
        menu = self.build_context_menu()
        try:
            menu.exec(self.tree.viewport().mapToGlobal(point))
        finally:
            menu.deleteLater()
