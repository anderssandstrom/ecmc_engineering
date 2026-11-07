"""Dialog regression checks using real Qt widgets and modal event loops.

Run with the Qt Python environment: python -m unittest test_engineering_qt
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
import tempfile
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch

from ecmc_engineering_qt import EngineeringWindow, QtCore, QtWidgets
from ecmc_object_editor_qt import ObjectDialog, OBJECT_ROLE
from ecmc_project_commands import read_values, render_command, replace_line, OBJECT_TYPES
from ecmc_ioc_project import create_ioc_project
import ecmc_engineering_studio as core


class DialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.window = EngineeringWindow()

    def finish_dialog(self, accept=True, configure=None):
        def finish():
            dialog = self.app.activeModalWidget()
            if configure:
                configure(dialog)
            if accept:
                if isinstance(dialog, ObjectDialog):
                    dialog.accept_values()
                else:
                    dialog.accept()
            else:
                dialog.reject()
        QtCore.QTimer.singleShot(0, finish)

    def test_form_returns_values_that_survive_dialog_deletion(self):
        field = QtWidgets.QLineEdit("ecmccfg")
        self.finish_dialog()
        values = self.window.form("Test", [("Module", field)])
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.assertEqual(values, ["ecmccfg"])

    def test_require_accept_and_cancel(self):
        with patch.object(self.window, "insert_command") as append:
            self.finish_dialog()
            self.window.add_require()
            append.assert_called_once_with('require ecmccfg "MASTER_ID=0"', False)
            append.reset_mock()
            self.finish_dialog(accept=False)
            self.window.add_require()
            append.assert_not_called()

    def test_slave_accept(self):
        def configure(dialog):
            dialog.findChild(QtWidgets.QComboBox, "field_HW_DESC").setCurrentText("EL1008")
            widget = dialog.findChild(QtWidgets.QWidget, "field_SLAVE_ID")
            widget.setCurrentText("23") if isinstance(widget, QtWidgets.QComboBox) else widget.setText("23")
        with patch.object(self.window, "insert_command") as append:
            self.finish_dialog(configure=configure)
            self.window.add_slave()
            append.assert_called_once()
            command = append.call_args.args[0]
            self.assertIn("SLAVE_ID=23", command)
            self.assertIn("HW_DESC=EL1008", command)

    def test_online_accept_and_cancel(self):
        def configure(dialog):
            dialog.findChild(QtWidgets.QCheckBox).setChecked(True)
        count = self.window.tabs.count()
        with patch.object(self.window, "error") as error:
            self.finish_dialog(configure=configure)
            self.window.open_online()
            error.assert_not_called()
        self.assertEqual(self.window.tabs.count(), count + 1)
        self.assertEqual(self.window.tabs.tabText(count), "Demo IOC")
        self.finish_dialog(accept=False)
        self.window.open_online()
        self.assertEqual(self.window.tabs.count(), count + 1)


class ObjectWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.window = EngineeringWindow()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = (Path(self.temp.name) / "startup.cmd").resolve()
        (self.path.parent / 'axis.yaml').write_text('axis:\n  id: ${AXIS_ID}\n')
        (self.path.parent / 'enc.yaml').write_text('encoder:\n  type: incremental\n')
        self.original = '''# project header
require ecmccfg "MASTER_ID=0,ENG_MODE=1"
epicsEnvSet(DRV_SID, 0)
${SCRIPTEXEC} ${ecmccfg_DIR}addSlave.cmd "HW_DESC=EL7041-0052, SLAVE_ID=0"
${SCRIPTEXEC} ${ecmccfg_DIR}applyComponent.cmd "COMP=Motor-Generic-2Phase-Stepper, CH_ID=1"
${SCRIPTEXEC} ${ecmccfg_DIR}addSlave.cmd "HW_DESC=EL1008, SLAVE_ID=1"
  ${SCRIPTEXEC} ${ecmccfg_DIR}loadYamlAxis.cmd, "FILE=./axis.yaml, AXIS_ID=1, AX_NAME=M1, DRV_SID=${DRV_SID}, DEV=${IOC}" # keep me
${SCRIPTEXEC} ${ecmccfg_DIR}loadYamlEnc.cmd "FILE=./enc.yaml, ENC_SID=${DRV_SID}"
'''
        self.path.write_text(self.original)
        self.window.load_project(self.path, confirm=False)
        self.window.copied_objects = ""

    def select(self, kind, index=0):
        matches = [obj for obj in self.window.all_objects() if obj.kind == kind]
        obj = matches[index]
        self.window.select_location(obj.source, obj.line)
        return obj

    def test_context_menus_and_hierarchy(self):
        w = self.window
        axis = self.select('axis')
        encoder = next(obj for obj in w.all_objects() if obj.kind == 'encoder')
        item = w.object_items[(encoder.source, encoder.line, encoder.kind)]
        self.assertEqual(item.parent().data(0, OBJECT_ROLE), axis)
        menu = w.build_context_menu()
        actions = {action.text(): action for action in menu.actions()}
        for label in ['Edit Object…', 'Copy Object and Children', 'Add Child', 'Insert Before', 'Insert After', 'Move Up', 'Remove Object and Children…']:
            self.assertIn(label, actions)
        child_actions = [action.text() for action in actions['Add Child'].menu().actions()]
        self.assertEqual(child_actions, ['Encoder…'])
        actions['Copy Object and Children'].trigger()
        self.assertIn('loadYamlAxis.cmd', w.copied_objects)
        self.assertIn('loadYamlEnc.cmd', w.copied_objects)
        self.assertNotIn('addSlave.cmd', w.copied_objects)

    def test_edit_keeps_raw_macros_and_comments(self):
        w = self.window
        self.select('axis')
        def accept():
            dialog = self.app.activeModalWidget()
            dialog.findChild(QtWidgets.QLineEdit, 'field_AX_NAME').setText('M2')
            dialog.accept_values()
        QtCore.QTimer.singleShot(0, accept)
        w.edit_object()
        text = w.buffers[self.path]
        self.assertIn('AX_NAME=M2', text)
        self.assertIn('DRV_SID=${DRV_SID}', text)
        self.assertIn('DEV=${IOC}', text)
        self.assertIn(' # keep me', text)
        self.assertIn('  ${SCRIPTEXEC}', text)
        self.assertEqual(self.path.read_text(), self.original)
        w.undo_object_change()
        self.assertEqual(w.buffers[self.path], self.original)
        w.undo_object_change(True)
        self.assertEqual(w.buffers[self.path], text)

    def test_insert_before_selected_and_save(self):
        w = self.window
        self.select('slave', 1)
        w.insert_command('epicsEnvSet(TEST, 2)', before=True)
        text = w.buffers[self.path]
        self.assertLess(text.index('epicsEnvSet(TEST'), text.index('HW_DESC=EL1008'))
        self.assertEqual(self.path.read_text(), self.original)
        self.assertTrue(w.save_all())
        self.assertEqual(self.path.read_text(), text)

    def test_delete_subtree_keeps_files_and_undo(self):
        w = self.window
        self.select('axis')
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            w.delete_objects()
        self.assertNotIn('loadYamlAxis.cmd', w.buffers[self.path])
        self.assertNotIn('loadYamlEnc.cmd', w.buffers[self.path])
        self.assertTrue((self.path.parent / 'axis.yaml').exists())
        self.assertIn('# project header', w.buffers[self.path])
        w.undo_object_change()
        self.assertEqual(w.buffers[self.path], self.original)

    def test_move_slave_with_component(self):
        w = self.window
        self.select('slave')
        w.move_object(1)
        text = w.buffers[self.path]
        self.assertLess(text.index('HW_DESC=EL1008'), text.index('HW_DESC=EL7041'))
        self.assertLess(text.index('HW_DESC=EL7041'), text.index('applyComponent.cmd'))
        self.assertIn('DRV_SID=${DRV_SID}', text)
        w.undo_object_change()
        self.assertEqual(w.buffers[self.path], self.original)

    def test_paste_at_eof_without_newline(self):
        w = self.window
        self.select('slave', 1)
        w.copy_objects()
        text = w.buffers[self.path].rstrip('\n')
        w.apply_text(self.path, text)
        w.tree.setCurrentItem(w.tree.topLevelItem(0))
        w.insert_command(w.copied_objects)
        self.assertIn('ENC_SID=${DRV_SID}"\n${SCRIPTEXEC}', w.buffers[self.path])
        self.assertEqual(w.buffers[self.path].count('HW_DESC=EL1008'), 2)

    def test_component_channel_check_and_zero_slave_context(self):
        w = self.window
        self.select('slave')
        defaults = w.initial_values('component')
        self.assertEqual(defaults['COMP_S_ID'], '0')
        with self.assertRaisesRegex(ValueError, 'CH_ID'):
            w.check_object_values('component', {'COMP': 'Motor-Generic-2Phase-Stepper', 'CH_ID': '0'})

    def test_all_types_have_working_dialogs(self):
        for kind in OBJECT_TYPES:
            dialog = ObjectDialog(self.window, kind, self.window.initial_values(kind))
            self.assertGreaterEqual(dialog.table.rowCount(), 0)
            dialog.deleteLater()

    def test_example_commands_round_trip_without_resolving_macros(self):
        w = self.window
        path = Path('examples/startup_local_hw_lookup.cmd').resolve()
        text = path.read_text()
        model = __import__('ecmc_engineering_studio').build_startup_tree(path, text, w.inventory)
        for obj in model.files[0].objects:
            if obj.kind not in OBJECT_TYPES:
                continue
            values = read_values(obj.kind, text.splitlines()[obj.line - 1])
            rendered = render_command(obj.kind, values)
            self.assertEqual(read_values(obj.kind, rendered), values, obj.title)

    def test_replace_preserves_custom_target_and_comment(self):
        original = '  $(SCRIPTEXEC) ${ECMC_CONFIG_ROOT}addSlave.cmd, "HW_DESC=EL1008, SLAVE_ID=1" # hardware\n'
        command = render_command('slave', {'HW_DESC': 'EL1008', 'SLAVE_ID': '2'})
        updated = replace_line(original, command)
        self.assertTrue(updated.startswith('  $(SCRIPTEXEC) ${ECMC_CONFIG_ROOT}addSlave.cmd,'))
        self.assertTrue(updated.endswith(' # hardware\n'))
        self.assertEqual(read_values('slave', updated)['SLAVE_ID'], '2')

    def test_create_linked_file_is_buffered_until_save(self):
        w = self.window
        self.select('axis')
        linked = (self.path.parent / 'axis.yaml')
        linked.unlink()
        w.create_linked_file()
        self.assertIn(linked, w.buffers)
        self.assertIsNone(w.saved[linked])
        self.assertFalse(linked.exists())
        self.assertTrue(w.save_all())
        self.assertIn('axis:', linked.read_text())

    def test_save_checks_all_files_before_writing(self):
        w = self.window
        self.select('axis')
        w.insert_command('# edit startup')
        linked = self.path.parent / 'axis.yaml'
        w.show_file(linked)
        w.editor.insertPlainText('# local edit\n')
        linked.write_text('# external edit\n')
        with patch.object(w, 'error') as error:
            self.assertFalse(w.save_all())
            error.assert_called_once()
        self.assertEqual(self.path.read_text(), self.original)
        self.assertEqual(linked.read_text(), '# external edit\n')

    def test_accepting_unchanged_object_does_not_reformat_source(self):
        w = self.window
        self.select('axis')
        QtCore.QTimer.singleShot(0, lambda: self.app.activeModalWidget().accept_values())
        w.edit_object()
        self.assertEqual(w.buffers[self.path], self.original)
        self.assertEqual(w.undo_objects, [])

    def test_inline_require_macro_real_dialog(self):
        w = self.window
        self.select('require')
        def accept():
            dialog = self.app.activeModalWidget()
            fields = dialog.findChildren(QtWidgets.QLineEdit)
            fields[0].setText('EC_RATE')
            fields[1].setText('1000')
            dialog.accept()
        QtCore.QTimer.singleShot(0, accept)
        w.add_inline_macro()
        self.assertIn('EC_RATE=1000', w.buffers[self.path])
        self.assertIn('MASTER_ID=0', w.buffers[self.path])

    def test_conditional_prefix_is_retained(self):
        original = '${LOAD_IT}ecmcConfigOrDie "Cfg.SetAxisEnable(1,0)" # condition\n'
        updated = replace_line(original, 'ecmcConfigOrDie "Cfg.SetAxisEnable(2,0)"')
        self.assertEqual(updated, '${LOAD_IT}ecmcConfigOrDie "Cfg.SetAxisEnable(2,0)" # condition\n')

    def test_add_remove_custom_field_updates_preview(self):
        dialog = ObjectDialog(self.window, 'slave', {'HW_DESC': 'EL1008', 'SLAVE_ID': '1'})
        dialog.add_row('', '')
        row = dialog.table.rowCount() - 1
        dialog.table.item(row, 0).setText('CUSTOM')
        dialog.table.cellWidget(row, 1).setText('${VALUE}')
        self.assertIn('CUSTOM=${VALUE}', dialog.preview.toPlainText())
        dialog.table.setCurrentCell(row, 0)
        dialog.remove_row()
        self.assertNotIn('CUSTOM=', dialog.preview.toPlainText())
        dialog.deleteLater()

    def test_selected_axis_launch_passes_id_without_live_connection(self):
        w = self.window
        self.select('axis')
        navigator = QtWidgets.QWidget()
        navigator._open_motion = Mock()
        with patch.object(w, 'form', return_value=['IOC:TEST', '7']), patch('ecmc_ioc_navigator.IocNavigator', return_value=navigator) as create:
            w.open_object_online('motion')
            create.assert_called_once_with(prefix='IOC:TEST', ssh_host=w.online_host)
            navigator._open_motion.assert_called_once_with({'id': '7'})
        self.assertIs(w.tabs.currentWidget(), navigator)

    def test_selected_slave_launch_passes_master_and_position(self):
        w = self.window
        self.select('slave')
        browser = QtWidgets.QWidget()
        with patch.object(w, 'form', return_value=['test-host', '0', '2']), patch('ecmc_sdo_qt.SdoBrowserWindow', return_value=browser) as create:
            w.open_object_online('sdo')
            create.assert_called_once_with('test-host', '2', '0')

    def test_tree_shortcut_signals_do_not_pass_checked_argument(self):
        w = self.window
        self.select('slave', 1)
        actions = {action.text(): action for action in w.tree.actions()}
        with patch.object(sys, 'excepthook') as unhandled:
            actions['Copy Objects'].trigger()
            self.assertIn('HW_DESC=EL1008', w.copied_objects)
            actions['Paste Objects'].trigger()
            unhandled.assert_not_called()
        self.assertEqual(w.buffers[self.path].count('HW_DESC=EL1008'), 2)

    def test_require_preserves_quoted_and_macro_versions(self):
        for version in ['"PSI_default"', '${ECMCCFG_VER=master}']:
            line = 'require ecmccfg ' + version + ' "MASTER_ID=0"'
            values = read_values('require', line)
            self.assertEqual(values['VERSION'], version.strip('"'))
            values['MACROS'] = 'MASTER_ID=0,ENG_MODE=1'
            self.assertEqual(read_values('require', render_command('require', values)), values)

    def test_new_ioc_project_real_dialog_and_parameters_editing(self):
        w = self.window
        name = 'MTEST04-MTN-CPP-LOGIC'
        def accept():
            dialog = self.app.activeModalWidget()
            dialog.findChild(QtWidgets.QLineEdit).setText(name)
            dialog.accept()
        QtCore.QTimer.singleShot(0, accept)
        with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=str(self.path.parent)):
            w.new_project()
        directory = self.path.parent / name
        startup = directory / (name + '_startup.cmd')
        parameters = directory / (name + '_parameters.yaml')
        self.assertEqual(w.startup, startup)
        self.assertEqual(startup.read_bytes(), b'')
        self.assertEqual(sorted(p.name for p in directory.iterdir()), [parameters.name, startup.name])
        self.assertEqual(parameters.read_text(),
                         'ioc_name: "MTEST04-MTN-CPP-LOGIC"\nioc_host: ""\nioc_port: 50001\n'
                         'ioc_user: root\nepics_version: 7.0.10\nos: deb12\ncpu_architecture: x86_64\n')
        item = w.tree.topLevelItem(1)
        self.assertEqual(item.text(0), parameters.name)
        w.tree.setCurrentItem(item)
        self.assertEqual(w.current, parameters)
        self.assertEqual(w.insertion_path(), startup)
        w.editor.setPlainText(w.editor.toPlainText().replace('ioc_host: ""', 'ioc_host: c6025a-08'))
        self.assertTrue(w.save_all())
        self.assertIn('ioc_host: c6025a-08', parameters.read_text())
        w.load_project(startup, confirm=False)
        self.assertIn(parameters, w.buffers)
        self.assertEqual(w.tree.topLevelItemCount(), 2)

    def test_ioc_project_refuses_existing_directory_and_invalid_names(self):
        directory = self.path.parent / 'Existing'
        directory.mkdir()
        sentinel = directory / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            create_ioc_project(self.path.parent, 'Existing')
        self.assertEqual(list(directory.iterdir()), [sentinel])
        for name in ['', '../outside', '/absolute', 'name/child', 'name with spaces']:
            with self.assertRaises(ValueError):
                create_ioc_project(self.path.parent, name)

    def test_cancel_ioc_creation_keeps_current_project(self):
        w = self.window
        with patch.object(w, 'form', return_value=['NEW-IOC']), patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=''):
            w.new_project()
        self.assertEqual(w.startup, self.path)
        self.assertFalse((self.path.parent / 'NEW-IOC').exists())
        with patch.object(w, 'form', return_value=['NEW-IOC']), patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=str(self.path.parent)), patch.object(w, 'confirm_switch', return_value=False):
            w.new_project()
        self.assertEqual(w.startup, self.path)
        self.assertFalse((self.path.parent / 'NEW-IOC').exists())

    def test_ioc_creation_cleans_up_after_write_failure(self):
        original_open = Path.open
        def fail_parameters(path, *args, **kwargs):
            if path.name == 'BROKEN_parameters.yaml':
                raise OSError('simulated write failure')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', fail_parameters):
            with self.assertRaisesRegex(OSError, 'simulated write failure'):
                create_ioc_project(self.path.parent, 'BROKEN')
        self.assertFalse((self.path.parent / 'BROKEN').exists())

    def test_new_axis_encoder_and_plc_files_use_cfg(self):
        w = self.window
        for kind, filename in [('axis', 'axis.yaml'), ('encoder', 'encoder.yaml'), ('plc', 'main.plc')]:
            w.tree.setCurrentItem(w.tree.topLevelItem(0))
            self.assertEqual(w.initial_values(kind)['FILE'], './cfg/' + filename)
            QtCore.QTimer.singleShot(0, lambda: self.app.activeModalWidget().accept_values())
            w.add_object(kind)
            target = self.path.parent / 'cfg' / filename
            self.assertIn(target, w.buffers)
            self.assertIsNone(w.saved[target])
            self.assertFalse(target.exists())
            self.assertIn('FILE=./cfg/' + filename, w.buffers[self.path])
        self.assertEqual(w.initial_values('axis')['FILE'], './cfg/axis_2.yaml')
        self.assertTrue(w.save_all())
        for filename in ['axis.yaml', 'encoder.yaml', 'main.plc']:
            self.assertTrue((self.path.parent / 'cfg' / filename).is_file())

    def test_new_bare_filename_goes_in_cfg_and_existing_file_is_preserved(self):
        w = self.window
        w.tree.setCurrentItem(w.tree.topLevelItem(0))
        command = render_command('axis', {'FILE': 'myaxis.yaml', 'AXIS_ID': '2'})
        with patch.object(w, 'prompt_object', return_value=command):
            w.add_object('axis')
        self.assertIn(self.path.parent / 'cfg' / 'myaxis.yaml', w.buffers)
        self.assertIn('FILE=./cfg/myaxis.yaml', w.buffers[self.path])
        original = (self.path.parent / 'axis.yaml').read_text()
        command = render_command('axis', {'FILE': 'axis.yaml', 'AXIS_ID': '3'})
        with patch.object(w, 'prompt_object', return_value=command):
            w.add_object('axis')
        self.assertEqual((self.path.parent / 'axis.yaml').read_text(), original)

    def test_add_menu_has_no_master_and_require_has_master_macro(self):
        menu = QtWidgets.QMenu()
        self.window.populate_add_menu(menu)
        labels = [action.text() for action in menu.actions()]
        self.assertNotIn('Master…', labels)
        self.assertIn('Require…', labels)
        self.assertEqual(self.window.initial_values('require')['MACROS'], 'MASTER_ID=0')
        values = {'MODULE': 'ecmccfg', 'MACROS': 'ENG_MODE=1', 'MASTER_ID': '2'}
        command = render_command('require', values)
        self.assertIn('MASTER_ID=2', command)

    def prepare_host_project(self):
        startup = create_ioc_project(self.path.parent, 'HOST-TEST')
        startup.write_text(self.original)
        parameters = startup.with_name('HOST-TEST_parameters.yaml')
        parameters.write_text(parameters.read_text().replace('ioc_host: ""', 'ioc_host: c6025a-08'))
        self.window.load_project(startup, confirm=False)
        return parameters

    def test_online_navigator_uses_unsaved_yaml_host(self):
        w = self.window
        parameters = self.prepare_host_project()
        self.assertEqual(w.project_online_host(), 'c6025a-08')
        w.online_host = 'old-manual-host'
        w.show_file(parameters)
        w.editor.setPlainText('ioc_host: "new-controller" # use this host\n')
        def accept():
            dialog = self.app.activeModalWidget()
            dialog.findChild(QtWidgets.QCheckBox).setChecked(True)
            dialog.accept()
        QtCore.QTimer.singleShot(0, accept)
        navigator = QtWidgets.QWidget()
        with patch('ecmc_ioc_navigator.IocNavigator', return_value=navigator) as create:
            w.open_online()
            self.assertEqual(create.call_args.kwargs['ssh_host'], 'new-controller')
        self.assertIn('c6025a-08', parameters.read_text())

    def test_sdo_and_axis_tools_use_yaml_host(self):
        w = self.window
        self.prepare_host_project()
        self.select('slave')
        def sdo_form(title, fields):
            self.assertEqual(fields[0][1].text(), 'c6025a-08')
            return [fields[0][1].text(), '0', '0']
        with patch.object(w, 'form', side_effect=sdo_form), patch('ecmc_sdo_qt.SdoBrowserWindow', return_value=QtWidgets.QWidget()) as create:
            w.open_object_online('sdo')
            create.assert_called_once_with('c6025a-08', '0', '0')
        self.select('axis')
        navigator = QtWidgets.QWidget()
        navigator._open_motion = Mock()
        with patch.object(w, 'form', return_value=['IOC:TEST', '1']), patch('ecmc_ioc_navigator.IocNavigator', return_value=navigator) as create:
            w.open_object_online('motion')
            self.assertEqual(create.call_args.kwargs['ssh_host'], 'c6025a-08')

    def test_blank_or_missing_host_falls_back_and_project_switch_resets(self):
        w = self.window
        parameters = self.prepare_host_project()
        w.online_host = 'manual-host'
        for text in ['ioc_host: ""\n', 'ioc_host:\n', 'ioc_host: null\n', 'ioc_host: ~\n', 'ioc_name: TEST\n']:
            w.buffers[parameters] = text
            self.assertEqual(w.project_online_host(), 'manual-host')
        w.load_project(self.path, confirm=False)
        self.assertEqual(w.project_online_host(), '')

    def test_add_encoder_follows_axis_from_object_or_linked_file(self):
        w = self.window
        source = ('require ecmccfg "MASTER_ID=0"\n'
                  '${SCRIPTEXEC} ${ecmccfg_DIR}loadYamlAxis.cmd "FILE=./axis.yaml,AXIS_ID=1"\n'
                  '${SCRIPTEXEC} ${ecmccfg_DIR}loadPLCFile.cmd "FILE=./main.plc,PLC_ID=0"\n'
                  '${SCRIPTEXEC} ${ecmccfg_DIR}loadYamlAxis.cmd "FILE=./axis.yaml,AXIS_ID=2"\n')
        for linked in [False, True]:
            with self.subTest(linked_file_selected=linked):
                self.path.write_text(source)
                w.load_project(self.path, confirm=False)
                axis = self.select('axis')
                if linked:
                    w.tree.setCurrentItem(w.tree.currentItem().child(0))
                QtCore.QTimer.singleShot(0, lambda: self.app.activeModalWidget().accept_values())
                if linked:
                    # The toolbar Add action also works with axis.yaml selected.
                    w.add_object('encoder')
                else:
                    menu = w.build_context_menu()
                    child = next(action.menu() for action in menu.actions() if action.text() == 'Add Child')
                    child.actions()[0].trigger()
                lines = w.buffers[self.path].splitlines()
                self.assertIn('loadYamlEnc.cmd', lines[axis.line])
                self.assertIn('loadPLCFile.cmd', lines[axis.line + 1])
                encoder = next(obj for obj in w.all_objects() if obj.kind == 'encoder')
                self.assertEqual(encoder.parent_axis_line, axis.line)
                self.assertEqual(self.path.read_text(), source)
                w.undo_object_change()
                self.assertEqual(w.buffers[self.path], source)

    def test_multiple_encoders_stay_together_before_plc(self):
        w = self.window
        source = ('require ecmccfg "MASTER_ID=0"\n'
                  '${SCRIPTEXEC} ${ecmccfg_DIR}loadYamlAxis.cmd "FILE=./axis.yaml,AXIS_ID=1"\n'
                  '${SCRIPTEXEC} ${ecmccfg_DIR}loadYamlEnc.cmd "FILE=./enc.yaml"\n'
                  '${SCRIPTEXEC} ${ecmccfg_DIR}loadPLCFile.cmd "FILE=./main.plc,PLC_ID=0"\n')
        self.path.write_text(source)
        w.load_project(self.path, confirm=False)
        self.select('axis')
        QtCore.QTimer.singleShot(0, lambda: self.app.activeModalWidget().accept_values())
        w.add_object('encoder')
        kinds = [obj.kind for obj in w.all_objects()]
        self.assertEqual(kinds, ['require', 'axis', 'encoder', 'encoder', 'plc'])
        self.assertIn('FILE=./enc.yaml', w.buffers[self.path].splitlines()[2])

    def test_displaced_encoder_does_not_pull_new_encoder_past_plc(self):
        w = self.window
        source = ('require ecmccfg "MASTER_ID=0"\n'
                  '${SCRIPTEXEC} ${ecmccfg_DIR}loadYamlAxis.cmd "FILE=./axis.yaml,AXIS_ID=1"\n'
                  '${SCRIPTEXEC} ${ecmccfg_DIR}loadPLCFile.cmd "FILE=./main.plc,PLC_ID=0"\n'
                  '${SCRIPTEXEC} ${ecmccfg_DIR}loadYamlEnc.cmd "FILE=./enc.yaml"\n')
        self.path.write_text(source)
        w.load_project(self.path, confirm=False)
        self.select('axis')
        QtCore.QTimer.singleShot(0, lambda: self.app.activeModalWidget().accept_values())
        w.add_object('encoder')
        self.assertEqual([obj.kind for obj in w.all_objects()], ['require', 'axis', 'encoder', 'plc', 'encoder'])

    def test_motion_starters_satisfy_ecb_schema_and_preserve_hardware_macros(self):
        schema = json.loads(Path('ecmccfg/scripts/jinja2/ecbSchema.json').read_text())
        for kind in ['axis', 'encoder']:
            text = core.ValidatorApp._starter_file_content(None, kind)
            issues = core._validate_ecb_yaml(Path(kind + '.yaml'), text, kind, schema)
            self.assertEqual(issues, [], [issue.message for issue in issues])
            fields = {entry.path: entry.value for entry in core._parse_simple_yaml_paths(text)}
            for key in ['encoder.type', 'encoder.numerator', 'encoder.denominator', 'encoder.bits', 'encoder.absBits', 'encoder.position']:
                self.assertIn(key, fields)
            self.assertEqual(fields['encoder.type'], '0')
            required = core._scan_file_macro_usage(text).required
            self.assertIn('ENC_SID', required)
            if kind == 'axis':
                for key in ['epics.name', 'drive.setpoint', 'drive.control', 'drive.status',
                            'controller.Kp', 'trajectory.axis.velocity', 'trajectory.axis.acceleration',
                            'input.limit.forward', 'input.limit.backward', 'monitoring.target.enable']:
                    self.assertIn(key, fields)
                self.assertNotIn('axis.name', fields)
                self.assertTrue({'DRV_SID', 'LIMIT_FWD', 'LIMIT_BWD'} <= required)

    def test_add_axis_prepares_full_template_and_exposes_connection_fields(self):
        w = self.window
        w.tree.setCurrentItem(w.tree.topLevelItem(0))
        QtCore.QTimer.singleShot(0, lambda: self.app.activeModalWidget().accept_values())
        w.add_object('axis')
        created = self.path.parent / 'cfg' / 'axis.yaml'
        self.assertEqual(w.buffers[created], Path('templates/axis.yaml').read_text())
        self.assertIn('LIMIT_FWD', OBJECT_TYPES['axis'][2].split())
        self.assertIn('ENC_SID', OBJECT_TYPES['encoder'][2].split())
        self.assertIn('ENC_CH', OBJECT_TYPES['encoder'][2].split())

    def test_open_project_directory_loads_matching_files(self):
        w = self.window
        startup = create_ioc_project(self.path.parent, 'OPEN-TEST')
        with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=str(startup.parent)):
            w.open_project_dialog()
        self.assertEqual(w.startup, startup)
        self.assertIn(startup.with_name('OPEN-TEST_parameters.yaml'), w.buffers)
        self.assertEqual(w.tree.topLevelItemCount(), 2)

    def test_open_startup_file_still_supported(self):
        w = self.window
        startup = create_ioc_project(self.path.parent, 'OPEN-FILE')
        with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(str(startup), '')):
            w.open_dialog()
        self.assertEqual(w.startup, startup)

    def test_project_directory_fallback_and_cancel(self):
        w = self.window
        directory = self.path.parent / 'renamed-project'
        directory.mkdir()
        startup = directory / 'old-name_startup.cmd'
        startup.write_text('')
        w.load_project_directory(directory)
        self.assertEqual(w.startup, startup)
        (directory / 'startup.cmd').write_text('')
        with patch.object(QtWidgets.QInputDialog, 'getItem', return_value=('', False)):
            w.load_project_directory(directory)
        self.assertEqual(w.startup, startup)
        with patch.object(QtWidgets.QInputDialog, 'getItem', return_value=('startup.cmd', True)):
            w.load_project_directory(directory)
        self.assertEqual(w.startup, directory / 'startup.cmd')
        with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=''):
            w.open_project_dialog()
        self.assertEqual(w.startup, directory / 'startup.cmd')

    def test_empty_directory_does_not_replace_project(self):
        directory = self.path.parent / 'empty'
        directory.mkdir()
        with patch.object(self.window, 'error') as error:
            self.window.load_project_directory(directory)
            error.assert_called_once()
        self.assertEqual(self.window.startup, self.path)


if __name__ == "__main__":
    unittest.main()
