# Qt engineering migration

The Qt workspace uses the existing startup parser, repository inventory and
validator. `ecmc_project_commands.py` handles source command values and rendering;
`ecmc_object_editor_qt.py` supplies field dialogs and tree actions. The Tk editor
remains available throughout migration.

## Available

- Create IOC project folders with matching `_startup.cmd` and `_parameters.yaml`
  files, or create standalone startup files. New configuration files use `cfg/`.
- Open projects, edit source and linked files, save all, and validate
  in-memory edits. Save checks for externally modified files before writing.
- Object hierarchy: slave components/configuration, axis encoders, PLC variables.
- Right-click add/edit, insert before/after, add child, copy/paste, remove,
  move, add macro, copy macros, and linked-file navigation.
- Field dialogs for all recognised kinds, including Require, slaves,
  components, axes, encoders, PLCs, plugins, data storage, SDOs, data items,
  state machines, substitutions, PVT, LUTs, and ECMC commands.
  Master selection belongs in the Require macros (`MASTER_ID`); Add Master is
  omitted, while existing master commands remain readable and editable.
- Raw and resolved properties, source/script context, object filtering, and
  clickable validation results.
- Object-action undo/redo, unsaved-change prompts, and starter linked files.
- Embedded IOC navigator and selected-axis/slave online tool launch actions.

## Still to migrate

- Specialised ECMC signature/parameter builder (the command field works now).
- Editor completion, macro hover, EtherCAT entry pickers, and targeted fixes.
- Multiple-object selection, grouped/sorted view alternatives, and file browser.
- Richer component filtering and live field-specific help.
- Persistent workspace layout and project/IOC association.

## Behaviour to know

- Object changes remain unsaved until Save All, including newly created linked
  files. Copy/paste does not duplicate referenced files or renumber IDs; validate
  after pasting to detect duplicate identities.
- Undo Object Action refuses to overwrite subsequent source-text edits. Undo
  those text edits first, then undo the object action.
- Moving a group preserves its contiguous command block. Interleaved groups and
  continued commands must be edited in the source editor.
- Dialog values come from original source, never resolved macro values. Unchanged
  accepted dialogs do not reformat the command. Custom script targets and trailing
  comments are preserved for supported script-execution forms; other invocation
  forms remain editable as text.
- Headless tests exercise actual modal Qt dialogs, source round trips, tree
  actions, save handling, and launch argument routing. Online targets are mocked;
  tests do not connect to controllers.

## Later integration

Controller slave discovery/import and Podman simulation are separate follow-up
features. Simulation should use compatible ECMC/ecmccfg revisions supporting
`SIM=1`, with the existing Python hardware models as needed. The currently pinned
configuration library must be checked before enabling that workflow.
