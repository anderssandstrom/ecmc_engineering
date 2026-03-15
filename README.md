# ecmc Engineering

`ecmc Engineering Studio` is a desktop tool for inspecting, validating, and editing ecmc startup projects with linked YAML, PLC, and ECMC command files.

It is intended to make large startup configurations easier to understand and safer to edit.

## Clone

Recommended:

```bash
git clone --recurse-submodules <repo-url>
cd ecmc_engineering
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

You can also clone in two steps:

```bash
git clone <repo-url>
cd ecmc_engineering
git submodule update --init --recursive
```

## Repository Layout

- `ecmc_engineering_studio.py`: main GUI application
- `compat_dataclasses.py`: compatibility helper
- `ecmccfg/`: submodule with startup scripts, hardware descriptions, examples, and framework files
- `ecmccomp/`: submodule with supported component definitions
- `ecmc/`: submodule used for ECMC command parsing and command help
- `examples/`: local example startup files and configs for the studio itself

## Start

Run from the repository root:

```bash
python ecmc_engineering_studio.py --startup examples/startup_local_hw.cmd
```

Example using a startup file from the `ecmccfg` submodule:

```bash
python ecmc_engineering_studio.py --startup ecmccfg/examples/PSI/lab_setup/stepper_bissc/startup_local_hw.cmd
```

You can also start without `--startup` and open a file from the GUI afterwards:

```bash
python ecmc_engineering_studio.py
```

## What The Tool Helps With

- browse startup objects in execution order or grouped object view
- inspect linked YAML and PLC files
- add, edit, copy, paste, move, and remove startup objects
- validate startup files, linked files, macros, axis conflicts, and component compatibility
- work with ECMC commands from `ecmcConfigOrDie "..."` and `ecmcConfig "..."`
- get EtherCAT entry suggestions in YAML, PLC, and relevant ECMC command fields
- see object context, resolved values, and targeted help for the selected object

## Typical Workflow

1. Open a startup file.
2. Inspect the project in `Flow` view or switch to `Objects` view for large systems.
3. Use filters, search, and `Compact` mode to focus on slaves, axes, PLCs, issues, or unsaved changes.
4. Select an object to see its `Details`, `Context`, `Resolved`, and `Help`.
5. Edit from the object buttons, the `Quick Edit` row, or the full edit dialog.
6. Validate the startup and use `Help` / `Problems & Suggestions` to jump to issues and apply fixes.
7. Save the current file or use `Save All`.

## Requirements

- Python 3
- `tkinter`
- all git submodules present:
  - `ecmccfg`
  - `ecmccomp`
  - `ecmc`

## Notes

- The tool reads startup scripts, linked YAML/PLC files, `ecmccfg` hardware data, `ecmccomp` component metadata, and ECMC command information from the submodules.
- On some Python installations, `tkinter` is missing. In that case the GUI will not start until Tk support is installed.
- On macOS, some Tk widgets can behave a bit differently than on Linux. The main target environment is Red Hat/Linux, but the tool is also intended to remain usable on macOS.
