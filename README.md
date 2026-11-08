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
- `ecmc_cfg_tool/`: submodule with the live IOC navigator and online configuration, tuning, and diagnostic tools
- `ecmc_podman/`: submodule with the Podman simulation runtime, image builder, and IOC simulation launcher
- `examples/`: local example startup files and configs for the studio itself

## Start

### Qt workspace preview

The new Qt workspace offers a compact project toolbar, an object tree, a text
editor, clickable validation results, and the online navigator in a separate
tab within the same window:

```bash
bash start_engineering_qt.sh
bash start_engineering_qt.sh --startup examples/startup_local_hw.cmd
```

Use **New IOC Project...**, enter the IOC name, and select its parent directory.
For an IOC named `MTEST04-MTN-CPP-LOGIC`, Studio creates:

```text
MTEST04-MTN-CPP-LOGIC/
  MTEST04-MTN-CPP-LOGIC_startup.cmd
  MTEST04-MTN-CPP-LOGIC_parameters.yaml
```

The startup file is empty. The parameters file contains the IOC name, an empty
`ioc_host`, port `50001`, user `root`, EPICS version `7.0.10`, OS `deb12`, and CPU
architecture `x86_64`. Both files appear in the tree; select the parameters file
to edit it. Opening the matching startup file later also loads its parameters.
Existing project directories are never overwritten. The New button's dropdown
also provides **New Startup File...** for a standalone startup file.

**Open Project...** selects an IOC directory and opens its matching
`<directory-name>_startup.cmd` and parameters file. It also recognises a single
`*_startup.cmd` or `startup.cmd`, and asks which file to use if there are multiple
candidates. The Open button's dropdown provides **Open Startup File...** to
select a startup file directly.

Then use **Add → Require / Slave** to build the startup. New `require ecmccfg`
entries include `MASTER_ID=0` in the macros field; change that value to select the
master. There is no separate Add Master action.

New axis, encoder, and PLC files go in `cfg/`, starting with `axis.yaml`,
`encoder.yaml`, and `main.plc`. Studio prepares their starter contents when the
object is added; **Save All** creates the directory and writes the files. Further
files get distinct names such as `axis_2.yaml`. Existing files can also be linked.
Axis and encoder starters are based on the best-practice motion examples in
`ecmccfg`. The axis template includes drive and encoder scaling/links, EPICS
motor-record settings, controller gains, trajectory, limit inputs, soft limits,
and monitoring. The additional encoder template includes scaling, bit counts,
position input, and commented status/homing options.
Configure the marked scaling/tuning values and supply `DRV_SID`, `ENC_SID`,
`LIMIT_FWD`, and `LIMIT_BWD` for the axis; additional encoders use `ENC_SID` and
optional `ENC_CH`. The initial axis drive setup is CSV/DS402 and must be adapted
for other drive types or CSP. Template files live in `templates/` and are shared
by the Qt and Tk editors.

Changes remain in memory
until **Save All**; switching projects or closing prompts about unsaved work.
**Connect / Demo...** opens the online navigator. Demo populates sample objects;
the navigator's tool launch actions are not simulated.

Right-click an object for **Edit**, **Add Child**, **Insert Before / After**,
**Copy / Paste**, **Move Up / Down**, **Remove**, and macro actions. Components
are grouped under slaves, encoders under axes, and variables under PLCs. Double-click
an object to edit it. All recognised object types have field-based dialogs with
command previews and additional macro rows. Hardware, component, and ID fields
offer suggestions. Component dialogs check channel limits and supported macros
when hardware metadata is available.

The **Values** pane shows original command values alongside resolved values;
**Context / Help** shows source location and script documentation. Edits preserve
macro references, command indentation, and trailing comments. Object copy/delete
includes children but retains linked files. **Undo Object Action / Redo Object
Action** is available in the context menu. Source-editor undo remains separate.
Missing axis, encoder, and PLC files can be created as unsaved starter files from
**Create Linked File**; complete their configuration before use.

Axis and slave context menus also offer **Online Tools** with editable target
IDs. The tools open inside the Qt workspace. The IOC prefix and SSH host are
remembered for the session. A non-empty `ioc_host` in the project's parameters
YAML takes priority when opening the navigator or selected-object tools, including
unsaved edits to that file. The host can still be changed in the launch dialog;
when YAML has no host, the manually entered session host is used. Switching
projects clears the remembered host. No live settings are automatically written back to
project files.

The migration is still in progress. The Tk application below retains specialised
command-building and assistance features not yet ported. See
[Qt migration status](docs/qt-migration.md). Controller hardware import is not
implemented yet.
The launcher uses the same Qt Python selection as the online tools, including
the `ECMC_PYTHON` override.

Run the Qt regression checks using a Python environment with Qt installed:

```bash
python -m unittest -v test_engineering_qt test_simulation_qt
```

### Existing Tk workspace

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

## Simulation Runtime

The `ecmc_podman` submodule provides the runtime for IOC simulations. Initialize
it in an existing checkout with:

```bash
git submodule update --init --recursive
```

Its upstream URL uses GitHub SSH access. With Podman installed and running, use
the scripts from the engineering repository root:

```bash
./ecmc_podman/build-image.sh
./ecmc_podman/run-sim.sh --ioc-name SIMIOC /path/to/project/IOC_startup.cmd
```

See [the simulation runtime documentation](ecmc_podman/readme.md) for setup,
module versions, and Python hardware models.

In the **Qt engineering tool**, open your project and click **Simulate**. Choose
the container image and whether to **Enable Python hardware simulation**, then
click **Start**. With the default options this runs:

```bash
./ecmc_podman/run-sim.sh --image ecmc-dev:portable --ioc-name SIMIOC --hardware-sim /path/to/project/IOC_startup.cmd
```

Podman must be running and the `ecmc-dev:portable` image must be available.
Unsaved project changes prompt you to **Save** or **Cancel** before launch. The
startup directory (including `cfg/`) is mounted read-only into the container.

The **Simulation — SIMIOC** tab provides IOC shell output and simulation logs.
Type an IOC shell command such as `dbl` in the command field and press **Enter**
or **Send**. **Stop Simulation** sends `exit` to the IOC and waits for the IOC and
hardware models to shut down. Clicking **Simulate** while it is running returns
you to this tab. The **×** button on the tab closes it; if the simulation is
running, it sends `exit` and removes the tab once the process has stopped.

**Open Online Tools** opens the IOC navigator in a separate window using
`ecmc_cfg_tool/start_ioc.sh SIMIOC`, with Channel Access directed to localhost.
Wait for IOC startup to finish before connecting. The simulation launcher publishes
the local CA ports; only one simulation can run in this workspace at a time.

## Online IOC Tools

For an existing checkout, initialize the online tools with:

```bash
git submodule update --init --recursive
```

The `ecmc_cfg_tool` submodule uses its upstream SSH URL, so cloning it requires
GitHub SSH access. The parent repository pins a specific revision;
updating submodules with the command above checks out that pinned revision.

In Engineering Studio, click **Online Tools...**, enter the IOC prefix and optional
SSH host, and choose **Open Navigator**. **Demo** opens the navigator without an
IOC connection. The fields are remembered for the current Studio session. Launch
errors, including missing Qt dependencies, are shown in Studio.

Right-click an axis or configured-axis object and choose **Online Tools** to open
Axis Configuration, Controller Tuning, Motion, or ISO230 with its axis ID filled
in. Right-click a slave to open the SDO Browser with its slave ID filled in.
Review the target fields before launching: the chosen IOC must use the same axis
IDs, and the SDO browser requires the actual EtherCAT slave position and master
ID. A master ID absent from the object must be entered explicitly. Unresolved
IDs must be replaced with numeric values. These actions share the current
session's IOC prefix and SSH host with the navigator launcher.

Axis Configuration currently uses the online tool's bundled YAML template;
project YAML transfer and saving live changes back into project files remain
future integration steps.

You can also launch the live IOC navigator from the repository root:

```bash
./ecmc_cfg_tool/start_ioc.sh IOC:ECMC
```

Optionally supply an SSH host for remote EtherCAT tools, or explore the navigator
without connecting to an IOC:

```bash
./ecmc_cfg_tool/start_ioc.sh IOC:ECMC c6025a
./ecmc_cfg_tool/start_ioc.sh --demo
```

The navigator opens online tools in tabs or separate windows. At this stage the
project editor and online navigator are separate applications; project-to-IOC
mapping and configuration write-back are not yet integrated.

Online tools require `PyQt5` or `PySide6`. Live EPICS access additionally requires
`pyepics` or the `caget` and `caput` command-line tools. Remote SDO access uses SSH
and EtherCAT tools on the remote host. These dependencies are optional for the
Tkinter project editor. Set `ECMC_PYTHON` to select the online tools' Python
interpreter if needed.

See [the online tools documentation](ecmc_cfg_tool/README_qt_app.md) for individual
axis, controller, motion, DAQ, and SDO launchers and their requirements.

## What The Tool Helps With

- browse startup objects in execution order or grouped object view
- inspect linked YAML and PLC files
- add, edit, copy, paste, move, and remove startup objects
- validate startup files, linked files, macros, axis conflicts, and component compatibility
- work with ECMC commands from `ecmcConfigOrDie "..."` and `ecmcConfig "..."`
- get EtherCAT entry suggestions in YAML, PLC, and relevant ECMC command fields
- see object context, resolved values, and targeted help for the selected object

## Typical Workflow

To start a new project, click **New Startup...** and choose a new `.cmd` filename.
Use **Add... → Require** to add the module requirement (the default module is
`ecmccfg`), then **Add... → Slave** to choose hardware and a slave ID. Add axis,
macro, or PLC objects from the same menu. With no object selected, additions go
at the end of the startup file; with an object selected, they go after it.
The **Add** action also offers to create a startup file if none is open.
Object additions are written to the startup file immediately.

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

The `ecmc_cfg_tool` submodule is needed when using the online tools described above.

## Notes

On macOS, if caQtDM is installed in `/Applications`, this alias can be useful
for launching it from a terminal. Add it to `~/.zshrc`:

```bash
alias caqtdm='/Applications/caQtDM.app/Contents/MacOS/caQtDM'
```

Shell aliases apply only to interactive terminals. The online tools find caQtDM
on `PATH` or in `/Applications/caQtDM.app` (also `~/Applications` on macOS).
For another installation, export `CAQTDM_EXECUTABLE` with the full executable path
before starting the engineering tool.

Panel lookup supports both installed and repository layouts: an explicit
navigator `--caqtdm-dir` takes priority, followed by `ECMC_CAQTDM_DIR`, existing
`CAQTDM_DISPLAY_PATH` and `EPICS_DISPLAY_PATH` directories, then this checkout's
`ecmccfg/qt`. The selected panel's directory is added to the launched process's
display paths so embedded panels can be found too.

- The tool reads startup scripts, linked YAML/PLC files, `ecmccfg` hardware data, `ecmccomp` component metadata, and ECMC command information from the submodules.
- On some Python installations, `tkinter` is missing. In that case the GUI will not start until Tk support is installed.
- On macOS, some Tk widgets can behave a bit differently than on Linux. The main target environment is Red Hat/Linux, but the tool is also intended to remain usable on macOS.
