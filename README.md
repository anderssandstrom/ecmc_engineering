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

## Installation and requirements

The supported host platforms are Linux and macOS. Windows is not currently
supported directly because the simulation launchers use Bash, Unix paths and a
Linux Podman container. Running the complete application inside WSL2 may work,
but is not currently a tested or supported configuration.

Install these base tools:

- Python 3.9 or newer for the setup script
- Git and Bash
- all Git submodules (`git submodule update --init --recursive`)
- one Qt Python binding: `PyQt5` or `PySide6`

For live IOC access and the embedded online navigator, also install one EPICS
client backend:

- `pyepics` (preferred), or
- the EPICS `caget` and `caput` command-line tools on `PATH`

The simplest setup (including on Red Hat) is:

```bash
bash setup_engineering_venv.sh
bash start_engineering_qt.sh
```

The setup script creates `.venv` in the checkout and installs Qt, PyEpics,
NumPy, PyYAML and Jinja2 from `requirements.txt`. It needs access to a Python
package index. Rerun it after pulling dependency changes or to install missing
packages. No activation or `sudo pip` is needed; `.venv` is ignored by Git.

If Red Hat's default `python3` is too old, install a newer Python with pip/venv
support through your system administrator or package manager, then select it:

```bash
bash setup_engineering_venv.sh python3.11
```

An existing `.venv` is reused; choosing a different interpreter does not rebuild
it. Move the old environment aside first when changing Python versions.
The launcher uses `ECMC_PYTHON` when explicitly set, otherwise the checkout's
`.venv`, otherwise the first `python` or `python3` with a supported Qt binding.
Use `unset ECMC_PYTHON` to return to automatic selection. Manually managed
environments can use `PySide6` instead of `PyQt5`.

The virtual environment supplies Python packages; Qt still needs the host's
graphics libraries and a graphical session (or X forwarding over SSH).

Simulation additionally requires:

- Podman, installed and running
- enough disk space for container images and build layers
- approximately 8 GB of Podman VM memory when compiling on macOS

On macOS, initialise and start the Podman Linux VM once:

```bash
podman machine init
podman machine start
```

The simulation image creates its own `/opt/ecmc-python` virtual environment and
installs the pinned packages from `ecmc_podman/container/requirements.txt` while
the image is built. This is separate from the optional host environment above.

Optional features require additional software:

- caQtDM for graphical control panels
- SSH for remote SDO access; the remote IOC host must provide its EtherCAT tools
- Tkinter only for the legacy Tk workspace

Start the Qt Studio after installing the applicable dependencies:

```bash
bash start_engineering_qt.sh
```

## Run the simulation checks

From the engineering checkout:

```sh
./run_simulation_tests.py
```

The runner checks the offline simulator suite first, builds or reuses
the local image once, then runs the slave inventory, motion/PLC IOC,
and open- and closed-loop slit suites in separate containers. It stops
on the first failure, shuts down the active IOC, and saves a JSON
summary and logs under `test-results/simulation/<run>/`. The slit stage
also writes its detailed JSON and JUnit reports.

Use repeatable `--suite offline|inventory|motion|slit` options to select
stages. For example, `./run_simulation_tests.py --suite motion` starts the
motion IOC and runs its Python checker. Use `--prebuilt` to reuse a
prepared image and `--slit-loop open` to run only the open-loop slit cases.
For a focused run with the same startup and reporting:

```sh
./run_simulation_tests.py --suite motion --motion-case variables --motion-case m2m
./run_simulation_tests.py --suite slit --slit-loop closed --slit-case noise
```

Use `--slit-test` for a test name substring and `--startat N` to resume
the numbered slit run with the same loop and filters. The JSON summary records
the selected suites and filters.

Install the runner dependencies into the same Python environment you use
to run the script:

```sh
python3 -m pip install jinja2 pyyaml pyepics
```

The offline checks use Jinja2 and PyYAML. The live stages need Podman,
and the motion/slit checkers need host PyEpics.
The live stages use localhost CA port 5064, so stop any existing container
using that port first. The inventory IOC can take several minutes to load.
Its startup timeout defaults to 15 minutes. Override it with
`--inventory-startup-timeout SECONDS`. Motion and slit keep their separate
`--startup-timeout` setting.

The runner's failure and shutdown paths can be checked without Podman or
an IOC:

```sh
python3 -m unittest discover -s tests -p test_run_simulation_tests.py
```

ECMC's `tests/test_homing_sequences.py` also checks successful position
compare using a deterministic simulated clock and output terminal. The
live slit fixture checks the missing-clock and cancellation path.

Keep the engineering root as the launcher and workspace entry point.
Put simulation fixtures and acceptance checks together under
`ecmccfg/tests/integration/<suite>/`, offline model checks under
`ecmccfg/tests/simulation/`, and generated reports under `test-results/`.
The existing manual launchers remain available for interactive debugging.

## Repository Layout

- `engineering/`: Qt and Tk workspace application and its Python modules
- `tests/`: engineering workspace and launcher regression checks
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
python -m unittest discover -s tests -p "test_*qt.py" -v
```

### Existing Tk workspace

Run from the repository root:

```bash
python -m engineering.ecmc_engineering_studio --startup examples/startup_local_hw.cmd
```

Example using a startup file from the `ecmccfg` submodule:

```bash
python -m engineering.ecmc_engineering_studio --startup ecmccfg/examples/PSI/lab_setup/stepper_bissc/startup_local_hw.cmd
```

You can also start without `--startup` and open a file from the GUI afterwards:

```bash
python -m engineering.ecmc_engineering_studio
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

### Simulation prerequisites and disk space

The host needs:

- **Podman**, with its service available. Native Linux runs Podman directly;
  macOS users must create and start a Podman machine (`podman machine init`,
  then `podman machine start`).
- **Bash** to run the container launch scripts.
- **Git** to clone this repository and initialize its submodules. Git is also
  used when building from the local ecmc, ecmccfg and ecmccomp checkouts.
- **Python 3** for local-source preparation and project-owned launchers such as
  `simulation/run.py`. EPICS and the generic hardware-model Python dependencies
  are already included in the container image.

The current `ecmc-dev:local` image occupies approximately **2.7 GB unpacked**;
allow about **3 GB per installed image**. Registry transfers and exported
compressed archives are normally smaller. Local builds need several additional
gigabytes temporarily for cached layers, downloaded packages and compiler
output, so substantially more than 3 GB of free Podman storage is recommended.
For builds inside a Podman VM, 8 GB of VM memory is a useful starting point.

Check the size of an installed image with:

```bash
podman images ecmc-dev:local
podman image inspect ecmc-dev:local --format '{{.Size}} bytes'
```

In the **Qt engineering tool**, open your project and click **Simulate**. Choose
the container image and select **No simulation plant**, **Generic hardware
simulation**, or a detected project plant, then click **Start**. **Use local
ecmc, ecmccfg and ecmccomp submodules** is enabled by default. With the generic
plant selected this runs:

```bash
./ecmc_podman/run-sim.sh --local-sources --image ecmc-dev:local --ioc-name SIMIOC --hardware-sim /path/to/project/IOC_startup.cmd
```

An IOC project can provide a coupled plant by adding `simulation/run.py`.
Engineering Qt detects that standard layout and uses the project launcher in
place of the generic launcher, passing the selected image, engineering checkout,
and local-sources choice. The start dialog can instead select the generic
hardware simulation for that run. See [Project simulation layout](docs/project-simulation.md)
for the directory structure and launcher/lifecycle contract.

Podman must be running. The first local-source launch builds `ecmc-dev:local`
from the existing submodules, including local edits, and may take several minutes
while the modules compile. Later launches reuse it
unless sources or build inputs change; changed inputs use cached build layers.
The checkouts are left untouched. Uncheck the local-sources option and choose
an existing image to run without building.
Unsaved project changes prompt you to **Save** or **Cancel** before launch. The
startup directory (including `cfg/`) is mounted read-only into the container.

The **Simulation — SIMIOC** view has separate **IOC shell**, **Terminal**, and
**Online tools** tabs. The IOC navigator is embedded by default; select **Open
online tools in a separate window** when starting the simulation to keep it detached.
The IOC shell provides startup output and simulation logs. The terminal opens a
general Bash shell in the same container; connect it after the container has started.
Type an IOC shell command such as `dbl` in the command field and press **Enter**
or **Send**. **Stop Simulation** sends `exit` to the IOC and waits for the IOC and
hardware models to shut down. Clicking **Simulate** while it is running returns
you to this tab. The **×** button on the tab closes it; if the simulation is
running, it sends `exit` and removes the tab once the process has stopped.

**Open Online Tools** opens the IOC navigator in a separate window using
`ecmc_cfg_tool/start_ioc.sh SIMIOC`, with Channel Access directed to localhost.
Wait for IOC startup to finish before connecting. The simulation launcher publishes
the local CA ports; only one simulation can run in this workspace at a time.

## Hardware-profile SIM IOC with Podman

The dedicated hardware-profile matrix in
`ecmccfg/tests/integration/hardware_profiles/ioc` covers CSV/CSP and
multiple encoder/stepper/servo terminal combinations. Run from this repository
after checking out `feature/hardware-abstraction` in the `ecmccfg` submodule:

```bash
bash start_hardware_profiles_sim.sh
```

The launcher uses `ecmc_podman/run-sim.sh --local-sources` and
`ecmc-dev:local` by default, building against the **currently checked-out**
`ecmc`, `ecmccfg` and `ecmccomp` submodules without updating them.
The IOC uses `SIM=1` and `SIMPROFILES:`, with no physical EtherCAT access.
The Python `--hardware-sim` model is **off by default**. Use\n`./start_hardware_profiles_sim.sh --hardware-sim` to start the supported\nmotor models for manual open-loop motion tests. The hardware-model subprocess\ndiscovers simulated slave PDOs under `SIMPROFILES:SIM:` and is supervised\nalongside the IOC. No automatic moves are issued.\n\nCurrent model coverage includes EL7031, EL7041, EL7047, EL7062 and standard\nEL72xx drives. EL7037 is not in the runner's stepper model list, and external\nencoder-terminal models do not automatically follow motor motion. Start with a\nself-contained open-loop stepper axis (for example S7031, axis 1), and leave\nother axes disabled until their feedback/limit behavior is understood. EL7062\nlimit input bits may need manual simulation initialization. This is not a full\nacceptance test for all ten axes.

```bash
bash start_hardware_profiles_sim.sh --dry-run
bash start_hardware_profiles_sim.sh --prebuilt --image ecmc-dev:local
bash start_hardware_profiles_sim.sh --no-ports
bash start_hardware_profiles_sim.sh --hardware-sim  # opt-in motor/encoder plant models
```

Stop the IOC by typing `exit` in its shell. Ensure no other simulator has
the localhost Channel Access ports 5064 TCP/UDP open. For offline profile
rendering tests (no IOC or Podman), run
`python3 ecmccfg/tests/test_hardware_profile_sim.py -v`.
The simulator config and test coverage are documented in
[`ecmccfg/tests/integration/hardware_profiles/README.md`](ecmccfg/tests/integration/hardware_profiles/README.md).

**Important:** The pinned `ecmccfg` submodule revision may not yet contain
the hardware-profile fixture. Switch that submodule to the feature branch,
then use `--local-sources` (default) so the newly selected code is built.
The branch's full startup test against ECMC RC1 still requires validation.

## Slit simulation with the current local submodules

The dedicated slit acceptance IOC is in
`ecmccfg/tests/integration/slit/ioc`. It runs with simulated EtherCAT and uses
the `SIMSLIT:` PV prefix. Run all commands below from the **ecmc_engineering
repository root**, with Podman running and the submodules initialized.

### One-command launcher

Use a Python environment with PyEpics installed (`python3 -m pip install pyepics`).
The convenience wrapper builds or reuses the local-source image, starts the IOC,
waits for initialization, runs the selected tests, and stops the IOC afterward:

```bash
# Run the complete suite:
./run_slit_tests.py

# Run only closed-loop feedback tests:
./run_slit_tests.py --case feedback --loop closed

# List selected tests without building or starting anything:
./run_slit_tests.py --list --case feedback

# Start the IOC and coupled plant for manual panel testing:
./run_slit_tests.py --plant-only
```

IOC output is saved separately in `test-results/slit/ioc-<timestamp>-<id>.log`,
so the terminal shows test progress and results without interleaved IOC alarms.
The launcher prints the log path and a `tail -f` command for following it in
another terminal. Startup failures and shutdown output are also kept there.
Use `--ioc-log /path/to/ioc.log` to choose a file (appends when reused), or
`--show-ioc-log` to also stream IOC output into the test terminal. Test-runner
errors, including local PyEpics errors, remain visible in the test terminal.

The wrapper sets localhost Channel Access settings for its child processes and
uses `SIMSLIT:`. It never starts the generic `--hardware-sim` models. Ctrl+C
stops the test runner/plant and then the IOC. Test failures are returned as a
nonzero exit status. The default image is `ecmc-dev:local`; use `--image TAG`
to choose another tag, or `--prebuilt` to skip checking/building local sources.
`--startup-timeout` controls the IOC initialization wait (default 120 seconds,
excluding build time). `--case`, `--test`, `--loop`, `--keep-going`, `--report`,
and `--startat` select test behavior; see `./run_slit_tests.py --help`.

To select a particular Python environment, use
`/path/to/python run_slit_tests.py` with the same options. The runner uses that
same interpreter. Stop any other IOC using the published local CA ports before
launching. For separate IOC and test terminals, follow the manual steps below.

### Open caQtDM axis panels

With the slit IOC running, open all four simulation axes:

```bash
./start_slit_axes_caqtdm.sh
```

This opens TR_LO (5), TR_HI (6), CENTERY (12), and GAPY (13) using
`SIMSLIT:` and localhost Channel Access. For one axis, pass its name and ID:

```bash
./start_axis_caqtdm.sh TR_LO 5
./start_axis_caqtdm.sh --ioc MYIOC --dev MYDEVICE --address 192.168.1.10 M1 1
```

The single-axis launcher defaults to IOC/device prefix `SIMSLIT`; `--dev`
defaults to the selected IOC. Its address defaults to `EPICS_CA_ADDR_LIST` or
localhost. Both scripts locate displays relative to the repository and run `caqtdm` from
PATH. On macOS, install the wrapper described below.
Use `--dry-run` to inspect commands without opening windows. During automated
tests, use panels only for observation. For manual motion, run the slit wrapper
with `--plant-only`.

### Open simulated hardware panels

With the slit IOC running, open caQtDM panels for all seven configured terminals:

```bash
./start_slit_hardware_caqtdm.sh
```

This opens EK1100 (0), EL2008 (5), EL5042 (9), EL7031 (12), EL7041-0052 (14),
EL7062_OPT (20), and EL2252 (21) for `SIMSLIT:`, master 0, on localhost.
The launcher uses the option-driven two-channel panel for EL7062_OPT.
It opens status/control displays only; it does not start another plant or
simulator. Use panels for observation during automated tests. For manual
interaction, keep `./run_slit_tests.py --plant-only` running.
`caqtdm` must be on PATH; `--dry-run` prints all launch commands.

### Start the slit IOC

In the first terminal:

```bash
./ecmc_podman/run-sim.sh \
  --local-sources \
  --image ecmc-dev:local \
  --ioc-name SIMSLIT \
  ecmccfg/tests/integration/slit/ioc/startup.cmd
```

This builds from the current local `ecmc`, `ecmccfg`, and `ecmccomp` checkouts,
including local edits and non-ignored new files. It does not fetch newer
revisions or change the checkouts. ECMC is built inside the pinned
`ECMC_makefile` wrapper supplied by the container recipe. The launcher adjusts
the startup's module versions to those installed in the image.

The first build may take a while. Later launches reuse the image when source
and build fingerprints match; changed inputs trigger a build using cached
layers. Restart the IOC with the same command after changing sources.

**Do not add `--hardware-sim` for this fixture.** The slit test runner supplies
its own coupled drive and encoder models. A second hardware simulator would
write to the same simulation fields.

Wait for IOC initialization to finish before running tests. Leave this terminal
open; type `exit` at its IOC prompt after stopping the tests or plant below.

### Run the acceptance tests

In a second terminal, use a Python environment with PyEpics installed and point
Channel Access at the local container:

```bash
# Install once in your chosen Python environment, if needed:
python3 -m pip install pyepics

export EPICS_CA_AUTO_ADDR_LIST=NO
export EPICS_CA_ADDR_LIST=127.0.0.1
```

Start with the closed-loop feedback tests:

```bash
python3 ecmccfg/tests/integration/slit/run.py \
  --prefix SIMSLIT: --run --case feedback --loop closed
```

Run the whole suite, which defaults to both open and closed loop:

```bash
python3 ecmccfg/tests/integration/slit/run.py \
  --prefix SIMSLIT: --run
```

Use `--case limits`, `--case brake`, or another case group to select a subset.
Omit `--run` to list the selected tests without accessing the IOC. The runner
checks the simulation-mode PV and dedicated fixture schema before issuing
writes. Reports default to `ecmccfg/test-results/slit/slit-report.json`.

### Run the plant for manual panel testing

Instead of the acceptance suite, start only its coupled hardware model:

```bash
python3 ecmccfg/tests/integration/slit/run.py \
  --prefix SIMSLIT: --plant-only
```

Keep this process running while using panels connected to `SIMSLIT:`. Stop it
with Ctrl+C before running the acceptance suite; run only one slit plant or
test runner at a time.

See the [slit fixture documentation](ecmccfg/tests/integration/slit/ioc/README.md)
and [slit test suite documentation](ecmccfg/tests/integration/slit/README.md)
for case groups, model behavior, and coverage limits. The local-source launcher
and GUI have automated checks; a full container build and live slit acceptance
run with these checkouts have not yet been verified.

## macOS caQtDM command

Linux installations keep using `caqtdm` from PATH. On macOS, install the small
wrapper once from the engineering repository root:

```bash
mkdir -p "$HOME/.local/bin"
cp tools/macos/caqtdm "$HOME/.local/bin/caqtdm"
chmod +x "$HOME/.local/bin/caqtdm"
export PATH="$HOME/.local/bin:$PATH"
```

Add `export PATH="$HOME/.local/bin:$PATH"` to `~/.zshrc` to keep it for new
terminals. Launch the panels from a terminal with that PATH so their child
scripts inherit it. The wrapper forwards all arguments to
`/Applications/caQtDM.app/Contents/MacOS/caQtDM`.

## Open the caQtDM main panel

On non-PSI systems, make the panels and helper files in `ecmccfg/qt`
accessible at **`/ioc/modules/qt`**, for example through a symlink to that
directory. Some panel actions use this absolute installation path to launch
other panels or helper scripts. Opening a panel directly from the checkout
does not guarantee that all of its launch buttons will work.

The **hardware overview may not work yet** outside the PSI environment:
its `start_ecmc_overview.py` helper lives in a separate repository and is not
provided by `ecmccfg/qt`. That helper and its dependencies must also be installed
and discoverable by the launcher.

Pass the IOC prefix to the generic main-panel launcher:

```bash
./start_main_caqtdm.sh MYIOC
# Local simulation:
./start_main_caqtdm.sh --address 127.0.0.1 SIMSLIT
```

A trailing colon on the prefix is accepted. The Channel Access address defaults
to `EPICS_CA_ADDR_LIST` when set, otherwise `127.0.0.1`, matching the axis
launcher. Broadcast discovery is disabled; use `--address HOST` for a remote IOC. `caqtdm` is launched from PATH. Use `--dry-run` to inspect the command.

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


# Test on redhat:

1057  podman build   -t localhost/ecmc-dev:portable   -f ecmc_podman/Containerfile   ecmc_podman/
 1058  cd ecmc_podman/
 1060  bash build-image.sh 
 1063  podman images
 1064  ./ecmc_podman/run-sim.sh --image ecmc-dev:portable --ioc-name SIMIOC --hardware-sim myProjects/MY_TEST_PROJECT/MY_TEST_PROJECT_startup.cmd
 1065  ./ecmc_podman/run-sim.sh --image ecmc-dev --ioc-name SIMIOC --hardware-sim myProjects/MY_TEST_PROJECT/MY_TEST_PROJECT_startup.cmd
 1066  ./ecmc_podman/run-sim.sh --image localhost/ecmc-dev --ioc-name SIMIOC --hardware-sim myProjects/MY_TEST_PROJECT/MY_TEST_PROJECT_startup.cmd
 1067  podman images
 1068  ./ecmc_podman/run-sim.sh   --image localhost/ecmc-dev:v11.2.0_RC1   --ioc-name SIMIOC   --hardware-sim   myProjects/MY_TEST_PROJECT/MY_TEST_PROJECT_startup.cmd

 ./ecmc_podman/run-sim.sh   --image localhost/ecmc-dev:v11.2.0_RC1   --ioc-name SIMIOC   --hardware-sim   myProjects/MY_TEST_PROJECT/MY_TEST_PROJECT_startup.cmd

needs to use v11.2.0_RC1

This needs some cleanup to make it seemless from ecmc_engineering.. 
Also teh virtual environment needs to be somehow generated
