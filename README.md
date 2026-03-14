# ecmc Engineering

`ecmc Engineering` is a small desktop tool for inspecting and editing ecmc startup projects with linked YAML and PLC files.

## Clone

Recommended, clone with submodules from the start:

```bash
git clone --recurse-submodules <repo-url>
cd ecmc_engineering
```

You can also clone first and fetch submodules afterwards:

```bash
git clone <repo-url>
cd ecmc_engineering
git submodule update --init --recursive
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

## Repository Layout

- `ecmc_engineering_studio.py`: main application
- `compat_dataclasses.py`: compatibility helper
- `ecmccfg/`: submodule with scripts, hardware, and startup framework
- `ecmccomp/`: submodule with component definitions
- `examples/`: local example startup files and configs

## Start

Run from the repository root:

```bash
python ecmc_engineering_studio.py --startup examples/startup_local_hw.cmd
```

You can also open another startup file, for example one inside the `ecmccfg` submodule:

```bash
python ecmc_engineering_studio.py --startup ecmccfg/examples/PSI/lab_setup/stepper_bissc/startup_local_hw.cmd
```

## Notes

- The tool expects `ecmccfg` and `ecmccomp` to be present.
- On some Python installations, `tkinter` may be missing. In that case the GUI will not start until Tk support is installed.
