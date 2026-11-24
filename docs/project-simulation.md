# Project simulation layout

Engineering Qt automatically uses a project-owned simulation when an IOC
project has this layout:

```text
IOC-PROJECT/
├── IOC-PROJECT_startup.cmd (or .script)
└── simulation/
    ├── config.json         project simulation metadata
    ├── run.py              required launcher
    ├── prepare.py          optional staging
    ├── plant.py            optional coupled plant
    ├── check.py            optional acceptance checks
    └── README.md           project-specific model and usage notes
```

Only `simulation/run.py` is required for discovery. The other names are the
standard locations for their respective responsibilities; projects can omit
what they do not need.

When the directory exists, Engineering Qt shows it as a **Simulation** branch
in the Project objects tree. Source and documentation files can be opened,
edited, and saved there; hidden files, `__pycache__`, and `.pyc` files are not
shown.

To enable the Qt **Online tools** button for a project plant, provide its IOC
name in `simulation/config.json`:

```json
{
  "ioc_name": "SIMPGM",
  "ioc_shell": true
}
```

The tools connect to that IOC on localhost. Without `ioc_name`, the project
launcher remains usable but Online Tools are omitted because Qt cannot safely
infer the project's prefix.

Set `ioc_shell` to `true` only when `run.py` forwards its stdin unchanged to the
IOC shell. Engineering Qt then shows the command field and supports commands
such as `dbl`. It writes each submitted command to the project launcher's stdin.

## Launcher contract

`run.py` owns the complete simulation lifecycle: staging any modified copy of
the IOC configuration, starting the IOC, starting the coupled plant, forwarding
output, and stopping all child processes and containers. It must accept:

```text
--engineering PATH   ecmc_engineering checkout containing ecmc_podman
--image TAG          container image selected in Engineering Qt
--local-sources      optional; build/reuse the checkout's local submodules
```

The launcher must run in the foreground and return nonzero if startup or the
plant fails. It should print a clear readiness line after both IOC and plant are
ready. It must handle `SIGINT`, stop its plant and IOC, remove its container,
and then exit. Engineering Qt sends that signal when the user stops or closes
the Simulation tab.

The project directory is the launcher's working directory. Launcher output is
shown in the Simulation tab. The launcher must forward IOC and plant output to
stdout; if it also keeps an IOC log, it should tee the same stream to that file
rather than redirecting it away from stdout. A launcher should use unbuffered
output and must not daemonize itself. Simulation-only generated files should go
in an ignored directory such as `.simulation/`; source IOC configuration should
remain unchanged.

The Simulation tab shows the project launcher's console only. Generic container
Terminal and `SIMIOC` Online Tools tabs are intentionally omitted because a
project launcher owns its container name and IOC prefix. A project can document
or provide its own control-panel launcher in `simulation/README.md`.

When `simulation/run.py` is absent, Engineering Qt uses its normal generic
`ecmc_podman/run-sim.sh` simulation instead.

The Start Simulation dialog always offers **No simulation plant** and **Generic
hardware simulation**. When a project launcher is present it also offers
**Project plant (simulation/run.py)**, selected by default. Choosing no plant
runs the IOC without `--hardware-sim`; choosing the generic plant uses the normal
`run-sim.sh --hardware-sim` path. Exactly one choice is applied.
