#!/usr/bin/env bash
# Start the dedicated ecmccfg hardware-profile matrix as a Podman SIM IOC.
# No physical EtherCAT or automatic moves. Run from any working directory.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
podman_runner="$here/ecmc_podman/run-sim.sh"
startup="$here/ecmccfg/tests/integration/hardware_profiles/ioc/startup.cmd"
image="ecmc-dev:local"
prebuilt=0
dry_run=0
no_ports=0
hardware_sim=0
container_name=""
ioc_name="SIMPROFILES"

usage() {
    cat <<'EOF'
Usage: ./start_hardware_profiles_sim.sh [options]

Start ecmccfg/tests/integration/hardware_profiles/ioc/startup.cmd using the
ecmc_podman SIM runtime. By default, build/reuse the current local submodules.

Options:
  --image IMAGE          Container image (default: ecmc-dev:local)
  --prebuilt             Use an existing image; skip local-source build
  --container-name NAME  Optional Podman container name
  --no-ports             Do not publish local EPICS Channel Access ports
  --hardware-sim         Run supported Python motor/feedback models (opt-in)
  --dry-run              Print the build/container commands without running
  -h, --help             Show this help

The IOC is configured with SIM=1 and has PV prefix SIMPROFILES:.
By default, no external hardware model runs. --hardware-sim starts the supported\nmodels (not all terminals are modeled). No moves are issued automatically.
Type 'exit' in the IOC shell to stop it.
EOF
}

fail() { echo "hardware-profiles-sim: $*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --image|--container-name)
            (($# >= 2)) && [[ -n "$2" ]] || fail "$1 requires a value"
            case "$1" in
                --image) image=$2 ;;
                --container-name) container_name=$2 ;;
            esac
            shift 2 ;;
        --prebuilt) prebuilt=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --no-ports) no_ports=1; shift ;;
        --hardware-sim) hardware_sim=1; shift ;;
        *) fail "unknown argument: $1 (try --help)" ;;
    esac
done

[[ -f "$podman_runner" ]] ||
    fail "ecmc_podman submodule missing; run: git submodule update --init --recursive"
[[ -f "$startup" ]] ||
    fail "hardware-profile IOC missing in ecmccfg submodule; check out feature/hardware-abstraction in ecmccfg"
for file in axis_csv.yaml axis_csp.yaml axis_el7062.yaml; do
    [[ -f "${startup%/*}/$file" ]] || fail "missing simulation config: $file"
done

cmd=("$podman_runner")
if (( !prebuilt )); then
    # Build against the local worktrees, including an unmerged ecmccfg branch.
    cmd+=(--local-sources)
fi
cmd+=(--image "$image" --ioc-name "$ioc_name")
if ((hardware_sim)); then
    # The hardware model discovers SIM PDO records under this prefix.
    cmd+=(--hardware-sim --sim-prefix "$ioc_name:SIM:")
fi
if [[ -n "$container_name" ]]; then
    cmd+=(--container-name "$container_name")
fi
if ((no_ports)); then
    cmd+=(--no-ports)
fi
if ((dry_run)); then
    cmd+=(--dry-run)
fi
cmd+=("$startup")

echo "Hardware-profile SIM IOC: $startup"
echo "Image: $image ($([[ $prebuilt == 1 ]] && echo prebuilt || echo local sources))"
echo "Prefix: $ioc_name:"
echo "Hardware models: $([[ $hardware_sim == 1 ]] && echo enabled || echo disabled)"
printf 'Command: '
printf '%q ' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
