#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build=(--local-sources)
if [[ ${1:-} == --prebuilt ]]; then
    build=()
    shift
fi

exec "$root/ecmc_podman/run-sim.sh" "${build[@]}" --image ecmc-dev:local \
    --ioc-name SIMMOTION --hardware-sim "$@" \
    "$root/ecmccfg/tests/integration/motion/ioc/startup.cmd"
