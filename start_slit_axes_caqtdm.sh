#!/usr/bin/env bash
# Open the two physical and two virtual axes of the dedicated slit SIM IOC.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
dry=0
case "${1:-}" in
    -h|--help)
        echo 'Usage: start_slit_axes_caqtdm.sh [--dry-run]'
        echo 'Opens SIMSLIT TR_LO (5), TR_HI (6), CENTERY (12), GAPY (13) on localhost.'
        exit 0 ;;
    --dry-run) dry=1; shift ;;
esac
(($# == 0)) || { echo 'Expected only optional --dry-run' >&2; exit 1; }
names=(TR_LO TR_HI CENTERY GAPY)
ids=(5 6 12 13)
# Validate every command before opening any panels.
for i in 0 1 2 3; do
    "$here/start_axis_caqtdm.sh" --dry-run --address 127.0.0.1 "${names[$i]}" "${ids[$i]}" >/dev/null
done
for i in 0 1 2 3; do
    if ((dry)); then
        "$here/start_axis_caqtdm.sh" --dry-run --address 127.0.0.1 "${names[$i]}" "${ids[$i]}"
    else
        "$here/start_axis_caqtdm.sh" --address 127.0.0.1 "${names[$i]}" "${ids[$i]}" &
    fi
done
