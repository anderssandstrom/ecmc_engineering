#!/usr/bin/env bash
# Open hardware status panels for the dedicated slit simulation fixture.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
dry=0
case "${1:-}" in
    -h|--help)
        echo 'Usage: start_slit_hardware_caqtdm.sh [--dry-run]'
        echo 'Opens all seven slit SIM hardware panels (SIMSLIT, master 0, localhost).'
        echo 'Requires a running slit IOC; this does not start additional hardware models.'
        exit 0 ;;
    --dry-run) dry=1; shift ;;
esac
(($# == 0)) || { echo 'Expected only optional --dry-run' >&2; exit 1; }
fail() { echo "slit-hardware: $*" >&2; exit 1; }
binary=caqtdm
command -v "$binary" >/dev/null 2>&1 || fail 'caqtdm not found; add its executable or wrapper to PATH'
export EPICS_CA_AUTO_ADDR_LIST=NO
export EPICS_CA_ADDR_LIST=127.0.0.1
export EPICS_DISPLAY_PATH="$here/ecmccfg/qt${EPICS_DISPLAY_PATH:+:$EPICS_DISPLAY_PATH}"
cd "$here/ecmccfg/qt"
ids=(000 005 009 012 014 020 021)
types=(EK1100 EL2008 EL5042 EL7031 EL7041-0052 EL7062_OPT EL2252)
panels=(EK1100 Ex20x8 EL5042 EL70x1 EL70x1 EX72XX_OPT_2ch EL2252)
for panel in "${panels[@]}"; do
    [[ -f ecmc${panel}.ui ]] || { echo "Missing display: ecmc${panel}.ui" >&2; exit 1; }
done
for i in 0 1 2 3 4 5 6; do
    cmd=("$binary" -macro "IOC=SIMSLIT,SYS=SIMSLIT,MasterID=0,SlaveID=${ids[$i]},HWType=${types[$i]},CH2_OFF=0,PANEL=${panels[$i]}" "ecmcGenericSlaveOverview.ui")
    if ((dry)); then
        printf 'EPICS_CA_AUTO_ADDR_LIST=NO EPICS_CA_ADDR_LIST=127.0.0.1 '
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}" &
    fi
done
