#!/usr/bin/env bash
# Open an ECMC axis panel using the bundled displays.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ioc=SIMSLIT
dev=
address=${EPICS_CA_ADDR_LIST:-127.0.0.1}
dry=0
usage() {
    cat <<'EOF'
Usage: start_axis_caqtdm.sh [options] AXIS_NAME AXIS_ID
  --ioc PREFIX       IOC prefix, without trailing colon (default: SIMSLIT)
  --dev PREFIX       Axis device prefix (default: IOC prefix)
  --address HOST     Channel Access address (default: EPICS_CA_ADDR_LIST or 127.0.0.1)
  --dry-run          Print command without opening a panel
  -h, --help         Show help
Examples:
  ./start_axis_caqtdm.sh TR_LO 5
  ./start_axis_caqtdm.sh --ioc MYIOC --dev MYDEVICE --address 192.168.1.10 M1 1
The caqtdm executable or wrapper must be on PATH.
EOF
}
fail() { echo "start-axis: $*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry=1; shift ;;
        --ioc|--dev|--address)
            (($# >= 2)) && [[ -n $2 ]] || fail "$1 requires a value"
            case "$1" in
                --ioc) ioc=${2%:} ;;
                --dev) dev=${2%:} ;;
                --address) address=$2 ;;
            esac
            shift 2 ;;
        --) shift; break ;;
        -*) fail "unknown option: $1" ;;
        *) break ;;
    esac
done
(($# == 2)) || { usage >&2; exit 1; }
axis=$1
id=$2
dev=${dev:-$ioc}
[[ $id =~ ^[0-9]+$ ]] || fail 'AXIS_ID must be a nonnegative integer'
for value in "$ioc" "$dev" "$axis"; do
    [[ -n $value && $value != *[[:space:],\;\=\$]* ]] || fail 'Prefixes and axis names must not contain whitespace or macro separators'
done
binary=caqtdm
command -v "$binary" >/dev/null 2>&1 || fail 'caqtdm not found; add its executable or wrapper to PATH'
[[ -f $here/ecmccfg/qt/ecmcAxis.ui ]] || fail 'Missing axis display; initialize the ecmccfg submodule'
export EPICS_CA_AUTO_ADDR_LIST=NO
export EPICS_CA_ADDR_LIST=$address
export EPICS_DISPLAY_PATH="$here/ecmccfg/qt${EPICS_DISPLAY_PATH:+:$EPICS_DISPLAY_PATH}"
cd "$here/ecmccfg/qt"
cmd=("$binary" -macro "IOC=$ioc,DEV=$dev,Axis=$axis,AX_ID=$id" ecmcAxis.ui)
if ((dry)); then
    printf 'EPICS_CA_AUTO_ADDR_LIST=NO EPICS_CA_ADDR_LIST=%q ' "$address"
    printf '%q ' "${cmd[@]}"
    printf '\n'
else
    exec "${cmd[@]}"
fi
