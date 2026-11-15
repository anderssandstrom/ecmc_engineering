#!/usr/bin/env bash
# Open the ECMC main panel using the bundled displays.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
address=${EPICS_CA_ADDR_LIST:-127.0.0.1}
dry=0
usage() {
    cat <<'EOF'
Usage: start_main_caqtdm.sh [options] PREFIX
  --address HOST     CA address (default: EPICS_CA_ADDR_LIST or 127.0.0.1)
  --dry-run          Print command without opening a panel
  -h, --help         Show help
Examples:
  ./start_main_caqtdm.sh MYIOC
  ./start_main_caqtdm.sh --address 127.0.0.1 SIMSLIT
A trailing colon in PREFIX is accepted. Broadcast discovery is disabled;
use --address for a remote IOC.
The caqtdm executable or wrapper must be on PATH.
EOF
}
fail() { echo "start-main: $*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry=1; shift ;;
        --address)
            (($# >= 2)) && [[ -n $2 ]] || fail "$1 requires a value"
            address=$2
            shift 2 ;;
        --) shift; break ;;
        -*) fail "unknown option: $1" ;;
        *) break ;;
    esac
done
(($# == 1)) || { usage >&2; exit 1; }
ioc=${1%:}
[[ -n $ioc && $ioc != *[[:space:],\;\=\$]* ]] || fail 'PREFIX must not be empty or contain whitespace or macro separators'
binary=caqtdm
command -v "$binary" >/dev/null 2>&1 || fail 'caqtdm not found; add its executable or wrapper to PATH'
[[ -f $here/ecmccfg/qt/ecmcMain.ui ]] || fail 'Missing main display; initialize the ecmccfg submodule'
if [[ -n $address ]]; then
    export EPICS_CA_AUTO_ADDR_LIST=NO
    export EPICS_CA_ADDR_LIST=$address
fi
export EPICS_DISPLAY_PATH="$here/ecmccfg/qt${EPICS_DISPLAY_PATH:+:$EPICS_DISPLAY_PATH}"
cd "$here/ecmccfg/qt"
cmd=("$binary" -macro "IOC=$ioc" ecmcMain.ui)
if ((dry)); then
    if [[ -n $address ]]; then
        printf 'EPICS_CA_AUTO_ADDR_LIST=NO EPICS_CA_ADDR_LIST=%q ' "$address"
    fi
    printf '%q ' "${cmd[@]}"
    printf '\n'
else
    exec "${cmd[@]}"
fi
