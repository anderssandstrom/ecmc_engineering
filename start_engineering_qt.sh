#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "${SCRIPT_DIR}/ecmc_cfg_tool/qt_runtime.sh" ]; then
  echo "Initialize online tools: git submodule update --init --recursive" >&2
  exit 1
fi
source "${SCRIPT_DIR}/ecmc_cfg_tool/qt_runtime.sh"
PYTHON_BIN="$(find_qt_python)" || { print_qt_python_error; exit 1; }
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/ecmc_engineering_qt.py" "$@"
