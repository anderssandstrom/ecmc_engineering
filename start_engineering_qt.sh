#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "${SCRIPT_DIR}/ecmc_cfg_tool/qt_runtime.sh" ]; then
  echo "Initialize online tools: git submodule update --init --recursive" >&2
  exit 1
fi
source "${SCRIPT_DIR}/ecmc_cfg_tool/qt_runtime.sh"
if [ -n "${ECMC_PYTHON:-}" ]; then
  PYTHON_BIN="${ECMC_PYTHON}"
elif [ -d "${SCRIPT_DIR}/.venv" ]; then
  PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="$(find_qt_python)" || {
    print_qt_python_error
    echo "Set up the engineering environment: bash \"${SCRIPT_DIR}/setup_engineering_venv.sh\"" >&2
    exit 1
  }
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 || ! _ecmc_qt_python_has_binding "${PYTHON_BIN}"; then
  echo "Selected Python is missing or has no Qt binding: ${PYTHON_BIN}" >&2
  echo "Run bash \"${SCRIPT_DIR}/setup_engineering_venv.sh\" to prepare .venv, or correct ECMC_PYTHON if set." >&2
  exit 1
fi
export ECMC_PYTHON="${PYTHON_BIN}"
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" exec "${PYTHON_BIN}" -m engineering.ecmc_engineering_qt "$@"
