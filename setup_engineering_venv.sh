#!/usr/bin/env bash
# Usage: bash setup_engineering_venv.sh [python-interpreter]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -gt 1 ]; then
  echo "Usage: bash setup_engineering_venv.sh [python-interpreter]" >&2
  exit 1
fi
PYTHON_BIN="${1:-python3}"
VENV="${SCRIPT_DIR}/.venv"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}. Pass an installed Python 3.9+ interpreter as the argument." >&2
  exit 1
fi
"${PYTHON_BIN}" -c 'import sys; sys.exit("Python 3.9+ is required. Try: bash setup_engineering_venv.sh python3.11" if sys.version_info < (3, 9) else 0)'
if [ ! -f "${SCRIPT_DIR}/ecmc_cfg_tool/qt_runtime.sh" ]; then
  echo "Initialize online tools first: git submodule update --init --recursive" >&2
  exit 1
fi
if [ ! -e "${VENV}" ]; then
  if ! "${PYTHON_BIN}" -m venv "${VENV}"; then
    echo "Could not create ${VENV}. Install venv/pip support for ${PYTHON_BIN}, then retry." >&2
    exit 1
  fi
fi
if [ ! -x "${VENV}/bin/python" ] || [ ! -f "${VENV}/pyvenv.cfg" ]; then
  echo "${VENV} is not a usable virtual environment. Move it aside and rerun this script." >&2
  exit 1
fi
"${VENV}/bin/python" -c 'import sys; sys.exit("Existing .venv needs Python 3.9+. Move it aside and rerun setup." if sys.version_info < (3, 9) else 0)'
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
"${VENV}/bin/python" -m pip check
"${VENV}/bin/python" -c 'from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg; import epics, numpy, yaml, jinja2; print("Python dependencies verified.")'
echo "Ready. Start with: bash \"${SCRIPT_DIR}/start_engineering_qt.sh\""
