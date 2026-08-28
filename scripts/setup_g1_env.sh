#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
ARCHITECTURE="$($PYTHON_BIN -c 'import platform; print(platform.machine())')"
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
  echo "This handoff was validated for Python 3.10; found $PYTHON_VERSION." >&2
  exit 2
fi

echo "Creating isolated G1 hand environment (Python $PYTHON_VERSION, $ARCHITECTURE)."
echo "The existing g1ik environment will not be modified."

if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
  "$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
fi

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
"$VENV_PYTHON" -m pip install --upgrade "pip>=25,<27" "setuptools>=75,<81" wheel
"$VENV_PYTHON" -m pip install --only-binary=:all: \
  -r "$PROJECT_ROOT/requirements-g1.txt"
"$VENV_PYTHON" -m pip install --no-deps --editable "$PROJECT_ROOT"

if [[ ! -f "$PROJECT_ROOT/config/g1_runtime.env" ]]; then
  cp "$PROJECT_ROOT/config/g1_runtime.env.example" \
    "$PROJECT_ROOT/config/g1_runtime.env"
  echo "Created config/g1_runtime.env from the template; edit serial paths before live use."
fi

"$VENV_PYTHON" "$PROJECT_ROOT/scripts/g1_preflight.py" --strict-g1

echo
echo "Environment ready. Next read docs/G1_MIGRATION_HANDOFF_ZH.md."
