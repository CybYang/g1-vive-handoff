#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing $PYTHON_BIN; run ./scripts/setup_g1_env.sh first." >&2
  exit 2
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/g1_preflight.py"
"$PYTHON_BIN" -m pytest -q \
  "$PROJECT_ROOT/tests/test_g1_deployment.py" \
  "$PROJECT_ROOT/tests/test_vive_focus_udp_receiver.py" \
  "$PROJECT_ROOT/tests/test_vive_openxr_hand.py" \
  "$PROJECT_ROOT/tests/test_inspire_g2_pose_adapter.py" \
  "$PROJECT_ROOT/tests/test_inspire_g2_v2_objective.py" \
  "$PROJECT_ROOT/tests/test_inspire_g2_hardware_mapping.py" \
  "$PROJECT_ROOT/tests/test_inspire_g2_vive_hardware.py"
