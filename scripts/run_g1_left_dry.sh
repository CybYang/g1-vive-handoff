#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=g1_runtime_common.sh
source "$SCRIPT_DIR/g1_runtime_common.sh"

ARGS=(
  "$G1_PYTHON" "$G1_PROJECT_ROOT/scripts/inspire_g2_vive_headless.py"
  --bind "$BIND_ADDRESS"
  --port "$LEFT_UDP_PORT"
  --hand-side left
  --status-interval "$STATUS_INTERVAL_S"
)
g1_append_sender_filter ARGS
ARGS+=("$@")

echo "Dry left pipeline on UDP $LEFT_UDP_PORT; no serial port will be opened."
g1_run_with_optional_cpuset "$LEFT_CPUSET" "${ARGS[@]}"
