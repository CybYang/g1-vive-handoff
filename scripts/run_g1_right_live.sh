#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=g1_runtime_common.sh
source "$SCRIPT_DIR/g1_runtime_common.sh"

if [[ "${1:-}" != "RUN_LIVE_RETARGETING" ]]; then
  echo "Live right-hand output was NOT started." >&2
  echo "After the read-only checks, run:" >&2
  echo "  $0 RUN_LIVE_RETARGETING [extra options]" >&2
  exit 2
fi
shift

ARGS=(
  "$G1_PYTHON" "$G1_PROJECT_ROOT/scripts/inspire_g2_vive_hardware.py"
  --bind "$BIND_ADDRESS"
  --port "$RIGHT_UDP_PORT"
  --hand-side right
  --right-port "$RIGHT_SERIAL_PORT"
  --right-hand-id "$RIGHT_HAND_ID"
  --hardware-rate-hz "$HARDWARE_RATE_HZ"
  --max-step-units "$MAX_STEP_UNITS"
  --health-check-period-s "$HEALTH_CHECK_PERIOD_S"
  --status-interval "$STATUS_INTERVAL_S"
  --confirm-hardware RUN_LIVE_RETARGETING
)
g1_append_sender_filter ARGS
ARGS+=("$@")

echo "Starting LIVE right hand: UDP $RIGHT_UDP_PORT -> $RIGHT_SERIAL_PORT (ID $RIGHT_HAND_ID)."
echo "Stop with Ctrl+C. Keep an operator at the emergency stop."
g1_run_with_optional_cpuset "$RIGHT_CPUSET" "${ARGS[@]}"
