#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=g1_runtime_common.sh
source "$SCRIPT_DIR/g1_runtime_common.sh"

if [[ "${1:-}" != "RUN_LIVE_RETARGETING" ]]; then
  echo "Live left-hand output was NOT started." >&2
  echo "After the read-only checks, run:" >&2
  echo "  $0 RUN_LIVE_RETARGETING [extra options]" >&2
  exit 2
fi
shift

ARGS=(
  "$G1_PYTHON" "$G1_PROJECT_ROOT/scripts/inspire_g2_vive_hardware.py"
  --bind "$BIND_ADDRESS"
  --port "$LEFT_UDP_PORT"
  --hand-side left
  --left-port "$LEFT_SERIAL_PORT"
  --left-hand-id "$LEFT_HAND_ID"
  --hardware-rate-hz "$HARDWARE_RATE_HZ"
  --max-step-units "$MAX_STEP_UNITS"
  --health-check-period-s "$HEALTH_CHECK_PERIOD_S"
  --status-interval "$STATUS_INTERVAL_S"
  --confirm-hardware RUN_LIVE_RETARGETING
)
g1_append_sender_filter ARGS
ARGS+=("$@")

echo "Starting LIVE left hand: UDP $LEFT_UDP_PORT -> $LEFT_SERIAL_PORT (ID $LEFT_HAND_ID)."
echo "Stop with Ctrl+C. Keep an operator at the emergency stop."
g1_run_with_optional_cpuset "$LEFT_CPUSET" "${ARGS[@]}"
