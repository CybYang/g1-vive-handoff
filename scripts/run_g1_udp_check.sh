#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=g1_runtime_common.sh
source "$SCRIPT_DIR/g1_runtime_common.sh"

SIDE="${1:-left}"
case "$SIDE" in
  left) PORT="$LEFT_UDP_PORT" ;;
  right) PORT="$RIGHT_UDP_PORT" ;;
  *) echo "usage: $0 [left|right]" >&2; exit 2 ;;
esac

ARGS=(
  "$G1_PYTHON" "$G1_PROJECT_ROOT/scripts/vive_focus_udp_receiver.py"
  --bind "$BIND_ADDRESS"
  --port "$PORT"
  --status-interval 1
)
if [[ -n "$VIVE_SENDER_IP" ]]; then
  ARGS+=(--sender-ip "$VIVE_SENDER_IP")
fi

echo "Read-only UDP diagnostic on $SIDE port $PORT; no serial port will be opened."
exec "${ARGS[@]}"
