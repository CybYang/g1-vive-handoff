#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "${1:-}" != "RUN_LIVE_RETARGETING" ]]; then
  echo "Live bimanual output was NOT started." >&2
  echo "Validate each hand independently first. Then run:" >&2
  echo "  $0 RUN_LIVE_RETARGETING" >&2
  exit 2
fi
shift

mkdir -p "$PROJECT_ROOT/log"
LEFT_LOG="$PROJECT_ROOT/log/g1_left_live.log"
RIGHT_LOG="$PROJECT_ROOT/log/g1_right_live.log"

left_pid=""
right_pid=""
cleaned=0
cleanup() {
  if [[ "$cleaned" -eq 1 ]]; then
    return
  fi
  cleaned=1
  for pid in "$left_pid" "$right_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
  wait "$left_pid" "$right_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

"$SCRIPT_DIR/run_g1_left_live.sh" RUN_LIVE_RETARGETING "$@" \
  > >(tee -a "$LEFT_LOG") 2>&1 &
left_pid=$!
"$SCRIPT_DIR/run_g1_right_live.sh" RUN_LIVE_RETARGETING "$@" \
  > >(tee -a "$RIGHT_LOG") 2>&1 &
right_pid=$!

echo "Started independent live processes: left PID=$left_pid, right PID=$right_pid"
echo "Logs: $LEFT_LOG and $RIGHT_LOG"
echo "Ctrl+C stops both processes."

set +e
wait -n "$left_pid" "$right_pid"
status=$?
set -e
echo "One hand process exited with status $status; stopping the other."
cleanup
exit "$status"
