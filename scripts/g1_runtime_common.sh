#!/usr/bin/env bash
set -euo pipefail

G1_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G1_PROJECT_ROOT="$(cd "$G1_SCRIPT_DIR/.." && pwd)"
G1_RUNTIME_CONFIG="$G1_PROJECT_ROOT/config/g1_runtime.env"

if [[ ! -f "$G1_RUNTIME_CONFIG" ]]; then
  echo "Missing $G1_RUNTIME_CONFIG" >&2
  echo "Copy config/g1_runtime.env.example to config/g1_runtime.env and edit it first." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$G1_RUNTIME_CONFIG"

: "${BIND_ADDRESS:=0.0.0.0}"
: "${LEFT_UDP_PORT:=5005}"
: "${RIGHT_UDP_PORT:=5006}"
: "${VIVE_SENDER_IP:=}"
: "${LEFT_SERIAL_PORT:=/dev/ttyUSB0}"
: "${RIGHT_SERIAL_PORT:=/dev/ttyUSB1}"
: "${LEFT_HAND_ID:=1}"
: "${RIGHT_HAND_ID:=1}"
: "${HARDWARE_RATE_HZ:=10}"
: "${MAX_STEP_UNITS:=2}"
: "${HEALTH_CHECK_PERIOD_S:=2}"
: "${STATUS_INTERVAL_S:=0.5}"
: "${LEFT_CPUSET:=}"
: "${RIGHT_CPUSET:=}"
: "${OMP_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"

export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
export PYTHONUNBUFFERED=1

G1_PYTHON="$G1_PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$G1_PYTHON" ]]; then
  echo "Missing package environment: $G1_PYTHON" >&2
  echo "Run ./scripts/setup_g1_env.sh first." >&2
  exit 2
fi

mkdir -p "$G1_PROJECT_ROOT/log"

g1_append_sender_filter() {
  local -n destination=$1
  if [[ -n "$VIVE_SENDER_IP" ]]; then
    destination+=(--sender-ip "$VIVE_SENDER_IP")
  fi
}

g1_run_with_optional_cpuset() {
  local cpuset=$1
  shift
  if [[ -n "$cpuset" ]]; then
    exec taskset --cpu-list "$cpuset" "$@"
  fi
  exec "$@"
}
