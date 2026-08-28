#!/usr/bin/env bash
set -euo pipefail

echo "Network interfaces:"
ip -br address
echo

echo "UDP listeners on the planned hand ports:"
ss -lunp | awk 'NR == 1 || /:5005|:5006/'
echo

echo "Persistent serial identifiers:"
if [[ -d /dev/serial/by-id ]]; then
  find /dev/serial/by-id -maxdepth 1 -type l -printf '%f -> %l\n' | sort
else
  echo "/dev/serial/by-id is not present."
fi
echo

echo "Serial identifiers by physical USB path:"
if [[ -d /dev/serial/by-path ]]; then
  find /dev/serial/by-path -maxdepth 1 -type l -printf '%f -> %l\n' | sort
else
  echo "/dev/serial/by-path is not present."
fi
echo

echo "Current ttyUSB/ttyACM devices:"
find /dev -maxdepth 1 \( -name 'ttyUSB*' -o -name 'ttyACM*' \) -printf '%p\n' | sort
