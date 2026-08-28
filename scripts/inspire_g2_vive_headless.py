#!/usr/bin/env python3
"""Run live VIVE -> V2 DexRetargeting -> angleSet mapping without serial I/O.

This G1 migration diagnostic consumes the same packet, coordinate transform,
optimizer and mapping as physical operation.  It deliberately never creates
an InspireG2LiveController, never opens /dev/tty*, and never sends a command.
"""

from __future__ import annotations

import socket
import sys
import time
from collections import deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dex_retargeting.retargeting_config import RetargetingConfig
from scripts.inspire_g2_hardware import load_joint_mapping, qpos_to_angle_set
from scripts.inspire_g2_vive_hardware import (
    HARDWARE_CONFIRMATION,
    _mapping_path,
    _parse_args,
    _validate_mapping,
)
from scripts.inspire_g2_vive_runtime import (
    build_hand_runtime,
    objective_status,
    selected_sides,
    solve_hand_runtime,
)
from scripts.vive_focus_udp_receiver import (
    PacketValidationError,
    ViveFocusUDPReceiver,
    _sequence_delta,
)


def _rate_hz(timestamps: deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    duration = timestamps[-1] - timestamps[0]
    return 0.0 if duration <= 0 else (len(timestamps) - 1) / duration


def _tracker_summary(frame) -> str:
    if not frame.trackers:
        return "none"
    return ",".join(
        f"id={tracker.tracker_id}:valid={int(tracker.pose_valid)}"
        for tracker in frame.trackers
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Reuse the exact physical-runtime defaults while satisfying its parse-time
    # guard internally.  This script never calls _build_hands or constructs a
    # serial controller, so the acknowledgement cannot enable motion here.
    arguments.extend(["--confirm-hardware", HARDWARE_CONFIRMATION])
    args = _parse_args(arguments)

    RetargetingConfig.set_default_urdf_dir(
        PROJECT_ROOT / "assets" / "robots" / "hands"
    )
    runtimes = {
        side: build_hand_runtime(side, args)
        for side in selected_sides(args.hand_side)
    }
    mappings = {
        side: load_joint_mapping(_mapping_path(side, args))
        for side in runtimes
    }
    for side, runtime in runtimes.items():
        _validate_mapping(side, mappings[side], runtime.retargeting.joint_names)

    print("HEADLESS DRY RUN: serial and robot output are disabled.")
    print(
        f"Using {args.hand_side} V2 pipeline; "
        f"listening on udp://{args.bind}:{args.port}."
    )
    if args.sender_ip:
        print(f"Accepting VIVE packets only from {args.sender_ip}.")

    receive_times: deque[float] = deque()
    last_sequence_by_sender: dict[str, int] = {}
    last_status = 0.0
    last_frame = None
    malformed = 0
    reordered = 0
    filtered = 0
    coalesced_total = 0
    desired_by_side: dict[str, list[int]] = {}

    try:
        with ViveFocusUDPReceiver(args.bind, args.port, timeout_s=0.05) as receiver:
            while True:
                now = time.monotonic()
                try:
                    frame, coalesced = receiver.receive_latest()
                except socket.timeout:
                    frame = None
                except PacketValidationError as exc:
                    malformed += 1
                    frame = None
                    if now - last_status >= args.status_interval:
                        print(f"Rejected malformed UDP packet: {exc}")

                if frame is not None:
                    sender_ip = frame.sender[0] if frame.sender else "unknown"
                    if args.sender_ip and sender_ip != args.sender_ip:
                        filtered += 1
                        frame = None
                    else:
                        delta = _sequence_delta(
                            last_sequence_by_sender.get(sender_ip), frame.sequence
                        )
                        if delta is not None and delta <= 0:
                            reordered += 1
                            frame = None
                        else:
                            last_sequence_by_sender[sender_ip] = frame.sequence

                if frame is not None:
                    coalesced_total += coalesced
                    frame_time = frame.received_monotonic or now
                    receive_times.append(frame_time)
                    while receive_times and frame_time - receive_times[0] > 2.0:
                        receive_times.popleft()
                    last_frame = frame
                    for side, runtime in runtimes.items():
                        qpos = solve_hand_runtime(
                            runtime,
                            getattr(frame, side),
                            args,
                            now,
                        )
                        if qpos is None:
                            continue
                        desired_by_side[side] = qpos_to_angle_set(
                            qpos,
                            runtime.retargeting.joint_names,
                            mappings[side],
                        )

                if now - last_status >= args.status_interval:
                    if last_frame is None:
                        print("Waiting for VIVE UDP packets; no serial output exists.")
                    else:
                        print(
                            f"seq={last_frame.sequence} "
                            f"rx={_rate_hz(receive_times):.1f}Hz "
                            f"trackers=[{_tracker_summary(last_frame)}] "
                            f"malformed={malformed} old={reordered} "
                            f"filtered={filtered} coalesced={coalesced_total}"
                        )
                        for side, runtime in runtimes.items():
                            print(
                                f"[{side}] solve={runtime.last_solve_ms:.1f}ms "
                                f"accepted={runtime.accepted} invalid={runtime.invalid} "
                                f"angleSet={desired_by_side.get(side, 'waiting')}; "
                                f"{objective_status(runtime.retargeting.optimizer.last_objective_terms)}"
                            )
                    last_status = now
    except KeyboardInterrupt:
        print("Headless dry run stopped; no serial port was opened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
