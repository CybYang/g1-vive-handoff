#!/usr/bin/env python3
# ruff: noqa: I001
"""VIVE/OpenXR hands -> current DexRetargeting V2 -> physical Inspire G2.

This is a hardware-only entry point.  It has no SAPIEN dependency and never
sends a command until the operator supplies the explicit confirmation string.
Each hand uses an independent optimizer and serial controller.  In bimanual
mode the two optimizers run concurrently, then the two serial writes occur in
a deterministic sequence.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dex_retargeting.retargeting_config import RetargetingConfig
from scripts.inspire_g2_hardware import (
    LEFT_MAPPING_PATH,
    RIGHT_MAPPING_PATH,
    InspireG2LiveController,
    JointMapping,
    load_joint_mapping,
    qpos_to_angle_set,
)
from scripts.inspire_g2_vive_runtime import (
    HandRuntime,
    build_hand_runtime,
    objective_status,
    selected_sides,
    solve_hand_runtime,
)
from scripts.vive_focus_udp_receiver import (
    DEFAULT_BIND_ADDRESS,
    DEFAULT_PORT,
    PacketValidationError,
    ViveFocusUDPReceiver,
    _sequence_delta,
)


HARDWARE_CONFIRMATION = "RUN_LIVE_RETARGETING"


def _positive(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _unit_interval(value: str) -> float:
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return number


def _non_negative(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return number


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drive one or both physical Inspire G2 hands from VIVE/OpenXR "
            "using the current enhanced DexRetargeting V2 objective."
        )
    )
    parser.add_argument("--bind", default=DEFAULT_BIND_ADDRESS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--sender-ip",
        default="",
        help="accept only this Unity sender IP (default: accept any sender)",
    )
    parser.add_argument(
        "--hand-side",
        choices=("right", "left", "both"),
        default="right",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--left-port",
        default="",
        help=(
            "left USB-RS485 port; default /dev/ttyUSB0 for left-only and "
            "bimanual operation"
        ),
    )
    parser.add_argument(
        "--right-port",
        default="",
        help=(
            "right USB-RS485 port; default /dev/ttyUSB0 for right-only or "
            "/dev/ttyUSB1 for bimanual operation"
        ),
    )
    parser.add_argument("--left-hand-id", type=int, default=1)
    parser.add_argument("--right-hand-id", type=int, default=1)
    parser.add_argument(
        "--left-mapping",
        type=Path,
        default=LEFT_MAPPING_PATH,
    )
    parser.add_argument(
        "--right-mapping",
        type=Path,
        default=RIGHT_MAPPING_PATH,
    )
    parser.add_argument("--hardware-rate-hz", type=_positive, default=10.0)
    parser.add_argument(
        "--max-step-units",
        type=_positive_int,
        default=20,
        help="maximum change per packet in 0.1-degree units (default: 20)",
    )
    parser.add_argument(
        "--health-check-period-s",
        type=_positive,
        default=2.0,
    )
    parser.add_argument(
        "--confirm-hardware",
        default="",
        metavar=HARDWARE_CONFIRMATION,
        help="required exact acknowledgement before serial ports are opened",
    )

    # These defaults intentionally match inspire_g2_vive_sim.py.  Both entry
    # points call the same runtime functions in inspire_g2_vive_runtime.py.
    parser.set_defaults(objective="v2")
    parser.add_argument("--scaling-factor", type=_positive, default=1.15)
    parser.add_argument("--low-pass-alpha", type=_unit_interval, default=0.1)
    parser.add_argument("--landmark-alpha", type=_unit_interval, default=0.5)
    parser.add_argument("--status-interval", type=_positive, default=1.0)
    parser.add_argument("--v2-joint-weight", type=_non_negative, default=0.02)
    parser.add_argument(
        "--v2-flexion-reference-weight", type=_non_negative, default=1.0
    )
    parser.add_argument(
        "--v2-ring-open-weight-multiplier", type=_positive, default=8.0
    )
    parser.add_argument("--v2-ring-open-fraction", type=float, default=0.10)
    parser.add_argument("--v2-ring-release-fraction", type=float, default=0.35)
    parser.add_argument(
        "--v2-pinky-open-weight-multiplier", type=_positive, default=8.0
    )
    parser.add_argument("--v2-pinky-open-fraction", type=float, default=0.10)
    parser.add_argument("--v2-pinky-release-fraction", type=float, default=0.35)
    parser.add_argument(
        "--v2-yaw-reference-weight", type=_non_negative, default=4.0
    )
    parser.add_argument(
        "--v2-middle-yaw-reference-weight", type=_non_negative, default=12.0
    )
    parser.add_argument(
        "--v2-thumb-yaw-reference-weight",
        type=_non_negative,
        default=0.0,
        help=(
            "experimental CMC1 opposition reference weight; zero preserves "
            "the current V2 behavior"
        ),
    )
    parser.add_argument(
        "--v2-thumb-cmc-reference-weight", type=_non_negative, default=12.0
    )
    parser.add_argument(
        "--v2-thumb-ip-reference-weight", type=_non_negative, default=8.0
    )
    parser.add_argument(
        "--v2-thumb-cmc-open-elevation-deg", type=float, default=20.0
    )
    parser.add_argument(
        "--v2-thumb-cmc-opposed-elevation-deg", type=float, default=55.0
    )
    parser.add_argument(
        "--v2-thumb-opposition-open-ratio", type=_positive, default=1.55
    )
    parser.add_argument(
        "--v2-thumb-opposition-opposed-ratio",
        type=_non_negative,
        default=0.85,
    )
    parser.add_argument(
        "--v2-thumb-opposition-exponent", type=_positive, default=0.7
    )
    parser.add_argument(
        "--v2-thumb-flexion-reference-weight",
        type=_non_negative,
        default=None,
        help="legacy combined thumb reference; normally leave unset",
    )
    parser.add_argument(
        "--v2-yaw-coupling-weight", type=_non_negative, default=0.01
    )
    parser.add_argument("--v2-collision-weight", type=_non_negative, default=0.3)
    parser.add_argument(
        "--v2-collision-min-distance", type=_non_negative, default=0.014
    )
    parser.add_argument("--v2-ring-debug", action="store_true")
    parser.add_argument("--v2-thumb-debug", action="store_true")

    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65_535:
        parser.error("--port must be in the range 0..65535")
    for name in ("left_hand_id", "right_hand_id"):
        if not 1 <= getattr(args, name) <= 254:
            parser.error(f"--{name.replace('_', '-')} must be in the range 1..254")
    for finger in ("ring", "pinky"):
        open_fraction = getattr(args, f"v2_{finger}_open_fraction")
        release_fraction = getattr(args, f"v2_{finger}_release_fraction")
        multiplier = getattr(args, f"v2_{finger}_open_weight_multiplier")
        if not 0.0 <= open_fraction < release_fraction <= 1.0:
            parser.error(
                f"{finger} fractions must satisfy 0 <= open < release <= 1"
            )
        if multiplier < 1.0:
            parser.error(
                f"--v2-{finger}-open-weight-multiplier must be at least 1"
            )
    if not (
        0.0
        <= args.v2_thumb_cmc_open_elevation_deg
        < args.v2_thumb_cmc_opposed_elevation_deg
        <= 90.0
    ):
        parser.error(
            "thumb CMC elevations must satisfy 0 <= open < opposed <= 90 degrees"
        )
    if not (
        args.v2_thumb_opposition_opposed_ratio
        < args.v2_thumb_opposition_open_ratio
    ):
        parser.error(
            "thumb opposition ratios must satisfy 0 <= opposed < open"
        )
    if args.confirm_hardware != HARDWARE_CONFIRMATION:
        parser.error(
            f"physical output requires --confirm-hardware {HARDWARE_CONFIRMATION}"
        )
    return args


def _resolved_ports(
    hand_side: str,
    left_port: str = "",
    right_port: str = "",
) -> dict[str, str]:
    sides = selected_sides(hand_side)
    if hand_side == "left":
        ports = {"left": left_port or "/dev/ttyUSB0"}
    elif hand_side == "right":
        ports = {"right": right_port or "/dev/ttyUSB0"}
    else:
        ports = {
            "left": left_port or "/dev/ttyUSB0",
            "right": right_port or "/dev/ttyUSB1",
        }
    if len(set(ports.values())) != len(sides):
        raise ValueError(
            "bimanual mode requires two distinct serial ports; use one "
            "USB-RS485 adapter per hand"
        )
    return ports


def _mapping_path(side: str, args: argparse.Namespace) -> Path:
    return args.left_mapping if side == "left" else args.right_mapping


def _hand_id(side: str, args: argparse.Namespace) -> int:
    return args.left_hand_id if side == "left" else args.right_hand_id


def _validate_mapping(
    side: str,
    mapping: tuple[JointMapping, ...],
    joint_names: list[str],
) -> None:
    missing = [item.urdf_joint for item in mapping if item.urdf_joint not in joint_names]
    if missing:
        raise ValueError(f"{side} mapping contains unknown URDF joints: {missing}")
    if len(mapping) != 13:
        raise ValueError(f"{side} mapping must contain exactly 13 hardware joints")


@dataclass
class HardwareHand:
    runtime: HandRuntime
    mapping: tuple[JointMapping, ...]
    controller: InspireG2LiveController
    solve_errors: int = 0
    last_error: str = ""
    last_desired: list[int] | None = None
    last_sent: list[int] | None = None


def _solve_one(
    hand: HardwareHand,
    openxr_hand: Any,
    args: argparse.Namespace,
    now: float,
) -> list[int] | None:
    try:
        qpos = solve_hand_runtime(hand.runtime, openxr_hand, args, now)
        if qpos is None:
            return None
        desired = qpos_to_angle_set(
            qpos,
            hand.runtime.retargeting.joint_names,
            hand.mapping,
        )
    except (RuntimeError, ValueError, KeyError, FloatingPointError) as exc:
        # Expected tracking/optimizer/mapping failures hold this hand instead
        # of sending partial data. Unexpected programming errors still stop
        # the process and close both serial ports in ``main``.
        hand.solve_errors += 1
        hand.last_error = f"{type(exc).__name__}: {exc}"
        return None
    hand.last_desired = desired
    return desired


def _rate_hz(timestamps: deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    duration = timestamps[-1] - timestamps[0]
    return 0.0 if duration <= 0 else (len(timestamps) - 1) / duration


def _build_hands(args: argparse.Namespace) -> list[HardwareHand]:
    ports = _resolved_ports(args.hand_side, args.left_port, args.right_port)
    hands: list[HardwareHand] = []
    try:
        for side in selected_sides(args.hand_side):
            runtime = build_hand_runtime(side, args)
            mapping = load_joint_mapping(_mapping_path(side, args))
            _validate_mapping(side, mapping, runtime.retargeting.joint_names)
            controller = InspireG2LiveController(
                port=ports[side],
                hand_id=_hand_id(side, args),
                update_rate_hz=args.hardware_rate_hz,
                max_step_units=args.max_step_units,
                health_check_period_s=args.health_check_period_s,
            )
            hands.append(HardwareHand(runtime, mapping, controller))
    except Exception:
        for hand in hands:
            hand.controller.close()
        raise
    return hands


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    RetargetingConfig.set_default_urdf_dir(
        PROJECT_ROOT / "assets" / "robots" / "hands"
    )
    ports = _resolved_ports(args.hand_side, args.left_port, args.right_port)

    print("PHYSICAL INSPIRE G2 OUTPUT ENABLED.")
    print(
        f"Using {args.hand_side} hand(s), current V2 objective, "
        f"udp://{args.bind}:{args.port}."
    )
    for side in selected_sides(args.hand_side):
        print(f"[{side}] serial={ports[side]} hand_id={_hand_id(side, args)}")
    if "right" in selected_sides(args.hand_side):
        print(
            "WARNING: the right mapping is nominal. For its first physical "
            "checkout use --max-step-units 2 and verify every joint direction."
        )
    if args.sender_ip:
        print(f"Accepting Unity packets only from {args.sender_ip}.")

    hands = _build_hands(args)
    executor = (
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="g2-slsqp")
        if len(hands) == 2
        else None
    )
    if executor is not None:
        print("Bimanual SLSQP: two independent optimizers running concurrently.")

    receive_times: deque[float] = deque()
    last_sequence_by_sender: dict[str, int] = {}
    last_status = 0.0
    malformed = 0
    reordered = 0
    filtered = 0
    coalesced_total = 0
    last_frame = None

    try:
        with ViveFocusUDPReceiver(
            args.bind,
            args.port,
            timeout_s=0.05,
        ) as receiver:
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

                    if executor is None:
                        desired_by_hand = [
                            _solve_one(
                                hands[0],
                                getattr(frame, hands[0].runtime.side),
                                args,
                                now,
                            )
                        ]
                    else:
                        futures = [
                            executor.submit(
                                _solve_one,
                                hand,
                                getattr(frame, hand.runtime.side),
                                args,
                                now,
                            )
                            for hand in hands
                        ]
                        desired_by_hand = [future.result() for future in futures]

                    # Serial writes remain sequential even though solving is
                    # parallel. Distinct adapters avoid bus/port contention.
                    for hand, desired in zip(hands, desired_by_hand, strict=True):
                        if desired is None:
                            continue
                        sent = hand.controller.send_if_due(desired)
                        if sent is not None:
                            hand.last_sent = sent

                if now - last_status >= args.status_interval:
                    if last_frame is None:
                        print("Waiting for Unity/VIVE UDP packets; hands hold position.")
                    else:
                        print(
                            f"seq={last_frame.sequence} "
                            f"rx={_rate_hz(receive_times):.1f}Hz "
                            f"malformed={malformed} old={reordered} "
                            f"filtered={filtered} coalesced={coalesced_total}"
                        )
                        for hand in hands:
                            runtime = hand.runtime
                            age_ms = (
                                float("inf")
                                if runtime.last_valid_frame == 0.0
                                else (now - runtime.last_valid_frame) * 1000.0
                            )
                            print(
                                f"[{runtime.side}] solve={runtime.last_solve_ms:.1f}ms "
                                f"hold_age={age_ms:.0f}ms accepted={runtime.accepted} "
                                f"invalid={runtime.invalid} errors={hand.solve_errors} "
                                f"sent={hand.controller.sent_count}; "
                                f"{objective_status(runtime.retargeting.optimizer.last_objective_terms)}"
                            )
                            if hand.last_error:
                                print(f"[{runtime.side}] last solve error: {hand.last_error}")
                            if args.v2_ring_debug:
                                print(runtime.last_ring_debug)
                            if args.v2_thumb_debug:
                                print(runtime.last_thumb_debug)
                                if hand.last_desired is not None:
                                    desired_thumb = hand.last_desired[10:13]
                                    sent_thumb = (
                                        None
                                        if hand.last_sent is None
                                        else hand.last_sent[10:13]
                                    )
                                    actual_thumb = hand.controller.read_actual()[10:13]
                                    print(
                                        f"[{runtime.side}] angleSet "
                                        "thumb(yaw,cmc,ip) "
                                        f"desired={desired_thumb} sent={sent_thumb} "
                                        f"actual={actual_thumb}"
                                    )
                    last_status = now
    except KeyboardInterrupt:
        print("Stopped by user; serial ports closed and last hand poses held.")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for hand in hands:
            hand.controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
