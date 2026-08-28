#!/usr/bin/env python3
"""Receive VIVE Focus Vision OpenXR hands and trackers over UDP.

The receiver is intentionally independent from retargeting and robot output.
Running this file only opens a UDP socket; it never opens ``/dev/ttyUSB*`` and
cannot move the physical Inspire hand.

Protocol version 1 contains two fixed-size OpenXR hands and an optional array
of VIVE Ultimate Tracker poses. Each hand has 26 XYZ positions (78 flattened
floats) and 26 validity flags. Hand and tracker positions are expressed in
Unity world coordinates in metres.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROTOCOL_VERSION = 1
DEFAULT_BIND_ADDRESS = "0.0.0.0"
DEFAULT_PORT = 5005
OPENXR_JOINT_COUNT = 26
POSITION_VALUE_COUNT = OPENXR_JOINT_COUNT * 3
MAX_PACKET_BYTES = 65_507

TRACKER_MIN_ID = 0
TRACKER_MAX_ID = 4
TRACKER_POSITION_VALUE_COUNT = 3
TRACKER_ROTATION_VALUE_COUNT = 4
FULL_POSE_TRACKING_MASK = 3

MAX_SEQUENCE = 2_147_483_647
SEQUENCE_MODULUS = MAX_SEQUENCE + 1

OPENXR_JOINT_NAMES = (
    "palm",
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_metacarpal",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
)

# MediaPipe order is wrist, four thumb joints, and four joints per finger.
# OpenXR additionally contains a palm point and metacarpals for the four long
# fingers, so those five points are omitted here.
OPENXR_TO_MEDIAPIPE_21 = np.asarray(
    [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25],
    dtype=np.int64,
)


class PacketValidationError(ValueError):
    """Raised when a UDP payload does not match the tracking protocol."""


@dataclass(frozen=True)
class HandJoints:
    """One OpenXR hand from a received tracking frame."""

    tracked: bool
    positions: np.ndarray
    valid: np.ndarray

    def as_mediapipe21(self) -> tuple[np.ndarray, np.ndarray]:
        """Return positions and validity in MediaPipe's 21-joint order."""
        positions = self.positions[OPENXR_TO_MEDIAPIPE_21].copy()
        valid = self.valid[OPENXR_TO_MEDIAPIPE_21].copy()
        if not self.tracked:
            valid[:] = False
        return positions, valid


@dataclass(frozen=True)
class TrackerPose:
    """One VIVE Ultimate Tracker observation from the current frame."""

    tracker_id: int
    is_tracked: bool
    tracking_state: int
    position: np.ndarray
    rotation: np.ndarray

    @property
    def pose_valid(self) -> bool:
        """Whether this frame contains a complete, currently tracked 6DoF pose."""
        return self.is_tracked and (
            self.tracking_state & FULL_POSE_TRACKING_MASK
        ) == FULL_POSE_TRACKING_MASK


@dataclass(frozen=True)
class ViveFocusFrame:
    """A validated protocol frame and its Ubuntu receive metadata."""

    sequence: int
    timestamp_unix_ms: int
    space: str
    units: str
    left: HandJoints
    right: HandJoints
    trackers: tuple[TrackerPose, ...] = ()
    sender: tuple[str, int] | None = None
    received_monotonic: float | None = None
    payload_bytes: int = 0


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PacketValidationError(f"{field} must be a JSON object")
    return value


def _require_integer(
    value: Any,
    field: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PacketValidationError(f"{field} must be an integer")
    if value < minimum:
        raise PacketValidationError(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise PacketValidationError(f"{field} must be <= {maximum}")
    return value


def _require_boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise PacketValidationError(f"{field} must be true or false")
    return value


def _parse_finite_vector(value: Any, field: str, length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise PacketValidationError(f"{field} must contain exactly {length} numbers")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        raise PacketValidationError(f"{field} contains a non-numeric value")
    vector = np.asarray(value, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise PacketValidationError(f"{field} contains NaN or infinity")
    return vector


def _parse_positions(value: Any, field: str) -> np.ndarray:
    positions = _parse_finite_vector(value, field, POSITION_VALUE_COUNT)
    return positions.reshape(OPENXR_JOINT_COUNT, 3)


def _parse_validity(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != OPENXR_JOINT_COUNT:
        raise PacketValidationError(
            f"{field} must contain exactly {OPENXR_JOINT_COUNT} flags"
        )
    validity = []
    for item in value:
        if isinstance(item, bool):
            validity.append(item)
        elif isinstance(item, int) and item in (0, 1):
            validity.append(bool(item))
        else:
            raise PacketValidationError(f"{field} flags must be 0, 1, or boolean")
    return np.asarray(validity, dtype=bool)


def _parse_hand(value: Any, field: str) -> HandJoints:
    hand = _require_mapping(value, field)
    return HandJoints(
        tracked=_require_boolean(hand.get("tracked"), f"{field}.tracked"),
        positions=_parse_positions(hand.get("positions"), f"{field}.positions"),
        valid=_parse_validity(hand.get("valid"), f"{field}.valid"),
    )


def _parse_tracker(value: Any, field: str) -> TrackerPose:
    tracker = _require_mapping(value, field)
    return TrackerPose(
        tracker_id=_require_integer(
            tracker.get("id"),
            f"{field}.id",
            minimum=TRACKER_MIN_ID,
            maximum=TRACKER_MAX_ID,
        ),
        is_tracked=_require_boolean(
            tracker.get("is_tracked"), f"{field}.is_tracked"
        ),
        tracking_state=_require_integer(
            tracker.get("tracking_state"), f"{field}.tracking_state"
        ),
        position=_parse_finite_vector(
            tracker.get("position"),
            f"{field}.position",
            TRACKER_POSITION_VALUE_COUNT,
        ),
        rotation=_parse_finite_vector(
            tracker.get("rotation"),
            f"{field}.rotation",
            TRACKER_ROTATION_VALUE_COUNT,
        ),
    )


def _parse_trackers(value: Any, field: str = "trackers") -> tuple[TrackerPose, ...]:
    if not isinstance(value, list):
        raise PacketValidationError(f"{field} must be a JSON array")

    trackers = tuple(
        _parse_tracker(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    tracker_ids = [tracker.tracker_id for tracker in trackers]
    if len(tracker_ids) != len(set(tracker_ids)):
        raise PacketValidationError(f"{field} contains duplicate tracker ids")
    return trackers


def decode_packet(
    payload: bytes,
    *,
    sender: tuple[str, int] | None = None,
    received_monotonic: float | None = None,
) -> ViveFocusFrame:
    """Decode and validate one UTF-8 JSON UDP datagram."""
    try:
        document = json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PacketValidationError("packet is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise PacketValidationError(f"packet is not valid JSON: {exc.msg}") from exc

    root = _require_mapping(document, "packet")
    version = _require_integer(root.get("v"), "v", minimum=1)
    if version != PROTOCOL_VERSION:
        raise PacketValidationError(
            f"unsupported protocol version {version}; expected {PROTOCOL_VERSION}"
        )

    space = root.get("space")
    if space != "unity_world":
        raise PacketValidationError(
            f"space must be 'unity_world', received {space!r}"
        )
    units = root.get("units")
    if units != "m":
        raise PacketValidationError(f"units must be 'm', received {units!r}")

    return ViveFocusFrame(
        sequence=_require_integer(
            root.get("seq"), "seq", maximum=MAX_SEQUENCE
        ),
        timestamp_unix_ms=_require_integer(
            root.get("timestamp_unix_ms"), "timestamp_unix_ms"
        ),
        space=space,
        units=units,
        left=_parse_hand(root.get("left"), "left"),
        right=_parse_hand(root.get("right"), "right"),
        trackers=_parse_trackers(root["trackers"]) if "trackers" in root else (),
        sender=sender,
        received_monotonic=received_monotonic,
        payload_bytes=len(payload),
    )


class FrameJSONLLogger:
    """Append normalized frames and receive diagnostics to a JSONL file."""

    def __init__(self, path: str | Path, max_megabytes: float = 100.0) -> None:
        if max_megabytes <= 0:
            raise ValueError("max_megabytes must be positive")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_megabytes * 1024 * 1024)
        self.bytes_written = self.path.stat().st_size if self.path.exists() else 0
        self.file = self.path.open("a", encoding="utf-8")
        self.limit_reached = self.bytes_written >= self.max_bytes

    @staticmethod
    def _hand_payload(hand: HandJoints) -> dict[str, Any]:
        return {
            "tracked": hand.tracked,
            "positions": hand.positions.reshape(-1).tolist(),
            "valid": hand.valid.astype(np.uint8).tolist(),
        }

    @staticmethod
    def _tracker_payload(tracker: TrackerPose) -> dict[str, Any]:
        return {
            "id": tracker.tracker_id,
            "is_tracked": tracker.is_tracked,
            "tracking_state": tracker.tracking_state,
            "position": tracker.position.tolist(),
            "rotation": tracker.rotation.tolist(),
        }

    def write(self, frame: ViveFocusFrame, sequence_delta: int | None) -> bool:
        """Write one frame, returning False after the size limit is reached."""
        if self.limit_reached:
            return False

        received_unix_ms = time.time_ns() // 1_000_000
        record = {
            "receiver": {
                "received_unix_ms": received_unix_ms,
                "sender_ip": frame.sender[0] if frame.sender else None,
                "sender_port": frame.sender[1] if frame.sender else None,
                "payload_bytes": frame.payload_bytes,
                "sequence_delta": sequence_delta,
                "sender_to_receiver_ms": (
                    received_unix_ms - frame.timestamp_unix_ms
                ),
                "left_valid": int(frame.left.valid.sum()),
                "right_valid": int(frame.right.valid.sum()),
                "tracker_count": len(frame.trackers),
                "tracker_pose_valid": {
                    str(tracker.tracker_id): tracker.pose_valid
                    for tracker in frame.trackers
                },
            },
            "packet": {
                "v": PROTOCOL_VERSION,
                "seq": frame.sequence,
                "timestamp_unix_ms": frame.timestamp_unix_ms,
                "space": frame.space,
                "units": frame.units,
                "left": self._hand_payload(frame.left),
                "right": self._hand_payload(frame.right),
                "trackers": [
                    self._tracker_payload(tracker) for tracker in frame.trackers
                ],
            },
        }
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        encoded_size = len(line.encode("utf-8"))
        if self.bytes_written + encoded_size > self.max_bytes:
            self.limit_reached = True
            return False
        self.file.write(line)
        self.file.flush()
        self.bytes_written += encoded_size
        return True

    def close(self) -> None:
        self.file.close()


class ViveFocusUDPReceiver:
    """Small reusable blocking UDP receiver for VIVE tracking frames."""

    def __init__(
        self,
        bind_address: str = DEFAULT_BIND_ADDRESS,
        port: int = DEFAULT_PORT,
        timeout_s: float | None = None,
        receive_buffer_bytes: int = 1 << 20,
    ) -> None:
        if not 0 <= port <= 65_535:
            raise ValueError("port must be in the range 0..65535")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive or None")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer_bytes
        )
        self.socket.settimeout(timeout_s)
        try:
            self.socket.bind((bind_address, port))
        except Exception:
            self.socket.close()
            raise

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.socket.getsockname()[:2]
        return str(host), int(port)

    def receive(self) -> ViveFocusFrame:
        payload, sender = self.socket.recvfrom(MAX_PACKET_BYTES)
        return decode_packet(
            payload,
            sender=(str(sender[0]), int(sender[1])),
            received_monotonic=time.monotonic(),
        )

    def receive_latest(self) -> tuple[ViveFocusFrame, int]:
        """Receive once, then discard queued older datagrams to minimize lag.

        Returns the newest decoded frame and the number of older queued
        datagrams that were coalesced.  This is intended for live rendering or
        robot control, where replaying a UDP backlog is worse than dropping it.
        """
        payload, sender = self.socket.recvfrom(MAX_PACKET_BYTES)
        coalesced = 0
        original_timeout = self.socket.gettimeout()
        self.socket.setblocking(False)
        try:
            while True:
                payload, sender = self.socket.recvfrom(MAX_PACKET_BYTES)
                coalesced += 1
        except BlockingIOError:
            pass
        finally:
            self.socket.settimeout(original_timeout)

        return (
            decode_packet(
                payload,
                sender=(str(sender[0]), int(sender[1])),
                received_monotonic=time.monotonic(),
            ),
            coalesced,
        )

    def close(self) -> None:
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _tracking_status(hand: HandJoints) -> str:
    if not hand.tracked:
        return "not-tracked"
    return f"tracked {int(hand.valid.sum())}/{OPENXR_JOINT_COUNT} valid"


def _tracker_status(tracker: TrackerPose) -> str:
    tracked = str(tracker.is_tracked).lower()
    if tracker.pose_valid:
        position = ",".join(f"{value:.3f}" for value in tracker.position)
        pose = f"valid pos=({position})"
    else:
        pose = "invalid"
    return (
        f"id={tracker.tracker_id} is_tracked={tracked} "
        f"state={tracker.tracking_state} pose={pose}"
    )


def _format_status(
    frame: ViveFocusFrame,
    rate_hz: float,
    estimated_lost: int,
    malformed: int,
    sequence_delta: int | None,
) -> str:
    sender = "unknown"
    if frame.sender is not None:
        sender = f"{frame.sender[0]}:{frame.sender[1]}"
    tracker_status = "; ".join(
        _tracker_status(tracker) for tracker in frame.trackers
    )
    if not tracker_status:
        tracker_status = "none"
    return (
        f"seq={frame.sequence}  seq-delta={sequence_delta or 0}  "
        f"rx={rate_hz:5.1f} Hz  seq-gaps~={estimated_lost}  "
        f"packet={frame.payload_bytes} B  malformed={malformed}  "
        f"left={_tracking_status(frame.left)}  "
        f"right={_tracking_status(frame.right)}  "
        f"trackers=[{tracker_status}]  "
        f"updated_ms={frame.timestamp_unix_ms}  from={sender}"
    )


def _rate_hz(receive_times: Sequence[float]) -> float:
    if len(receive_times) < 2:
        return 0.0
    duration = receive_times[-1] - receive_times[0]
    if duration <= 0:
        return 0.0
    return (len(receive_times) - 1) / duration


def _sequence_delta(previous: int | None, current: int) -> int | None:
    """Return a forward delta while recognizing the signed-int sequence wrap."""
    if previous is None:
        return None
    delta = current - previous
    if delta >= 0:
        return delta
    wrapped_delta = delta + SEQUENCE_MODULUS
    if wrapped_delta <= SEQUENCE_MODULUS // 2:
        return wrapped_delta
    return delta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Receive VIVE Focus Vision OpenXR hands and trackers over UDP. "
            "This diagnostic does not control the physical robot hand."
        )
    )
    parser.add_argument("--bind", default=DEFAULT_BIND_ADDRESS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--sender-ip",
        default="",
        help="accept only this headset IP (default: accept any sender)",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="seconds between status lines; minimum 1 second",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="exit successfully after the first valid frame",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="append validated frames and receive metadata to this JSONL file",
    )
    parser.add_argument(
        "--log-max-mb",
        type=float,
        default=100.0,
        help="stop appending after the log reaches this size (default: 100)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.status_interval < 1.0:
        raise SystemExit("--status-interval must be at least 1 second")
    if args.log_max_mb <= 0:
        raise SystemExit("--log-max-mb must be positive")

    receive_times: deque[float] = deque()
    last_frame: ViveFocusFrame | None = None
    last_sequence_by_sender: dict[str, int] = {}
    estimated_lost = 0
    malformed = 0
    filtered = 0
    last_status_time = time.monotonic()
    last_warning_time = last_status_time - args.status_interval
    last_wait_notice = last_status_time
    packet_logger = None
    log_limit_reported = False

    if args.log_file:
        try:
            packet_logger = FrameJSONLLogger(args.log_file, args.log_max_mb)
        except OSError as exc:
            raise SystemExit(f"Cannot open JSONL log {args.log_file}: {exc}") from exc

    try:
        with ViveFocusUDPReceiver(args.bind, args.port, timeout_s=1.0) as receiver:
            host, port = receiver.address
            print(
                f"Listening on udp://{host}:{port}; "
                "network diagnostic only, robot serial output is disabled."
            )
            if args.sender_ip:
                print(f"Accepting packets only from {args.sender_ip}.")
            if packet_logger is not None:
                print(
                    f"Appending validated frames to {packet_logger.path} "
                    f"(limit {args.log_max_mb:g} MiB)."
                )

            while True:
                try:
                    frame = receiver.receive()
                except socket.timeout:
                    now = time.monotonic()
                    if last_frame is None and now - last_wait_notice >= 5.0:
                        print(
                            "Still waiting for a valid packet. Check the Ubuntu "
                            "IP, UDP port, headset app, and firewall."
                        )
                        last_wait_notice = now
                    continue
                except PacketValidationError as exc:
                    malformed += 1
                    now = time.monotonic()
                    if now - last_warning_time >= args.status_interval:
                        print(f"Rejected malformed UDP packet: {exc}")
                        last_warning_time = now
                    continue

                if args.sender_ip and frame.sender[0] != args.sender_ip:
                    filtered += 1
                    continue

                now = frame.received_monotonic or time.monotonic()
                sender_ip = frame.sender[0] if frame.sender else "unknown"
                previous_sequence = last_sequence_by_sender.get(sender_ip)
                sequence_delta = _sequence_delta(previous_sequence, frame.sequence)
                if sequence_delta is not None and sequence_delta > 1:
                    estimated_lost += sequence_delta - 1
                if sequence_delta is None or sequence_delta > 0:
                    last_sequence_by_sender[sender_ip] = frame.sequence

                if packet_logger is not None:
                    logged = packet_logger.write(frame, sequence_delta)
                    if not logged and not log_limit_reported:
                        print(
                            "Log size limit reached; stopped appending to "
                            f"{packet_logger.path}."
                        )
                        log_limit_reported = True

                receive_times.append(now)
                while receive_times and now - receive_times[0] > 2.0:
                    receive_times.popleft()
                first_frame = last_frame is None
                last_frame = frame

                should_report = (
                    first_frame
                    or args.once
                    or now - last_status_time >= args.status_interval
                )
                if should_report:
                    print(
                        _format_status(
                            frame,
                            _rate_hz(receive_times),
                            estimated_lost,
                            malformed,
                            sequence_delta,
                        )
                    )
                    if filtered:
                        print(f"Filtered {filtered} packet(s) from other senders.")
                    last_status_time = now
                if args.once:
                    return 0
    except KeyboardInterrupt:
        print("\nReceiver stopped.")
        return 0
    except OSError as exc:
        raise SystemExit(f"UDP receiver or JSONL log I/O failed: {exc}") from exc
    finally:
        if packet_logger is not None:
            packet_logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
