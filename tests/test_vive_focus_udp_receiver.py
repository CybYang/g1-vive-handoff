import json
import socket

import numpy as np
import pytest

from scripts.vive_focus_udp_receiver import (
    MAX_SEQUENCE,
    OPENXR_TO_MEDIAPIPE_21,
    FrameJSONLLogger,
    PacketValidationError,
    ViveFocusUDPReceiver,
    _sequence_delta,
    decode_packet,
)


def _hand(tracked=True):
    return {
        "tracked": tracked,
        "positions": [float(index) for index in range(26 * 3)],
        "valid": [1] * 26,
    }


def _packet():
    return {
        "v": 1,
        "seq": 42,
        "timestamp_unix_ms": 1_700_000_000_000,
        "space": "unity_world",
        "units": "m",
        "left": _hand(),
        "right": _hand(tracked=False),
    }


def _tracker(
    tracker_id=0,
    *,
    is_tracked=True,
    tracking_state=3,
    position=None,
    rotation=None,
):
    return {
        "id": tracker_id,
        "is_tracked": is_tracked,
        "tracking_state": tracking_state,
        "position": position if position is not None else [0.25, 1.10, -0.40],
        "rotation": rotation if rotation is not None else [0.0, 0.0, 0.0, 1.0],
    }


def _encoded_packet():
    return json.dumps(_packet(), separators=(",", ":")).encode("utf-8")


def test_decode_packet_validates_and_shapes_both_hands():
    frame = decode_packet(_encoded_packet(), sender=("192.168.1.5", 45678))

    assert frame.sequence == 42
    assert frame.sender == ("192.168.1.5", 45678)
    assert frame.left.positions.shape == (26, 3)
    assert frame.left.valid.shape == (26,)
    assert frame.left.tracked
    assert not frame.right.tracked
    assert np.isfinite(frame.left.positions).all()
    assert frame.trackers == ()


def test_decode_packet_accepts_valid_tracker_zero():
    packet = _packet()
    packet["seq"] = 100
    packet["trackers"] = [_tracker()]

    frame = decode_packet(json.dumps(packet).encode("utf-8"))

    assert len(frame.trackers) == 1
    tracker = frame.trackers[0]
    assert tracker.tracker_id == 0
    assert tracker.is_tracked
    assert tracker.tracking_state == 3
    assert tracker.pose_valid
    assert np.allclose(tracker.position, [0.25, 1.10, -0.40])
    assert np.allclose(tracker.rotation, [0.0, 0.0, 0.0, 1.0])
    assert frame.left.positions.shape == (26, 3)


def test_decode_packet_keeps_untracked_tracker_but_invalidates_pose():
    packet = _packet()
    packet["seq"] = 101
    packet["trackers"] = [
        _tracker(
            is_tracked=False,
            tracking_state=0,
            position=[0.0, 0.0, 0.0],
        )
    ]

    tracker = decode_packet(json.dumps(packet).encode("utf-8")).trackers[0]

    assert tracker.tracker_id == 0
    assert not tracker.is_tracked
    assert not tracker.pose_valid
    assert np.array_equal(tracker.position, np.zeros(3))


def test_decode_packet_accepts_multiple_trackers_by_id():
    packet = _packet()
    packet["trackers"] = [_tracker(0), _tracker(1, position=[-0.2, 1.0, -0.3])]

    frame = decode_packet(json.dumps(packet).encode("utf-8"))

    assert [tracker.tracker_id for tracker in frame.trackers] == [0, 1]
    assert all(tracker.pose_valid for tracker in frame.trackers)


def test_tracker_requires_both_position_and_rotation_tracking_bits():
    packet = _packet()
    packet["trackers"] = [_tracker(is_tracked=True, tracking_state=1)]

    tracker = decode_packet(json.dumps(packet).encode("utf-8")).trackers[0]

    assert tracker.is_tracked
    assert not tracker.pose_valid


def test_openxr_to_mediapipe_mapping_removes_extra_metacarpals():
    frame = decode_packet(_encoded_packet())

    positions, valid = frame.left.as_mediapipe21()
    expected = frame.left.positions[OPENXR_TO_MEDIAPIPE_21]

    assert positions.shape == (21, 3)
    assert valid.shape == (21,)
    assert np.array_equal(positions, expected)
    assert valid.all()

    _, untracked_valid = frame.right.as_mediapipe21()
    assert not untracked_valid.any()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda packet: packet.update(v=2), "unsupported protocol version"),
        (lambda packet: packet["left"].update(positions=[0.0] * 3), "78 numbers"),
        (lambda packet: packet["right"].update(valid=[1] * 25), "26 flags"),
        (lambda packet: packet.update(units="cm"), "units must be 'm'"),
    ],
)
def test_decode_packet_rejects_protocol_mismatches(mutate, message):
    packet = _packet()
    mutate(packet)

    with pytest.raises(PacketValidationError, match=message):
        decode_packet(json.dumps(packet).encode("utf-8"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda tracker: tracker.update(position=[0.0, 0.0]),
            "position must contain exactly 3 numbers",
        ),
        (
            lambda tracker: tracker.update(rotation=[0.0, 0.0, 1.0]),
            "rotation must contain exactly 4 numbers",
        ),
        (
            lambda tracker: tracker.update(position=[float("nan"), 0.0, 0.0]),
            "position contains NaN or infinity",
        ),
        (
            lambda tracker: tracker.update(rotation=[0.0, 0.0, 0.0, float("inf")]),
            "rotation contains NaN or infinity",
        ),
        (lambda tracker: tracker.update(id=5), "id must be <= 4"),
        (lambda tracker: tracker.pop("rotation"), "rotation must contain exactly"),
    ],
)
def test_decode_packet_rejects_invalid_tracker_fields(mutate, message):
    packet = _packet()
    tracker = _tracker()
    mutate(tracker)
    packet["trackers"] = [tracker]

    with pytest.raises(PacketValidationError, match=message):
        decode_packet(json.dumps(packet).encode("utf-8"))


def test_decode_packet_rejects_duplicate_tracker_ids():
    packet = _packet()
    packet["trackers"] = [_tracker(0), _tracker(0)]

    with pytest.raises(PacketValidationError, match="duplicate tracker ids"):
        decode_packet(json.dumps(packet).encode("utf-8"))


def test_sequence_delta_handles_signed_integer_wrap():
    assert _sequence_delta(MAX_SEQUENCE, 0) == 1
    assert _sequence_delta(100, 99) == -1


def test_udp_receiver_round_trip_on_loopback():
    with ViveFocusUDPReceiver("127.0.0.1", 0, timeout_s=1.0) as receiver:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(_encoded_packet(), receiver.address)
            frame = receiver.receive()
        finally:
            sender.close()

    assert frame.sequence == 42
    assert frame.sender is not None
    assert frame.sender[0] == "127.0.0.1"
    assert frame.received_monotonic is not None
    assert frame.payload_bytes == len(_encoded_packet())


def test_udp_receiver_can_coalesce_a_live_backlog_to_the_newest_frame():
    with ViveFocusUDPReceiver("127.0.0.1", 0, timeout_s=1.0) as receiver:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for sequence in (50, 51, 52):
                packet = _packet()
                packet["seq"] = sequence
                sender.sendto(json.dumps(packet).encode("utf-8"), receiver.address)
            frame, coalesced = receiver.receive_latest()
        finally:
            sender.close()

    assert frame.sequence == 52
    assert coalesced == 2


def test_udp_receiver_can_continue_after_malformed_tracker_packet():
    malformed_packet = _packet()
    malformed_packet["trackers"] = [_tracker(position=[0.0])]
    valid_packet = _packet()
    valid_packet["seq"] = 43
    valid_packet["trackers"] = [_tracker()]

    with ViveFocusUDPReceiver("127.0.0.1", 0, timeout_s=1.0) as receiver:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            malformed_payload = json.dumps(malformed_packet).encode("utf-8")
            sender.sendto(malformed_payload, receiver.address)
            sender.sendto(json.dumps(valid_packet).encode("utf-8"), receiver.address)
            with pytest.raises(PacketValidationError):
                receiver.receive()
            frame = receiver.receive()
        finally:
            sender.close()

    assert frame.sequence == 43
    assert frame.trackers[0].pose_valid


def test_jsonl_logger_records_sequence_and_joint_diagnostics(tmp_path):
    packet = _packet()
    packet["trackers"] = [_tracker()]
    payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
    frame = decode_packet(
        payload,
        sender=("192.168.1.5", 45678),
        received_monotonic=10.0,
    )
    path = tmp_path / "capture.jsonl"

    logger = FrameJSONLLogger(path, max_megabytes=1.0)
    try:
        assert logger.write(frame, sequence_delta=3)
    finally:
        logger.close()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["receiver"]["sequence_delta"] == 3
    assert record["receiver"]["payload_bytes"] == len(payload)
    assert record["receiver"]["left_valid"] == 26
    assert record["packet"]["seq"] == 42
    assert len(record["packet"]["left"]["positions"]) == 78
    assert record["receiver"]["tracker_count"] == 1
    assert record["receiver"]["tracker_pose_valid"] == {"0": True}
    assert record["packet"]["trackers"][0]["id"] == 0
    assert record["packet"]["trackers"][0]["rotation"] == [0.0, 0.0, 0.0, 1.0]
