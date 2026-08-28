import numpy as np
import pytest

from dex_retargeting.constants import HandType
from scripts.vive_focus_udp_receiver import (
    OPENXR_JOINT_COUNT,
    OPENXR_TO_MEDIAPIPE_21,
    HandJoints,
)
from scripts.vive_openxr_hand import (
    CanonicalHandLowPass,
    OpenXRHandFrameError,
    canonicalize_openxr_hand,
)


def _right_handed_open_hand() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = [0.0, 0.0, 0.0]
    points[1:5] = [
        [-0.018, 0.008, 0.0],
        [-0.033, 0.017, 0.002],
        [-0.046, 0.027, 0.003],
        [-0.057, 0.038, 0.004],
    ]
    for indices, x in zip(
        ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)),
        (-0.032, -0.011, 0.011, 0.033),
        strict=True,
    ):
        for segment, index in enumerate(indices, start=1):
            points[index] = [x, 0.025 * segment, 0.001 * segment]
    return points


def _left_handed_open_hand() -> np.ndarray:
    points = _right_handed_open_hand().copy()
    points[:, 0] *= -1.0
    return points


def _openxr_hand(
    right_handed_points: np.ndarray,
    *,
    tracked: bool = True,
    invalid_mediapipe_index: int | None = None,
) -> HandJoints:
    # The sender's Unity basis is left-handed.  This reflection is its own
    # inverse, so it converts our synthetic right-handed points into Unity.
    unity_points = np.asarray(right_handed_points) @ np.diag([1.0, 1.0, -1.0])
    positions = np.zeros((OPENXR_JOINT_COUNT, 3), dtype=np.float64)
    positions[OPENXR_TO_MEDIAPIPE_21] = unity_points
    valid = np.zeros(OPENXR_JOINT_COUNT, dtype=bool)
    valid[OPENXR_TO_MEDIAPIPE_21] = True
    if invalid_mediapipe_index is not None:
        valid[OPENXR_TO_MEDIAPIPE_21[invalid_mediapipe_index]] = False
    return HandJoints(tracked=tracked, positions=positions, valid=valid)


def test_canonical_hand_is_wrist_centered_and_finite():
    canonical = canonicalize_openxr_hand(
        _openxr_hand(_right_handed_open_hand()),
        hand_type=HandType.right,
    )

    assert canonical.shape == (21, 3)
    assert canonical.dtype == np.float32
    assert np.array_equal(canonical[0], np.zeros(3, dtype=np.float32))
    assert np.isfinite(canonical).all()


def test_canonical_left_hand_is_wrist_centered_and_finite():
    canonical = canonicalize_openxr_hand(
        _openxr_hand(_left_handed_open_hand()),
        hand_type=HandType.left,
    )

    assert canonical.shape == (21, 3)
    assert canonical.dtype == np.float32
    assert np.array_equal(canonical[0], np.zeros(3, dtype=np.float32))
    assert np.isfinite(canonical).all()


def test_canonical_hand_removes_world_translation_and_rotation():
    points = _right_handed_open_hand()
    angle = np.deg2rad(61.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = points @ rotation.T + np.array([0.8, -0.4, 1.2])

    baseline = canonicalize_openxr_hand(_openxr_hand(points))
    moved = canonicalize_openxr_hand(_openxr_hand(transformed))

    assert np.allclose(moved, baseline, atol=1e-6)


@pytest.mark.parametrize(
    "hand",
    [
        _openxr_hand(_right_handed_open_hand(), tracked=False),
        _openxr_hand(_right_handed_open_hand(), invalid_mediapipe_index=12),
    ],
)
def test_untracked_or_incomplete_hand_is_rejected(hand):
    with pytest.raises(OpenXRHandFrameError):
        canonicalize_openxr_hand(hand)


def test_canonical_low_pass_holds_state_and_can_reset():
    low_pass = CanonicalHandLowPass(alpha=0.25)
    first = np.zeros((21, 3), dtype=np.float32)
    second = np.ones((21, 3), dtype=np.float32)
    second[0] = 0.0

    assert np.array_equal(low_pass.update(first), first)
    filtered = low_pass.update(second)
    assert np.array_equal(filtered[0], np.zeros(3, dtype=np.float32))
    assert np.allclose(filtered[1:], 0.25)

    low_pass.reset()
    assert np.array_equal(low_pass.update(second), second)
