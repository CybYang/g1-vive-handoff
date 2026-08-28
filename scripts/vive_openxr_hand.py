#!/usr/bin/env python3
"""Convert Unity/OpenXR hand joints into DexRetargeting's hand-local frame.

The Unity sender reports 26 OpenXR joints in Unity world coordinates.  Hand
retargeting must not depend on the headset world's translation or orientation,
so this module first selects the MediaPipe-compatible 21 joints, converts the
left-handed Unity basis to a right-handed basis, and then estimates the same
MANO-style wrist frame used by the original DexRetargeting camera example.

This module contains no MediaPipe, RealSense, SAPIEN, DDS, or hardware output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from dex_retargeting.constants import HandType, OPERATOR2MANO


MEDIAPIPE_JOINT_COUNT = 21
UNITY_LEFT_HANDED_TO_RIGHT_HANDED = np.diag([1.0, 1.0, -1.0])


class OpenXRHandFrameError(ValueError):
    """Raised when a tracked OpenXR hand cannot produce a safe 21-joint frame."""


class OpenXRHandLike(Protocol):
    tracked: bool

    def as_mediapipe21(self) -> tuple[np.ndarray, np.ndarray]: ...


def _hand_type(value: HandType | Literal["left", "right"] | str) -> HandType:
    if isinstance(value, HandType):
        return value
    try:
        return HandType[str(value).strip().lower()]
    except KeyError as exc:
        raise ValueError("hand_type must be 'left' or 'right'") from exc


def estimate_hand_frame(points: np.ndarray) -> np.ndarray:
    """Estimate the wrist frame used by DexRetargeting's camera example."""
    hand = np.asarray(points, dtype=np.float64)
    if hand.shape != (MEDIAPIPE_JOINT_COUNT, 3):
        raise OpenXRHandFrameError("hand points must have shape (21, 3)")
    if not np.isfinite(hand).all():
        raise OpenXRHandFrameError("hand points contain NaN or infinity")

    wrist, index_mcp, middle_mcp = hand[[0, 5, 9], :]
    x_vector = wrist - middle_mcp
    if float(np.linalg.norm(x_vector)) <= 1.0e-6:
        raise OpenXRHandFrameError("wrist and middle MCP are degenerate")

    plane_points = np.stack([wrist, index_mcp, middle_mcp], axis=0)
    plane_points -= np.mean(plane_points, axis=0, keepdims=True)
    _, singular_values, basis = np.linalg.svd(plane_points)
    if singular_values[1] <= 1.0e-6:
        raise OpenXRHandFrameError("wrist/index/middle points are collinear")

    normal = basis[2, :]
    x_axis = x_vector - np.dot(x_vector, normal) * normal
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1.0e-6:
        raise OpenXRHandFrameError("cannot estimate the hand longitudinal axis")
    x_axis /= x_norm
    z_axis = np.cross(x_axis, normal)

    # Match the sign convention used by SingleHandDetector: the lateral axis
    # points from the middle MCP toward the index MCP.
    if float(np.dot(z_axis, index_mcp - middle_mcp)) < 0.0:
        normal *= -1.0
        z_axis *= -1.0
    return np.stack([x_axis, normal, z_axis], axis=1)


def canonicalize_mediapipe21(
    positions: np.ndarray,
    valid: np.ndarray,
    *,
    hand_type: HandType | Literal["left", "right"] | str = HandType.right,
) -> np.ndarray:
    """Return finite wrist-centred MANO-style joints for DexRetargeting."""
    points = np.asarray(positions, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    if points.shape != (MEDIAPIPE_JOINT_COUNT, 3):
        raise OpenXRHandFrameError("positions must have shape (21, 3)")
    if validity.shape != (MEDIAPIPE_JOINT_COUNT,):
        raise OpenXRHandFrameError("valid must have shape (21,)")
    if not np.isfinite(points).all():
        raise OpenXRHandFrameError("positions contain NaN or infinity")
    if not validity.all():
        missing = np.flatnonzero(~validity).tolist()
        raise OpenXRHandFrameError(f"required OpenXR joints are invalid: {missing}")

    # Unity world coordinates are left-handed.  Negating Z performs the one
    # reflection required before estimating a right-handed hand-local frame.
    right_handed = points @ UNITY_LEFT_HANDED_TO_RIGHT_HANDED.T
    centred = right_handed - right_handed[0:1, :]
    frame = estimate_hand_frame(centred)
    canonical = centred @ frame @ OPERATOR2MANO[_hand_type(hand_type)]
    if not np.isfinite(canonical).all():
        raise OpenXRHandFrameError("canonical hand frame contains invalid values")
    canonical[0] = 0.0
    return canonical.astype(np.float32)


def canonicalize_openxr_hand(
    hand: OpenXRHandLike,
    *,
    hand_type: HandType | Literal["left", "right"] | str = HandType.right,
) -> np.ndarray:
    """Convert one validated receiver hand into DexRetargeting input."""
    if not hand.tracked:
        raise OpenXRHandFrameError("OpenXR hand is not tracked")
    positions, valid = hand.as_mediapipe21()
    return canonicalize_mediapipe21(
        positions,
        valid,
        hand_type=hand_type,
    )


@dataclass
class CanonicalHandLowPass:
    """Simple low-pass filter operating after wrist-frame canonicalization."""

    alpha: float = 0.5
    _state: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

    def update(self, points: np.ndarray) -> np.ndarray:
        current = np.asarray(points, dtype=np.float32)
        if current.shape != (MEDIAPIPE_JOINT_COUNT, 3):
            raise ValueError("points must have shape (21, 3)")
        if self._state is None:
            self._state = current.copy()
        else:
            self._state += self.alpha * (current - self._state)
        self._state[0] = 0.0
        return self._state.copy()

    def reset(self) -> None:
        self._state = None
