#!/usr/bin/env python3
"""Robot-specific pose adaptation for the 13-DoF Inspire G2 hand.

Vanilla DexPilot only constrains task-space vectors.  For this hand that leaves
multiple joint solutions for an open human pose, including curled pinky/ring
solutions and middle-finger yaw that intersects its neighbour.  This adapter
uses observable human bone angles for the four finger flexion joints. The two
yaw joints can preserve DexPilot's raw solution, be explicitly locked, or use
an operator calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

FINGER_LANDMARKS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

ROBOT_FLEX_LIMITS = {
    "index": (1.3089969, 1.4835299),
    "middle": (1.3089969, 1.4835299),
    "ring": (1.3089969, 1.4835299),
    "pinky": (1.3089969, 1.4835299),
}

ROBOT_THUMB_FLEX_LIMITS = {
    # Physical semantics; the vendor URDF names these joints thumb_mcp/dip.
    "cmc": 0.6108652,
    "ip": 0.9424778,
}
ROBOT_THUMB_YAW_LIMIT = 1.9198622

CALIBRATION_MIN_SPAN_DEG = {
    "mcp": 10.0,
    "pip": 20.0,
}

ROBOT_YAW_LIMIT = np.deg2rad(15.0)
MIDDLE_YAW_FULL_WEIGHT_MCP_FRACTION = 0.15
MIDDLE_YAW_RELEASE_MCP_FRACTION = 0.45

def _normalize_hand_side(hand_side: str) -> str:
    side = str(hand_side).strip().lower()
    if side not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right'")
    return side


def g2_mimic_joints(hand_side: str) -> dict[str, tuple[str, float]]:
    side = _normalize_hand_side(hand_side)
    return {
        f"{side}_index_dip_joint": (f"{side}_index_pip_joint", 1.1232),
        f"{side}_middle_dip_joint": (f"{side}_middle_pip_joint", 1.1232),
        f"{side}_ring_dip_joint": (f"{side}_ring_pip_joint", 1.1232),
        f"{side}_pinky_dip_joint": (f"{side}_pinky_pip_joint", 1.1232),
        f"{side}_thumb_pip_joint": (f"{side}_thumb_mcp_joint", 1.0853),
    }


def g2_collision_link_groups(hand_side: str) -> list[list[str]]:
    side = _normalize_hand_side(hand_side)
    return [
        [
            f"{side}_{finger}_mcp",
            f"{side}_{finger}_pip",
            f"{side}_{finger}_dip",
            f"{side}_{finger}_force_sensor",
        ]
        for finger in ("index", "middle", "ring", "pinky")
    ]


def g2_yaw_coupling_terms(hand_side: str) -> list[dict[str, Any]]:
    side = _normalize_hand_side(hand_side)
    return [
        {
            "joint_names": [
                f"{side}_index_yaw_joint",
                f"{side}_middle_yaw_joint",
            ],
            # Both mirrored G2 URDFs use opposite signed joint motion for
            # index/middle spreading, so their sum is the neutral coupling.
            "coefficients": [1.0, 1.0],
            "target": 0.0,
        }
    ]


# Keep the original public left-side constants for existing scripts/tests.
MIMIC_JOINTS = g2_mimic_joints("left")
G2_LEFT_COLLISION_LINK_GROUPS = g2_collision_link_groups("left")
G2_RIGHT_COLLISION_LINK_GROUPS = g2_collision_link_groups("right")
G2_LEFT_YAW_COUPLING_TERMS = g2_yaw_coupling_terms("left")
G2_RIGHT_YAW_COUPLING_TERMS = g2_yaw_coupling_terms("right")


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("degenerate hand landmark vector")
    return vector / norm


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(_unit(first), _unit(second)), -1.0, 1.0))
    return float(np.arccos(cosine))


def _scaled_flexion(
    human_angle: float,
    robot_max: float,
    deadband_deg: float,
    human_max_deg: float,
) -> float:
    deadband = np.deg2rad(deadband_deg)
    human_max = np.deg2rad(human_max_deg)
    fraction = np.clip(
        (human_angle - deadband) / (human_max - deadband),
        0.0,
        1.0,
    )
    return float(fraction * robot_max)


def _open_pose_weight_multiplier(
    flexion_fraction: float,
    weight_multiplier: float,
    open_fraction: float,
    release_fraction: float,
) -> float:
    """Fade an open-pose weight boost back to one as a finger bends."""
    if flexion_fraction <= open_fraction:
        return weight_multiplier
    if flexion_fraction >= release_fraction:
        return 1.0
    transition = (flexion_fraction - open_fraction) / (
        release_fraction - open_fraction
    )
    return float(weight_multiplier + transition * (1.0 - weight_multiplier))


def _validated_points(joint_pos: np.ndarray) -> np.ndarray:
    points = np.asarray(joint_pos, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("joint_pos must be a finite (21, 3) array")
    return points


def _palm_normal(points: np.ndarray) -> np.ndarray:
    wrist = points[0]
    return _unit(np.cross(points[5] - wrist, points[17] - wrist))


def human_four_finger_angles(
    joint_pos: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """Return raw human MCP/PIP flexion angles in radians."""
    points = _validated_points(joint_pos)
    palm_normal = _palm_normal(points)

    result = {}
    for finger, (mcp_i, pip_i, dip_i, _) in FINGER_LANDMARKS.items():
        proximal = points[pip_i] - points[mcp_i]
        middle = points[dip_i] - points[pip_i]

        # Flexion is the proximal phalanx's out-of-palm-plane component.
        mcp_angle = float(
            np.arcsin(
                np.clip(abs(np.dot(_unit(proximal), palm_normal)), 0.0, 1.0)
            )
        )
        pip_angle = _angle_between(proximal, middle)
        result[finger] = (mcp_angle, pip_angle)
    return result


def human_thumb_flexion_angles(joint_pos: np.ndarray) -> tuple[float, float]:
    """Return human thumb MCP/IP bending angles in radians."""
    points = _validated_points(joint_pos)
    metacarpal = points[2] - points[1]
    proximal = points[3] - points[2]
    distal = points[4] - points[3]
    return (
        _angle_between(metacarpal, proximal),
        _angle_between(proximal, distal),
    )


def human_thumb_cmc_elevation(joint_pos: np.ndarray) -> float:
    """Return thumb-metacarpal elevation out of the human palm plane.

    OpenXR positions do not expose a directly comparable scalar for the G2's
    second active CMC axis.  The metacarpal elevation is observable from joint
    positions and, unlike the local human MCP angle, describes motion at the
    base of the thumb.
    """
    points = _validated_points(joint_pos)
    metacarpal = points[2] - points[1]
    normal_component = abs(float(np.dot(_unit(metacarpal), _palm_normal(points))))
    return float(np.arcsin(np.clip(normal_component, 0.0, 1.0)))


def human_thumb_opposition_ratio(joint_pos: np.ndarray) -> float:
    """Return thumb-tip distance to the four-finger MCP centre / palm width.

    This dimensionless observation measures how far the thumb has travelled
    into the palm's usable grasp region.  A large value is an open/abducted
    thumb; a small value is an opposed thumb near the four-finger MCP centre.
    It is invariant to hand translation, rotation, and uniform hand size.
    """
    points = _validated_points(joint_pos)
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    if palm_width < 1e-8:
        raise ValueError("degenerate palm width")
    four_finger_mcp_centre = points[[5, 9, 13, 17]].mean(axis=0)
    return float(
        np.linalg.norm(points[4] - four_finger_mcp_centre) / palm_width
    )


def thumb_opposition_fraction(
    joint_pos: np.ndarray,
    *,
    open_ratio: float = 1.55,
    opposed_ratio: float = 0.85,
    exponent: float = 0.7,
) -> tuple[float, float]:
    """Map the human opposition observation onto the robot workspace [0, 1]."""
    if not 0.0 <= opposed_ratio < open_ratio:
        raise ValueError("thumb ratios must satisfy 0 <= opposed < open")
    if exponent <= 0:
        raise ValueError("thumb opposition exponent must be positive")
    ratio = human_thumb_opposition_ratio(joint_pos)
    linear = np.clip(
        (open_ratio - ratio) / (open_ratio - opposed_ratio),
        0.0,
        1.0,
    )
    return ratio, float(linear**exponent)


def human_yaw_angles(joint_pos: np.ndarray) -> dict[str, float]:
    """Estimate signed index/middle abduction in the human palm plane."""
    points = _validated_points(joint_pos)
    wrist = points[0]
    normal = _palm_normal(points)
    result = {}
    for finger in ("index", "middle"):
        mcp_i, pip_i, _, _ = FINGER_LANDMARKS[finger]
        reference = points[mcp_i] - wrist
        proximal = points[pip_i] - points[mcp_i]
        reference = reference - np.dot(reference, normal) * normal
        proximal = proximal - np.dot(proximal, normal) * normal
        reference = _unit(reference)
        proximal = _unit(proximal)
        result[finger] = float(
            np.arctan2(
                np.dot(np.cross(reference, proximal), normal),
                np.dot(reference, proximal),
            )
        )
    return result


def human_four_finger_flexion(
    joint_pos: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """Map raw angles with conservative defaults when no calibration exists."""
    angles = human_four_finger_angles(joint_pos)
    result = {}
    for finger, (mcp_angle, pip_angle) in angles.items():
        mcp_max, pip_max = ROBOT_FLEX_LIMITS[finger]
        result[finger] = (
            _scaled_flexion(mcp_angle, mcp_max, 8.0, 80.0),
            _scaled_flexion(pip_angle, pip_max, 10.0, 100.0),
        )
    return result


def load_calibration(
    path: Path | str,
    expected_hand: str = "left",
) -> dict[str, Any]:
    """Load and minimally validate one operator hand calibration."""
    side = _normalize_hand_side(expected_hand)
    calibration_path = Path(path)
    data = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("hand") != side:
        raise ValueError(f"expected a version-1 {side}-hand calibration")
    for finger in FINGER_LANDMARKS:
        if finger not in data.get("flexion", {}):
            raise ValueError(f"calibration is missing flexion.{finger}")
    for finger in ("index", "middle"):
        if finger not in data.get("yaw", {}):
            raise ValueError(f"calibration is missing yaw.{finger}")
    return data


def _calibrated_flexion(
    angle: float,
    open_angle: float,
    closed_angle: float,
    robot_max: float,
) -> float:
    span = closed_angle - open_angle
    if span <= np.deg2rad(5.0):
        raise ValueError("calibrated flexion span is too small")
    fraction = np.clip((angle - open_angle) / span, 0.0, 1.0)
    return float(fraction * robot_max)


def _flexion_axis_is_calibrated(config: dict[str, Any], axis: str) -> bool:
    """Support per-axis V2 validity and infer it for older calibration files."""
    explicit = config.get(f"{axis}_valid")
    if explicit is not None:
        return bool(explicit)
    try:
        span = float(config[f"{axis}_closed_deg"]) - float(
            config[f"{axis}_open_deg"]
        )
    except (KeyError, TypeError, ValueError):
        return False
    return span >= CALIBRATION_MIN_SPAN_DEG[axis]


def _calibrated_yaw(angle: float, config: dict[str, Any]) -> float:
    neutral = np.deg2rad(float(config["neutral_deg"]))
    if config.get("mode") == "together_spread":
        spread = np.deg2rad(float(config["spread_deg"]))
        span = spread - neutral
        if abs(span) < np.deg2rad(2.0):
            return 0.0
        fraction = float(np.clip((angle - neutral) / span, 0.0, 1.0))
        robot_spread = np.deg2rad(float(config["robot_spread_deg"]))
        return fraction * robot_spread

    lower = np.deg2rad(float(config["lower_deg"]))
    upper = np.deg2rad(float(config["upper_deg"]))
    limit = np.deg2rad(float(config.get("robot_limit_deg", 8.0)))
    sign = float(config.get("sign", 1.0))
    delta = angle - neutral
    if delta >= 0:
        span = max(upper - neutral, np.deg2rad(2.0))
    else:
        span = max(neutral - lower, np.deg2rad(2.0))
    return float(np.clip(delta / span, -1.0, 1.0) * limit * sign)


class InspireG2PoseAdapter:
    """Apply stable G2-specific four-finger kinematics to a DexPilot pose."""

    def __init__(
        self,
        alpha: float = 0.35,
        lock_yaw: bool = False,
        calibration: dict[str, Any] | None = None,
        hand_side: str = "left",
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.lock_yaw = lock_yaw
        self.calibration = calibration
        self.hand_side = _normalize_hand_side(hand_side)
        if calibration is not None and calibration.get("hand") != self.hand_side:
            raise ValueError(
                f"calibration is for {calibration.get('hand')!r}, "
                f"not {self.hand_side!r}"
            )
        self.joint_prefix = f"{self.hand_side}_"
        self.mimic_joints = g2_mimic_joints(self.hand_side)
        self._last_controlled: dict[str, float] | None = None

    def soft_joint_references(
        self,
        joint_pos: np.ndarray,
        flexion_weight: float = 1.0,
        yaw_weight: float = 4.0,
        middle_yaw_weight: float = 12.0,
        thumb_flexion_weight: float | None = None,
        thumb_yaw_weight: float = 0.0,
        thumb_cmc_weight: float = 12.0,
        thumb_ip_weight: float = 8.0,
        thumb_opposition_open_ratio: float = 1.55,
        thumb_opposition_opposed_ratio: float = 0.85,
        thumb_opposition_exponent: float = 0.7,
        thumb_cmc_open_elevation_deg: float = 20.0,
        thumb_cmc_opposed_elevation_deg: float = 55.0,
        ring_open_weight_multiplier: float = 8.0,
        ring_open_fraction: float = 0.10,
        ring_release_fraction: float = 0.35,
        pinky_open_weight_multiplier: float = 8.0,
        pinky_open_fraction: float = 0.10,
        pinky_release_fraction: float = 0.35,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Build V2 soft targets without overwriting the DexPilot solution.

        Four-finger flexion uses the same observable human angles as V1, but
        the optimizer is free to trade this target against fingertip error.
        Index/middle yaw use a neutral reference only; this is a soft prior,
        not the old calibrated hard mapping, so task-space evidence can still
        produce abduction while discouraging yaw as a morphology-compensation
        shortcut.
        """
        if (
            flexion_weight < 0
            or yaw_weight < 0
            or middle_yaw_weight < 0
            or thumb_yaw_weight < 0
            or thumb_cmc_weight < 0
            or thumb_ip_weight < 0
            or (thumb_flexion_weight is not None and thumb_flexion_weight < 0)
        ):
            raise ValueError("soft joint reference weights must be non-negative")
        if not (
            0.0
            <= thumb_opposition_opposed_ratio
            < thumb_opposition_open_ratio
        ):
            raise ValueError(
                "thumb opposition ratios must satisfy 0 <= opposed < open"
            )
        if thumb_opposition_exponent <= 0:
            raise ValueError("thumb opposition exponent must be positive")
        if not (
            0.0
            <= thumb_cmc_open_elevation_deg
            < thumb_cmc_opposed_elevation_deg
            <= 90.0
        ):
            raise ValueError(
                "thumb CMC elevations must satisfy "
                "0 <= open < opposed <= 90 degrees"
            )
        open_pose_parameters = {
            "ring": (
                ring_open_weight_multiplier,
                ring_open_fraction,
                ring_release_fraction,
            ),
            "pinky": (
                pinky_open_weight_multiplier,
                pinky_open_fraction,
                pinky_release_fraction,
            ),
        }
        for finger, (multiplier, open_fraction, release_fraction) in (
            open_pose_parameters.items()
        ):
            if multiplier < 1:
                raise ValueError(
                    f"{finger} open weight multiplier must be at least 1"
                )
            if not 0 <= open_fraction < release_fraction <= 1:
                raise ValueError(
                    f"{finger} flexion fractions must satisfy "
                    "0 <= open < release <= 1"
                )

        raw_flexion = human_four_finger_angles(joint_pos)
        values: dict[str, float] = {}
        weights: dict[str, float] = {}
        robot_flexion: dict[str, tuple[float, float]] = {}
        for finger, (mcp_angle, pip_angle) in raw_flexion.items():
            mcp_max, pip_max = ROBOT_FLEX_LIMITS[finger]
            config = (
                self.calibration.get("flexion", {}).get(finger, {})
                if self.calibration is not None
                else {}
            )
            if _flexion_axis_is_calibrated(config, "mcp"):
                mcp = _calibrated_flexion(
                    mcp_angle,
                    np.deg2rad(float(config["mcp_open_deg"])),
                    np.deg2rad(float(config["mcp_closed_deg"])),
                    mcp_max,
                )
            else:
                mcp = _scaled_flexion(mcp_angle, mcp_max, 8.0, 80.0)
            if _flexion_axis_is_calibrated(config, "pip"):
                pip = _calibrated_flexion(
                    pip_angle,
                    np.deg2rad(float(config["pip_open_deg"])),
                    np.deg2rad(float(config["pip_closed_deg"])),
                    pip_max,
                )
            else:
                pip = _scaled_flexion(pip_angle, pip_max, 10.0, 100.0)
            robot_flexion[finger] = (mcp, pip)

            finger_weight = flexion_weight
            if finger in open_pose_parameters:
                multiplier, open_fraction, release_fraction = (
                    open_pose_parameters[finger]
                )
                flexion_fraction = max(mcp / mcp_max, pip / pip_max)
                finger_weight *= _open_pose_weight_multiplier(
                    flexion_fraction,
                    multiplier,
                    open_fraction,
                    release_fraction,
                )

            for joint_suffix, value in (("mcp", mcp), ("pip", pip)):
                name = f"{self.joint_prefix}{finger}_{joint_suffix}_joint"
                values[name] = value
                weights[name] = finger_weight

        index_yaw_name = f"{self.joint_prefix}index_yaw_joint"
        values[index_yaw_name] = 0.0
        weights[index_yaw_name] = yaw_weight

        # The full middle-yaw range calibration can be invalid when the
        # operator cannot isolate middle-finger abduction, but its recorded
        # together-pose angle is still a useful neutral anchor.  Track the
        # signed deviation from that anchor so a stronger middle-yaw weight
        # removes the resting bias without locking intentional side motion.
        middle_target = 0.0
        middle_mcp, _ = robot_flexion["middle"]
        middle_mcp_fraction = middle_mcp / ROBOT_FLEX_LIMITS["middle"][0]
        if middle_mcp_fraction <= MIDDLE_YAW_FULL_WEIGHT_MCP_FRACTION:
            middle_yaw_gate = 1.0
        elif middle_mcp_fraction >= MIDDLE_YAW_RELEASE_MCP_FRACTION:
            middle_yaw_gate = 0.0
        else:
            middle_yaw_gate = 1.0 - (
                middle_mcp_fraction - MIDDLE_YAW_FULL_WEIGHT_MCP_FRACTION
            ) / (
                MIDDLE_YAW_RELEASE_MCP_FRACTION
                - MIDDLE_YAW_FULL_WEIGHT_MCP_FRACTION
            )
        middle_config = (
            self.calibration.get("yaw", {}).get("middle", {})
            if self.calibration is not None
            else {}
        )
        if middle_yaw_gate > 0 and "neutral_deg" in middle_config:
            middle_angle = human_yaw_angles(joint_pos)["middle"]
            neutral = np.deg2rad(float(middle_config["neutral_deg"]))
            delta = np.arctan2(
                np.sin(middle_angle - neutral),
                np.cos(middle_angle - neutral),
            )
            middle_target = float(np.clip(delta, -ROBOT_YAW_LIMIT, ROBOT_YAW_LIMIT))
        middle_yaw_name = f"{self.joint_prefix}middle_yaw_joint"
        values[middle_yaw_name] = middle_target
        weights[middle_yaw_name] = middle_yaw_weight * middle_yaw_gate

        thumb_human_mcp_angle, thumb_ip_angle = human_thumb_flexion_angles(
            joint_pos
        )
        thumb_cmc_elevation = human_thumb_cmc_elevation(joint_pos)
        thumb_yaw_name = f"{self.joint_prefix}thumb_yaw_joint"
        thumb_mcp_name = f"{self.joint_prefix}thumb_mcp_joint"
        thumb_dip_name = f"{self.joint_prefix}thumb_dip_joint"
        if thumb_yaw_weight > 0:
            _, opposition_fraction = thumb_opposition_fraction(
                joint_pos,
                open_ratio=thumb_opposition_open_ratio,
                opposed_ratio=thumb_opposition_opposed_ratio,
                exponent=thumb_opposition_exponent,
            )
            values[thumb_yaw_name] = (
                opposition_fraction * ROBOT_THUMB_YAW_LIMIT
            )
            weights[thumb_yaw_name] = thumb_yaw_weight
        if thumb_flexion_weight is None:
            # In the vendor model ``thumb_mcp_joint`` is physically the second
            # active CMC DoF.  Drive it from palm-relative metacarpal elevation,
            # not from the human anatomical MCP bend.
            values[thumb_mcp_name] = _scaled_flexion(
                thumb_cmc_elevation,
                ROBOT_THUMB_FLEX_LIMITS["cmc"],
                thumb_cmc_open_elevation_deg,
                thumb_cmc_opposed_elevation_deg,
            )
            cmc_weight = thumb_cmc_weight
            ip_weight = thumb_ip_weight
        else:
            # Backward-compatible path used by the dated stable launcher and
            # old experiments.  A value of zero exactly disables both legacy
            # thumb references.
            values[thumb_mcp_name] = _scaled_flexion(
                thumb_human_mcp_angle,
                ROBOT_THUMB_FLEX_LIMITS["cmc"],
                5.0,
                60.0,
            )
            cmc_weight = thumb_flexion_weight
            ip_weight = thumb_flexion_weight
        values[thumb_dip_name] = _scaled_flexion(
            thumb_ip_angle,
            ROBOT_THUMB_FLEX_LIMITS["ip"],
            5.0,
            70.0,
        )
        weights[thumb_mcp_name] = cmc_weight
        weights[thumb_dip_name] = ip_weight
        return values, weights

    def adapt(
        self,
        joint_pos: np.ndarray,
        qpos: Sequence[float],
        joint_names: Sequence[str],
    ) -> np.ndarray:
        result = np.asarray(qpos, dtype=np.float64).copy()
        if result.shape != (len(joint_names),):
            raise ValueError("qpos and joint_names length mismatch")
        name_to_index = {name: index for index, name in enumerate(joint_names)}
        raw_flexion = human_four_finger_angles(joint_pos)
        flexion = {}
        for finger, (mcp_angle, pip_angle) in raw_flexion.items():
            mcp_max, pip_max = ROBOT_FLEX_LIMITS[finger]
            config = (
                self.calibration.get("flexion", {}).get(finger, {})
                if self.calibration is not None
                else {}
            )
            mcp = (
                _calibrated_flexion(
                    mcp_angle,
                    np.deg2rad(float(config["mcp_open_deg"])),
                    np.deg2rad(float(config["mcp_closed_deg"])),
                    mcp_max,
                )
                if _flexion_axis_is_calibrated(config, "mcp")
                else _scaled_flexion(mcp_angle, mcp_max, 8.0, 80.0)
            )
            pip = (
                _calibrated_flexion(
                    pip_angle,
                    np.deg2rad(float(config["pip_open_deg"])),
                    np.deg2rad(float(config["pip_closed_deg"])),
                    pip_max,
                )
                if _flexion_axis_is_calibrated(config, "pip")
                else _scaled_flexion(pip_angle, pip_max, 10.0, 100.0)
            )
            flexion[finger] = (mcp, pip)

        controlled: dict[str, float] = {}
        for finger, (mcp, pip) in flexion.items():
            controlled[f"{self.joint_prefix}{finger}_mcp_joint"] = mcp
            controlled[f"{self.joint_prefix}{finger}_pip_joint"] = pip

        # The G2 has only index/middle yaw.  Vanilla fingertip-only IK can use
        # these joints to reduce a morphology mismatch and make fingers cross.
        # Neutral is the stable baseline until yaw anchors are measured.
        if self.lock_yaw:
            controlled[f"{self.joint_prefix}index_yaw_joint"] = 0.0
            controlled[f"{self.joint_prefix}middle_yaw_joint"] = 0.0
        elif self.calibration is not None:
            yaw_angles = human_yaw_angles(joint_pos)
            for finger in ("index", "middle"):
                config = self.calibration["yaw"][finger]
                controlled[f"{self.joint_prefix}{finger}_yaw_joint"] = (
                    _calibrated_yaw(yaw_angles[finger], config)
                    if config.get("enabled", False)
                    else 0.0
                )

        if self._last_controlled is not None:
            controlled = {
                name: self._last_controlled[name]
                + self.alpha * (value - self._last_controlled[name])
                for name, value in controlled.items()
            }
        self._last_controlled = controlled.copy()

        for name, value in controlled.items():
            result[name_to_index[name]] = value

        # Keep the rendered mimic joints consistent after overwriting their
        # driving joints.  These five joints are not hardware commands.
        for mimic_name, (source_name, multiplier) in self.mimic_joints.items():
            result[name_to_index[mimic_name]] = (
                result[name_to_index[source_name]] * multiplier
            )
        return result.astype(np.float32)
