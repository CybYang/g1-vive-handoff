import numpy as np

from scripts.inspire_g2_pose_adapter import (
    G2_RIGHT_COLLISION_LINK_GROUPS,
    G2_RIGHT_YAW_COUPLING_TERMS,
    ROBOT_THUMB_YAW_LIMIT,
    InspireG2PoseAdapter,
    _calibrated_yaw,
    human_thumb_cmc_elevation,
    human_thumb_flexion_angles,
    human_thumb_opposition_ratio,
    human_yaw_angles,
    thumb_opposition_fraction,
)

JOINT_NAMES = (
    "left_index_yaw_joint",
    "left_index_mcp_joint",
    "left_index_pip_joint",
    "left_index_dip_joint",
    "left_middle_yaw_joint",
    "left_middle_mcp_joint",
    "left_middle_pip_joint",
    "left_middle_dip_joint",
    "left_pinky_mcp_joint",
    "left_pinky_pip_joint",
    "left_pinky_dip_joint",
    "left_ring_mcp_joint",
    "left_ring_pip_joint",
    "left_ring_dip_joint",
    "left_thumb_yaw_joint",
    "left_thumb_mcp_joint",
    "left_thumb_pip_joint",
    "left_thumb_dip_joint",
)


def _open_hand_points() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = [0.0, 0.0, 0.0]
    for indices, x in zip(
        ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)),
        (-0.03, -0.01, 0.01, 0.03),
        strict=True,
    ):
        for segment, index in enumerate(indices, start=1):
            points[index] = [x, 0.025 * segment, 0.0]
    # Thumb is not consumed by the four-finger estimator.
    points[1:5] = [
        [-0.02, 0.0, 0.0],
        [-0.035, 0.01, 0.0],
        [-0.05, 0.02, 0.0],
        [-0.06, 0.03, 0.0],
    ]
    return points


def _curled_hand_points() -> np.ndarray:
    points = _open_hand_points()
    for mcp_i, pip_i, dip_i, tip_i in (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ):
        points[pip_i] = points[mcp_i] + [0.0, 0.02, 0.02]
        points[dip_i] = points[pip_i] + [0.0, 0.0, 0.02]
        points[tip_i] = points[dip_i] + [0.0, -0.01, 0.02]
    return points


def _calibration() -> dict:
    return {
        "version": 1,
        "hand": "left",
        "flexion": {
            finger: {
                "valid": True,
                "mcp_open_deg": 0.0,
                "mcp_closed_deg": 45.0,
                "pip_open_deg": 0.0,
                "pip_closed_deg": 45.0,
            }
            for finger in ("index", "middle", "ring", "pinky")
        },
        "yaw": {
            finger: {
                "enabled": False,
                "neutral_deg": 0.0,
                "lower_deg": -10.0,
                "upper_deg": 10.0,
                "robot_limit_deg": 8.0,
                "sign": 1.0,
            }
            for finger in ("index", "middle")
        },
    }


def test_open_hand_forces_four_finger_flexion_but_preserves_raw_yaw():
    qpos = np.full(len(JOINT_NAMES), 0.5, dtype=np.float32)
    adapted = InspireG2PoseAdapter(alpha=1.0).adapt(
        _open_hand_points(),
        qpos,
        JOINT_NAMES,
    )

    for finger in ("index", "middle", "ring", "pinky"):
        for joint in ("mcp", "pip", "dip"):
            assert adapted[JOINT_NAMES.index(f"left_{finger}_{joint}_joint")] == 0
    assert adapted[JOINT_NAMES.index("left_index_yaw_joint")] == 0.5
    assert adapted[JOINT_NAMES.index("left_middle_yaw_joint")] == 0.5


def test_explicit_yaw_lock_holds_index_and_middle_at_neutral():
    qpos = np.full(len(JOINT_NAMES), 0.5, dtype=np.float32)
    adapted = InspireG2PoseAdapter(alpha=1.0, lock_yaw=True).adapt(
        _open_hand_points(),
        qpos,
        JOINT_NAMES,
    )

    assert adapted[JOINT_NAMES.index("left_index_yaw_joint")] == 0
    assert adapted[JOINT_NAMES.index("left_middle_yaw_joint")] == 0


def test_together_spread_yaw_uses_a_natural_one_sided_range():
    config = {
        "mode": "together_spread",
        "neutral_deg": 3.0,
        "spread_deg": -10.0,
        "robot_spread_deg": -8.0,
    }

    assert np.isclose(_calibrated_yaw(np.deg2rad(3.0), config), 0.0)
    assert np.isclose(
        _calibrated_yaw(np.deg2rad(-10.0), config),
        np.deg2rad(-8.0),
    )
    assert np.isclose(_calibrated_yaw(np.deg2rad(8.0), config), 0.0)


def test_thumb_dexpilot_values_are_preserved_and_mimic_is_consistent():
    qpos = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    qpos[JOINT_NAMES.index("left_thumb_yaw_joint")] = 0.7
    qpos[JOINT_NAMES.index("left_thumb_mcp_joint")] = 0.3
    qpos[JOINT_NAMES.index("left_thumb_dip_joint")] = 0.4
    adapted = InspireG2PoseAdapter(alpha=1.0).adapt(
        _open_hand_points(),
        qpos,
        JOINT_NAMES,
    )

    assert adapted[JOINT_NAMES.index("left_thumb_yaw_joint")] == np.float32(0.7)
    assert adapted[JOINT_NAMES.index("left_thumb_mcp_joint")] == np.float32(0.3)
    assert adapted[JOINT_NAMES.index("left_thumb_dip_joint")] == np.float32(0.4)
    assert np.isclose(
        adapted[JOINT_NAMES.index("left_thumb_pip_joint")],
        0.3 * 1.0853,
    )


def test_operator_calibration_maps_recorded_fist_to_robot_flexion_limits():
    qpos = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    adapted = InspireG2PoseAdapter(
        alpha=1.0,
        lock_yaw=True,
        calibration=_calibration(),
    ).adapt(
        _curled_hand_points(),
        qpos,
        JOINT_NAMES,
    )

    for finger in ("index", "middle", "ring", "pinky"):
        assert np.isclose(
            adapted[JOINT_NAMES.index(f"left_{finger}_mcp_joint")],
            1.3089969,
        )
        assert np.isclose(
            adapted[JOINT_NAMES.index(f"left_{finger}_pip_joint")],
            1.4835299,
        )


def test_v2_soft_references_do_not_reuse_v1_yaw_calibration():
    calibration = _calibration()
    calibration["yaw"]["middle"]["neutral_deg"] = np.rad2deg(
        human_yaw_angles(_curled_hand_points())["middle"]
    )
    adapter = InspireG2PoseAdapter(
        alpha=1.0,
        calibration=calibration,
    )
    values, weights = adapter.soft_joint_references(
        _curled_hand_points(),
        flexion_weight=1.5,
        yaw_weight=4.0,
        middle_yaw_weight=12.0,
    )

    for finger in ("index", "middle", "ring", "pinky"):
        assert values[f"left_{finger}_mcp_joint"] > 0
        assert values[f"left_{finger}_pip_joint"] > 0
        assert weights[f"left_{finger}_mcp_joint"] == 1.5
        assert weights[f"left_{finger}_pip_joint"] == 1.5
    assert values["left_index_yaw_joint"] == 0
    assert values["left_middle_yaw_joint"] == 0
    assert weights["left_index_yaw_joint"] == 4.0
    assert weights["left_middle_yaw_joint"] == 0.0


def test_v2_middle_yaw_tracks_motion_relative_to_calibrated_neutral():
    points = _open_hand_points()
    neutral = human_yaw_angles(points)["middle"]
    calibration = _calibration()
    calibration["yaw"]["middle"]["neutral_deg"] = np.rad2deg(neutral)
    adapter = InspireG2PoseAdapter(alpha=1.0, calibration=calibration)

    neutral_values, _ = adapter.soft_joint_references(points)
    moved_points = points.copy()
    moved_points[10] += [0.006, 0.0, 0.0]
    moved_angle = human_yaw_angles(moved_points)["middle"]
    moved_values, moved_weights = adapter.soft_joint_references(moved_points)

    expected = np.clip(moved_angle - neutral, -np.deg2rad(15), np.deg2rad(15))
    assert np.isclose(neutral_values["left_middle_yaw_joint"], 0.0)
    assert np.isclose(moved_values["left_middle_yaw_joint"], expected)
    assert moved_weights["left_middle_yaw_joint"] == 12.0


def test_v2_middle_yaw_reference_releases_during_a_fist():
    calibration = _calibration()
    calibration["yaw"]["middle"]["neutral_deg"] = np.rad2deg(
        human_yaw_angles(_open_hand_points())["middle"]
    )
    adapter = InspireG2PoseAdapter(alpha=1.0, calibration=calibration)

    values, weights = adapter.soft_joint_references(
        _curled_hand_points(),
        middle_yaw_weight=12.0,
    )

    assert values["left_middle_yaw_joint"] == 0.0
    assert weights["left_middle_yaw_joint"] == 0.0


def test_v2_thumb_flexion_does_not_follow_a_four_finger_fist():
    adapter = InspireG2PoseAdapter(alpha=1.0, calibration=_calibration())

    open_values, open_weights = adapter.soft_joint_references(_open_hand_points())
    fist_values, fist_weights = adapter.soft_joint_references(_curled_hand_points())

    for joint in ("left_thumb_mcp_joint", "left_thumb_dip_joint"):
        assert np.isclose(fist_values[joint], open_values[joint])
    assert open_weights["left_thumb_mcp_joint"] == 12.0
    assert fist_weights["left_thumb_mcp_joint"] == 12.0
    assert open_weights["left_thumb_dip_joint"] == 8.0
    assert fist_weights["left_thumb_dip_joint"] == 8.0


def test_thumb_opposition_is_normalized_by_palm_width():
    open_points = _open_hand_points()
    opposed_points = open_points.copy()
    mcp_centre = opposed_points[[5, 9, 13, 17]].mean(axis=0)
    open_points[4] = mcp_centre + [-0.12, 0.0, 0.0]
    opposed_points[4] = mcp_centre

    open_ratio = human_thumb_opposition_ratio(open_points)
    opposed_ratio = human_thumb_opposition_ratio(opposed_points)
    _, open_fraction = thumb_opposition_fraction(open_points)
    _, opposed_fraction = thumb_opposition_fraction(opposed_points)

    assert open_ratio > opposed_ratio
    assert open_fraction == 0.0
    assert opposed_fraction == 1.0


def test_v2_thumb_yaw_reference_is_off_by_default():
    adapter = InspireG2PoseAdapter(alpha=1.0, calibration=_calibration())

    values, weights = adapter.soft_joint_references(_open_hand_points())

    assert "left_thumb_yaw_joint" not in values
    assert "left_thumb_yaw_joint" not in weights


def test_v2_thumb_yaw_reference_uses_full_workspace_when_opposed():
    points = _open_hand_points()
    points[4] = points[[5, 9, 13, 17]].mean(axis=0)
    adapter = InspireG2PoseAdapter(alpha=1.0, calibration=_calibration())

    values, weights = adapter.soft_joint_references(
        points,
        thumb_yaw_weight=2.0,
    )

    assert values["left_thumb_yaw_joint"] == ROBOT_THUMB_YAW_LIMIT
    assert weights["left_thumb_yaw_joint"] == 2.0


def test_v2_thumb_cmc_tracks_metacarpal_elevation_not_local_mcp_bend():
    points = _open_hand_points()
    metacarpal = np.array([-0.015, 0.01, 0.02])
    points[2] = points[1] + metacarpal
    points[3] = points[2] + metacarpal
    points[4] = points[3] + metacarpal
    adapter = InspireG2PoseAdapter(alpha=1.0, calibration=_calibration())

    cmc_elevation = human_thumb_cmc_elevation(points)
    mcp_angle, ip_angle = human_thumb_flexion_angles(points)
    values, weights = adapter.soft_joint_references(points)

    assert cmc_elevation > np.deg2rad(20.0)
    assert np.isclose(mcp_angle, 0.0)
    assert np.isclose(ip_angle, 0.0)
    assert values["left_thumb_mcp_joint"] > 0
    assert np.isclose(values["left_thumb_dip_joint"], 0.0)
    assert weights["left_thumb_mcp_joint"] == 12.0
    assert weights["left_thumb_dip_joint"] == 8.0


def test_v2_thumb_ip_tracks_human_ip_independently_from_cmc():
    points = _open_hand_points()
    points[4] = points[3] + [0.0, 0.0, 0.02]
    adapter = InspireG2PoseAdapter(alpha=1.0, calibration=_calibration())

    cmc_elevation = human_thumb_cmc_elevation(points)
    _, ip_angle = human_thumb_flexion_angles(points)
    values, weights = adapter.soft_joint_references(points)

    assert np.isclose(cmc_elevation, 0.0)
    assert ip_angle > 0
    assert np.isclose(values["left_thumb_mcp_joint"], 0.0)
    assert values["left_thumb_dip_joint"] > 0
    assert weights["left_thumb_mcp_joint"] == 12.0
    assert weights["left_thumb_dip_joint"] == 8.0


def test_v2_legacy_thumb_weight_preserves_the_stable_disabled_behavior():
    points = _open_hand_points()
    points[3] = points[2] + [0.0, 0.015, 0.015]
    points[4] = points[3] + [0.0, 0.0, 0.02]
    adapter = InspireG2PoseAdapter(alpha=1.0, calibration=_calibration())

    values, weights = adapter.soft_joint_references(
        points,
        thumb_flexion_weight=0.0,
    )

    assert values["left_thumb_mcp_joint"] > 0
    assert values["left_thumb_dip_joint"] > 0
    assert weights["left_thumb_mcp_joint"] == 0.0
    assert weights["left_thumb_dip_joint"] == 0.0


def test_v2_ring_and_pinky_references_are_stronger_only_near_the_open_pose():
    adapter = InspireG2PoseAdapter(alpha=1.0, calibration=_calibration())

    _, open_weights = adapter.soft_joint_references(
        _open_hand_points(),
        flexion_weight=1.5,
        ring_open_weight_multiplier=8.0,
        pinky_open_weight_multiplier=8.0,
    )
    _, curled_weights = adapter.soft_joint_references(
        _curled_hand_points(),
        flexion_weight=1.5,
        ring_open_weight_multiplier=8.0,
        pinky_open_weight_multiplier=8.0,
    )

    assert open_weights["left_index_mcp_joint"] == 1.5
    assert open_weights["left_ring_mcp_joint"] == 12.0
    assert open_weights["left_ring_pip_joint"] == 12.0
    assert open_weights["left_pinky_mcp_joint"] == 12.0
    assert open_weights["left_pinky_pip_joint"] == 12.0
    assert curled_weights["left_ring_mcp_joint"] == 1.5
    assert curled_weights["left_ring_pip_joint"] == 1.5
    assert curled_weights["left_pinky_mcp_joint"] == 1.5
    assert curled_weights["left_pinky_pip_joint"] == 1.5


def test_v2_ring_open_boost_can_be_disabled_without_affecting_pinky():
    adapter = InspireG2PoseAdapter(alpha=1.0, calibration=_calibration())

    _, weights = adapter.soft_joint_references(
        _open_hand_points(),
        flexion_weight=1.5,
        ring_open_weight_multiplier=1.0,
        pinky_open_weight_multiplier=8.0,
    )

    assert weights["left_ring_mcp_joint"] == 1.5
    assert weights["left_ring_pip_joint"] == 1.5
    assert weights["left_pinky_mcp_joint"] == 12.0
    assert weights["left_pinky_pip_joint"] == 12.0


def test_right_adapter_emits_only_right_joint_references():
    adapter = InspireG2PoseAdapter(alpha=1.0, hand_side="right")

    values, weights = adapter.soft_joint_references(_open_hand_points())

    assert values
    assert values.keys() == weights.keys()
    assert all(name.startswith("right_") for name in values)
    assert "right_index_mcp_joint" in values
    assert "right_thumb_dip_joint" in values


def test_right_adapter_updates_right_mimic_joints():
    right_joint_names = tuple(name.replace("left_", "right_") for name in JOINT_NAMES)
    qpos = np.zeros(len(right_joint_names), dtype=np.float32)
    qpos[right_joint_names.index("right_thumb_mcp_joint")] = 0.3
    adapted = InspireG2PoseAdapter(alpha=1.0, hand_side="right").adapt(
        _open_hand_points(),
        qpos,
        right_joint_names,
    )

    assert np.isclose(
        adapted[right_joint_names.index("right_thumb_pip_joint")],
        0.3 * 1.0853,
    )


def test_right_v2_constants_match_the_right_urdf_namespace():
    collision_names = {
        name for group in G2_RIGHT_COLLISION_LINK_GROUPS for name in group
    }
    coupling_names = set(G2_RIGHT_YAW_COUPLING_TERMS[0]["joint_names"])

    assert collision_names
    assert all(name.startswith("right_") for name in collision_names)
    assert coupling_names == {
        "right_index_yaw_joint",
        "right_middle_yaw_joint",
    }
