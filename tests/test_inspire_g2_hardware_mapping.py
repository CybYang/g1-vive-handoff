from pathlib import Path

import numpy as np

from scripts.inspire_g2_hardware import (
    InspireG2LiveController,
    load_joint_mapping,
    qpos_to_angle_set,
    slew_limited_command,
)

PROJECT_ROOT = Path(__file__).parent.parent
MAPPING_PATH = PROJECT_ROOT / "scripts" / "inspire_g2_left_mapping.yaml"
RIGHT_MAPPING_PATH = PROJECT_ROOT / "scripts" / "inspire_g2_right_mapping.yaml"

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
RIGHT_JOINT_NAMES = tuple(name.replace("left_", "right_") for name in JOINT_NAMES)


def test_open_qpos_maps_to_vendor_open_pose():
    mapping = load_joint_mapping(MAPPING_PATH)
    actual = qpos_to_angle_set(np.zeros(18), JOINT_NAMES, mapping)
    assert actual == [
        1800,
        1800,
        1800,
        0,
        1800,
        0,
        1850,
        1850,
        1850,
        1850,
        1700,
        1550,
        2040,
    ]


def test_flexion_upper_limits_map_to_vendor_closed_limits():
    mapping = load_joint_mapping(MAPPING_PATH)
    qpos = np.zeros(18)
    upper_by_name = {
        "left_index_mcp_joint": 1.3089969,
        "left_index_pip_joint": 1.4835299,
        "left_middle_mcp_joint": 1.3089969,
        "left_middle_pip_joint": 1.4835299,
        "left_pinky_mcp_joint": 1.3089969,
        "left_pinky_pip_joint": 1.4835299,
        "left_ring_mcp_joint": 1.3089969,
        "left_ring_pip_joint": 1.4835299,
        "left_thumb_yaw_joint": 1.9198622,
        "left_thumb_mcp_joint": 0.6108652,
        "left_thumb_dip_joint": 0.9424778,
    }
    for name, upper in upper_by_name.items():
        qpos[JOINT_NAMES.index(name)] = upper

    actual = qpos_to_angle_set(qpos, JOINT_NAMES, mapping)
    assert actual == [
        965,
        965,
        965,
        0,
        965,
        0,
        1050,
        1050,
        1050,
        1050,
        600,
        1000,
        1450,
    ]


def test_right_mapping_uses_right_joints_and_same_nominal_endpoints():
    mapping = load_joint_mapping(RIGHT_MAPPING_PATH)
    assert all(item.urdf_joint.startswith("right_") for item in mapping)

    open_pose = qpos_to_angle_set(
        np.zeros(len(RIGHT_JOINT_NAMES)), RIGHT_JOINT_NAMES, mapping
    )
    assert open_pose == [
        1800,
        1800,
        1800,
        0,
        1800,
        0,
        1850,
        1850,
        1850,
        1850,
        1700,
        1550,
        2040,
    ]


def test_slew_limit_clamps_range_and_per_packet_change():
    limits = np.asarray([(0, 100)] * 13)
    command = slew_limited_command(
        current=[50] * 13,
        desired=[-100, 200] + [70] * 11,
        limits=limits,
        max_step_units=5,
    )
    assert command == [45, 55] + [55] * 11


def test_actual_position_feedback_is_a_read_only_register_query():
    controller = object.__new__(InspireG2LiveController)
    requested = []
    controller._quiet_read = lambda register: requested.append(register) or list(
        range(13)
    )

    assert controller.read_actual() == list(range(13))
    assert requested == ["angleAct"]
