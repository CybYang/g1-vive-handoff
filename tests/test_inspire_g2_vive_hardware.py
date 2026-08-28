from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from dex_retargeting.retargeting_config import RetargetingConfig
from scripts.inspire_g2_hardware import load_joint_mapping
from scripts.inspire_g2_vive_hardware import (
    HARDWARE_CONFIRMATION,
    _parse_args,
    _resolved_ports,
    _validate_mapping,
)
from scripts.inspire_g2_vive_runtime import build_hand_runtime, solve_hand_runtime
from scripts.vive_focus_udp_receiver import (
    OPENXR_JOINT_COUNT,
    OPENXR_TO_MEDIAPIPE_21,
    HandJoints,
)

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        ("left", {"left": "/dev/ttyUSB0"}),
        ("right", {"right": "/dev/ttyUSB0"}),
        (
            "both",
            {"left": "/dev/ttyUSB0", "right": "/dev/ttyUSB1"},
        ),
    ],
)
def test_resolved_ports_have_safe_side_aware_defaults(side, expected):
    assert _resolved_ports(side) == expected


def test_bimanual_ports_must_be_distinct():
    with pytest.raises(ValueError, match="distinct serial ports"):
        _resolved_ports("both", "/dev/g2", "/dev/g2")


def test_hardware_confirmation_is_mandatory():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_confirmation_parses_without_opening_serial():
    args = _parse_args(["--confirm-hardware", HARDWARE_CONFIRMATION])
    assert args.objective == "v2"
    assert args.hand_side == "right"
    assert args.v2_thumb_yaw_reference_weight == 0.0


def test_experimental_thumb_yaw_options_parse_without_opening_serial():
    args = _parse_args(
        [
            "--confirm-hardware",
            HARDWARE_CONFIRMATION,
            "--v2-thumb-yaw-reference-weight",
            "2.0",
            "--v2-thumb-opposition-open-ratio",
            "1.55",
            "--v2-thumb-opposition-opposed-ratio",
            "0.85",
            "--v2-thumb-opposition-exponent",
            "0.7",
        ]
    )

    assert args.v2_thumb_yaw_reference_weight == 2.0
    assert args.v2_thumb_opposition_open_ratio == 1.55
    assert args.v2_thumb_opposition_opposed_ratio == 0.85
    assert args.v2_thumb_opposition_exponent == 0.7


@pytest.mark.parametrize("side", ["left", "right"])
def test_each_mapping_contains_13_names_for_its_side(side):
    mapping = load_joint_mapping(
        PROJECT_ROOT / "scripts" / f"inspire_g2_{side}_mapping.yaml"
    )
    names = [item.urdf_joint for item in mapping]
    _validate_mapping(side, mapping, names)
    assert len(mapping) == 13


def _synthetic_openxr_hand(side: str) -> HandJoints:
    points = np.zeros((21, 3), dtype=np.float64)
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
    if side == "left":
        points[:, 0] *= -1.0

    unity_points = points @ np.diag([1.0, 1.0, -1.0])
    positions = np.zeros((OPENXR_JOINT_COUNT, 3), dtype=np.float64)
    positions[OPENXR_TO_MEDIAPIPE_21] = unity_points
    valid = np.zeros(OPENXR_JOINT_COUNT, dtype=bool)
    valid[OPENXR_TO_MEDIAPIPE_21] = True
    return HandJoints(tracked=True, positions=positions, valid=valid)


def test_left_and_right_slsqp_instances_can_solve_concurrently():
    args = _parse_args(["--confirm-hardware", HARDWARE_CONFIRMATION])
    RetargetingConfig.set_default_urdf_dir(
        PROJECT_ROOT / "assets" / "robots" / "hands"
    )
    runtimes = [build_hand_runtime(side, args) for side in ("left", "right")]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                solve_hand_runtime,
                runtime,
                _synthetic_openxr_hand(runtime.side),
                args,
                1.0,
            )
            for runtime in runtimes
        ]
        qposes = [future.result() for future in futures]

    assert all(qpos is not None and qpos.shape == (18,) for qpos in qposes)
    assert all(np.isfinite(qpos).all() for qpos in qposes)


def test_mirrored_openxr_hands_produce_mirrored_joint_solutions():
    """Guard the left base/right palm wrist-origin correspondence."""
    args = _parse_args(["--confirm-hardware", HARDWARE_CONFIRMATION])
    RetargetingConfig.set_default_urdf_dir(
        PROJECT_ROOT / "assets" / "robots" / "hands"
    )
    runtimes = {
        side: build_hand_runtime(side, args) for side in ("left", "right")
    }
    qposes = {
        side: solve_hand_runtime(
            runtime,
            _synthetic_openxr_hand(side),
            args,
            1.0,
        )
        for side, runtime in runtimes.items()
    }

    left = runtimes["left"]
    right = runtimes["right"]
    assert left.config.wrist_link_name == "left_hand_base"
    assert right.config.wrist_link_name == "right_hand_palm"
    for left_name in left.retargeting.joint_names:
        suffix = left_name.removeprefix("left_")
        right_name = f"right_{suffix}"
        left_value = qposes["left"][
            left.retargeting.joint_names.index(left_name)
        ]
        right_value = qposes["right"][
            right.retargeting.joint_names.index(right_name)
        ]
        assert left_value == pytest.approx(right_value, abs=8.0e-3)


def test_thumb_debug_reports_human_and_simulated_pinch_distance():
    args = _parse_args(
        [
            "--confirm-hardware",
            HARDWARE_CONFIRMATION,
            "--v2-thumb-debug",
        ]
    )
    RetargetingConfig.set_default_urdf_dir(
        PROJECT_ROOT / "assets" / "robots" / "hands"
    )
    runtime = build_hand_runtime("right", args)

    qpos = solve_hand_runtime(
        runtime,
        _synthetic_openxr_hand("right"),
        args,
        1.0,
    )

    assert qpos is not None
    assert "pinch(human,robot)=" in runtime.last_thumb_debug
    assert "projected=" in runtime.last_thumb_debug
