#!/usr/bin/env python3
"""Read-only deployment self-check for the G1 hand-retargeting package.

This script never opens a serial port and never sends a robot command.  It
imports the packaged dependencies, builds both independent optimizers, solves
one synthetic OpenXR frame per side, and validates both 13-joint mappings.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dex_retargeting.retargeting_config import RetargetingConfig
from scripts.inspire_g2_hardware import load_joint_mapping, qpos_to_angle_set
from scripts.inspire_g2_vive_hardware import (
    HARDWARE_CONFIRMATION,
    _parse_args,
    _validate_mapping,
)
from scripts.inspire_g2_vive_runtime import build_hand_runtime, solve_hand_runtime
from scripts.vive_focus_udp_receiver import (
    OPENXR_JOINT_COUNT,
    OPENXR_TO_MEDIAPIPE_21,
    HandJoints,
)


REQUIRED_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "torch",
    "nlopt",
    "pin",
    "pytransform3d",
    "anytree",
    "PyYAML",
    "lxml",
    "six",
    "pyserial",
)


def _synthetic_openxr_hand(side: str) -> HandJoints:
    points = np.zeros((21, 3), dtype=np.float64)
    points[1:5] = [
        [-0.018, 0.008, 0.000],
        [-0.033, 0.017, 0.002],
        [-0.046, 0.027, 0.003],
        [-0.057, 0.038, 0.004],
    ]
    finger_indices = (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    )
    for indices, x_value in zip(
        finger_indices,
        (-0.032, -0.011, 0.011, 0.033),
        strict=True,
    ):
        for segment, index in enumerate(indices, start=1):
            points[index] = [x_value, 0.025 * segment, 0.001 * segment]
    if side == "left":
        points[:, 0] *= -1.0

    # Convert the right-handed test geometry back to the Unity left-handed
    # representation accepted by canonicalize_openxr_hand().
    unity_points = points @ np.diag([1.0, 1.0, -1.0])
    positions = np.zeros((OPENXR_JOINT_COUNT, 3), dtype=np.float64)
    positions[OPENXR_TO_MEDIAPIPE_21] = unity_points
    valid = np.zeros(OPENXR_JOINT_COUNT, dtype=bool)
    valid[OPENXR_TO_MEDIAPIPE_21] = True
    return HandJoints(tracked=True, positions=positions, valid=valid)


def _version_report() -> None:
    for distribution in REQUIRED_DISTRIBUTIONS:
        version = importlib.metadata.version(distribution)
        print(f"  {distribution}={version}")


def _parse_preflight_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-g1",
        action="store_true",
        help="fail unless this is Python 3.10 on aarch64",
    )
    return parser.parse_args()


def main() -> int:
    preflight_args = _parse_preflight_args()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    architecture = platform.machine().lower()
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Architecture: {architecture}")
    if python_version != "3.10":
        raise RuntimeError(f"Python 3.10 is required, found {python_version}")
    if preflight_args.strict_g1 and architecture not in {"aarch64", "arm64"}:
        raise RuntimeError(f"G1 PC2 must be aarch64, found {architecture}")
    if architecture not in {"aarch64", "arm64"}:
        print("NOTE: non-G1 architecture accepted for package testing only.")

    print("Dependency versions:")
    _version_report()

    urdf_root = PROJECT_ROOT / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(urdf_root)
    runtime_args = _parse_args(
        ["--confirm-hardware", HARDWARE_CONFIRMATION]
    )

    for side in ("left", "right"):
        runtime = build_hand_runtime(side, runtime_args)
        mapping_path = PROJECT_ROOT / "scripts" / f"inspire_g2_{side}_mapping.yaml"
        mapping = load_joint_mapping(mapping_path)
        _validate_mapping(side, mapping, runtime.retargeting.joint_names)
        qpos = solve_hand_runtime(
            runtime,
            _synthetic_openxr_hand(side),
            runtime_args,
            1.0,
        )
        if qpos is None or not np.isfinite(qpos).all():
            raise RuntimeError(f"{side} synthetic retargeting failed")
        angle_set = qpos_to_angle_set(
            qpos,
            runtime.retargeting.joint_names,
            mapping,
        )
        if len(angle_set) != 13:
            raise RuntimeError(f"{side} mapping did not produce 13 values")
        print(
            f"[{side}] OK: URDF={Path(runtime.config.urdf_path).name}, "
            f"dof={qpos.size}, solve={runtime.last_solve_ms:.1f} ms, "
            f"angleSet={angle_set}"
        )

    if "sapien" in sys.modules or "mediapipe" in sys.modules:
        raise RuntimeError("G1 preflight unexpectedly imported simulation/vision code")
    print("PASS: no serial port opened; both G1 hand pipelines are self-contained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
