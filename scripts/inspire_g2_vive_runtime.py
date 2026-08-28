#!/usr/bin/env python3
"""Shared VIVE/OpenXR -> Inspire G2 retargeting runtime.

This module contains the current retargeting algorithm and no visualization or
hardware I/O.  The SAPIEN preview and the physical-hand entry point import the
same functions so their V2 objective cannot silently diverge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from scripts.inspire_g2_pose_adapter import (
    InspireG2PoseAdapter,
    g2_collision_link_groups,
    g2_yaw_coupling_terms,
    human_four_finger_angles,
    human_thumb_cmc_elevation,
    human_thumb_flexion_angles,
    thumb_opposition_fraction,
)
from scripts.vive_openxr_hand import (
    CanonicalHandLowPass,
    OpenXRHandFrameError,
    canonicalize_openxr_hand,
)


def selected_sides(hand_side: str) -> tuple[str, ...]:
    if hand_side == "both":
        return ("left", "right")
    if hand_side in {"left", "right"}:
        return (hand_side,)
    raise ValueError("hand_side must be 'left', 'right', or 'both'")


@dataclass
class HandRuntime:
    side: str
    hand_type: HandType
    config: RetargetingConfig
    retargeting: Any
    adapter: InspireG2PoseAdapter
    landmark_filter: CanonicalHandLowPass
    retargeting_name_to_index: dict[str, int]
    robot: Any = None
    retargeting_to_sapien: np.ndarray | None = None
    last_valid_frame: float = 0.0
    last_solve_ms: float = 0.0
    accepted: int = 0
    invalid: int = 0
    last_ring_debug: str = "ring=waiting"
    last_thumb_debug: str = "thumb=waiting"


def build_hand_runtime(side: str, args: Any) -> HandRuntime:
    """Build one independent optimizer and temporal state for one hand."""
    hand_type = HandType[side]
    config_path = get_default_config_path(
        RobotName.inspire_g2,
        RetargetingType.dexpilot,
        hand_type,
    )
    if config_path is None:
        raise RuntimeError(f"{side} Inspire G2 DexPilot config was not found")

    override = {
        "scaling_factor": args.scaling_factor,
        "low_pass_alpha": args.low_pass_alpha,
    }
    if args.objective == "v2":
        override.update(
            {
                "joint_regularization_weight": args.v2_joint_weight,
                "joint_coupling_terms": g2_yaw_coupling_terms(side),
                "joint_coupling_weight": args.v2_yaw_coupling_weight,
                "collision_link_groups": g2_collision_link_groups(side),
                "collision_min_distance": args.v2_collision_min_distance,
                "collision_weight": args.v2_collision_weight,
            }
        )
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()
    return HandRuntime(
        side=side,
        hand_type=hand_type,
        config=config,
        retargeting=retargeting,
        adapter=InspireG2PoseAdapter(
            alpha=args.low_pass_alpha,
            hand_side=side,
        ),
        landmark_filter=CanonicalHandLowPass(args.landmark_alpha),
        retargeting_name_to_index={
            name: index for index, name in enumerate(retargeting.joint_names)
        },
    )


def solve_hand_runtime(
    runtime: HandRuntime,
    openxr_hand: Any,
    args: Any,
    now: float | None = None,
) -> np.ndarray | None:
    """Retarget one tracked hand; return ``None`` when tracking is invalid."""
    try:
        points = runtime.landmark_filter.update(
            canonicalize_openxr_hand(
                openxr_hand,
                hand_type=runtime.hand_type,
            )
        )
    except OpenXRHandFrameError:
        runtime.invalid += 1
        return None

    indices = runtime.retargeting.optimizer.target_link_human_indices
    ref_value = points[indices[1, :], :] - points[indices[0, :], :]
    values: dict[str, float] = {}
    weights: dict[str, float] = {}
    if args.objective == "v2":
        values, weights = runtime.adapter.soft_joint_references(
            points,
            flexion_weight=args.v2_flexion_reference_weight,
            yaw_weight=args.v2_yaw_reference_weight,
            middle_yaw_weight=args.v2_middle_yaw_reference_weight,
            thumb_flexion_weight=args.v2_thumb_flexion_reference_weight,
            thumb_yaw_weight=args.v2_thumb_yaw_reference_weight,
            thumb_cmc_weight=args.v2_thumb_cmc_reference_weight,
            thumb_ip_weight=args.v2_thumb_ip_reference_weight,
            thumb_opposition_open_ratio=args.v2_thumb_opposition_open_ratio,
            thumb_opposition_opposed_ratio=(
                args.v2_thumb_opposition_opposed_ratio
            ),
            thumb_opposition_exponent=args.v2_thumb_opposition_exponent,
            thumb_cmc_open_elevation_deg=args.v2_thumb_cmc_open_elevation_deg,
            thumb_cmc_opposed_elevation_deg=(
                args.v2_thumb_cmc_opposed_elevation_deg
            ),
            ring_open_weight_multiplier=args.v2_ring_open_weight_multiplier,
            ring_open_fraction=args.v2_ring_open_fraction,
            ring_release_fraction=args.v2_ring_release_fraction,
            pinky_open_weight_multiplier=args.v2_pinky_open_weight_multiplier,
            pinky_open_fraction=args.v2_pinky_open_fraction,
            pinky_release_fraction=args.v2_pinky_release_fraction,
        )
        runtime.retargeting.optimizer.set_joint_reference(values, weights)
    else:
        runtime.retargeting.optimizer.clear_joint_reference()

    solve_start = time.perf_counter()
    qpos = runtime.retargeting.retarget(ref_value)
    runtime.last_solve_ms = (time.perf_counter() - solve_start) * 1000.0

    name_to_index = runtime.retargeting_name_to_index
    prefix = f"{runtime.side}_"
    if args.objective == "v2" and args.v2_ring_debug:
        raw_mcp, raw_pip = human_four_finger_angles(points)["ring"]
        mcp_name = f"{prefix}ring_mcp_joint"
        pip_name = f"{prefix}ring_pip_joint"
        runtime.last_ring_debug = (
            f"{runtime.side} ring "
            f"human=({np.rad2deg(raw_mcp):.1f},"
            f"{np.rad2deg(raw_pip):.1f})deg "
            f"ref=({values[mcp_name]:.3f},{values[pip_name]:.3f})rad "
            f"q=({qpos[name_to_index[mcp_name]]:.3f},"
            f"{qpos[name_to_index[pip_name]]:.3f})rad "
            f"w={weights[mcp_name]:.2f}"
        )
    if args.objective == "v2" and args.v2_thumb_debug:
        human_mcp, human_ip = human_thumb_flexion_angles(points)
        human_cmc = human_thumb_cmc_elevation(points)
        opposition_ratio, opposition_fraction = thumb_opposition_fraction(
            points,
            open_ratio=args.v2_thumb_opposition_open_ratio,
            opposed_ratio=args.v2_thumb_opposition_opposed_ratio,
            exponent=args.v2_thumb_opposition_exponent,
        )
        yaw_name = f"{prefix}thumb_yaw_joint"
        cmc_name = f"{prefix}thumb_mcp_joint"
        passive_mcp_name = f"{prefix}thumb_pip_joint"
        ip_name = f"{prefix}thumb_dip_joint"
        mode = (
            "split"
            if args.v2_thumb_flexion_reference_weight is None
            else "legacy"
        )
        yaw_reference = values.get(yaw_name)
        yaw_reference_text = (
            "off" if yaw_reference is None else f"{yaw_reference:.3f}"
        )
        human_pinch_distance = float(np.linalg.norm(points[4] - points[8]))
        robot = runtime.retargeting.optimizer.robot
        robot.compute_forward_kinematics(qpos)
        thumb_tip = robot.get_link_pose(
            robot.get_link_index(f"{prefix}thumb_force_sensor")
        )[:3, 3]
        index_tip = robot.get_link_pose(
            robot.get_link_index(f"{prefix}index_force_sensor")
        )[:3, 3]
        robot_pinch_distance = float(np.linalg.norm(thumb_tip - index_tip))
        pinch_projected = bool(runtime.retargeting.optimizer.projected[0])
        runtime.last_thumb_debug = (
            f"{runtime.side} thumb mode={mode} "
            f"human(cmc,mcp,ip)=({np.rad2deg(human_cmc):.1f},"
            f"{np.rad2deg(human_mcp):.1f},{np.rad2deg(human_ip):.1f})deg "
            f"opp(ratio,fraction)=({opposition_ratio:.3f},"
            f"{opposition_fraction:.3f}) "
            f"ref(yaw,cmc,ip)=({yaw_reference_text},"
            f"{values[cmc_name]:.3f},{values[ip_name]:.3f})rad "
            "q(yaw,cmc,passive_mcp,ip)="
            f"({qpos[name_to_index[yaw_name]]:.3f},"
            f"{qpos[name_to_index[cmc_name]]:.3f},"
            f"{qpos[name_to_index[passive_mcp_name]]:.3f},"
            f"{qpos[name_to_index[ip_name]]:.3f})rad "
            f"pinch(human,robot)=({human_pinch_distance * 1000:.1f},"
            f"{robot_pinch_distance * 1000:.1f})mm "
            f"projected={int(pinch_projected)} "
            f"w(yaw,cmc,ip)=({weights.get(yaw_name, 0.0):.2f},"
            f"{weights[cmc_name]:.2f},{weights[ip_name]:.2f})"
        )

    runtime.accepted += 1
    runtime.last_valid_frame = time.monotonic() if now is None else now
    return qpos


def objective_status(terms: dict[str, float]) -> str:
    if not terms:
        return "objective=waiting"
    return (
        f"loss total={terms.get('total', float('nan')):.5f} "
        f"task={terms.get('task', float('nan')):.5f} "
        f"joint={terms.get('joint', 0.0):.5f} "
        f"coupling={terms.get('coupling', 0.0):.5f} "
        f"collision={terms.get('collision', 0.0):.5f} "
        f"temporal={terms.get('temporal', 0.0):.5f} "
        f"gap={terms.get('minimum_collision_distance_m', float('nan')) * 1000:.1f}mm"
    )
