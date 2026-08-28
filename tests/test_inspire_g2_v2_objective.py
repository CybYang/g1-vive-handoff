from pathlib import Path

import numpy as np

from dex_retargeting.retargeting_config import RetargetingConfig
from scripts.inspire_g2_pose_adapter import (
    G2_LEFT_COLLISION_LINK_GROUPS,
    G2_LEFT_YAW_COUPLING_TERMS,
    G2_RIGHT_COLLISION_LINK_GROUPS,
    G2_RIGHT_YAW_COUPLING_TERMS,
)


PROJECT_ROOT = Path(__file__).parent.parent
ROBOT_DIR = PROJECT_ROOT / "assets" / "robots" / "hands"
CONFIG_PATH = (
    PROJECT_ROOT
    / "src"
    / "dex_retargeting"
    / "configs"
    / "teleop"
    / "inspire_g2_hand_left_dexpilot.yml"
)


def _build_v2():
    RetargetingConfig.set_default_urdf_dir(ROBOT_DIR)
    config = RetargetingConfig.load_from_file(
        CONFIG_PATH,
        override={
            "low_pass_alpha": 0.0,
            "joint_regularization_weight": 0.02,
            "joint_coupling_terms": G2_LEFT_YAW_COUPLING_TERMS,
            "joint_coupling_weight": 0.01,
            "collision_link_groups": G2_LEFT_COLLISION_LINK_GROUPS,
            "collision_min_distance": 0.014,
            "collision_weight": 0.3,
        },
    )
    return config.build()


def _build_right_v2():
    config_path = CONFIG_PATH.with_name("inspire_g2_hand_right_dexpilot.yml")
    RetargetingConfig.set_default_urdf_dir(ROBOT_DIR)
    config = RetargetingConfig.load_from_file(
        config_path,
        override={
            "low_pass_alpha": 0.0,
            "joint_regularization_weight": 0.02,
            "joint_coupling_terms": G2_RIGHT_YAW_COUPLING_TERMS,
            "joint_coupling_weight": 0.01,
            "collision_link_groups": G2_RIGHT_COLLISION_LINK_GROUPS,
            "collision_min_distance": 0.014,
            "collision_weight": 0.3,
        },
    )
    return config.build()


def _full_qpos(optimizer, target_values: dict[str, float]) -> np.ndarray:
    qpos = np.zeros(optimizer.robot.dof, dtype=np.float64)
    for name, value in target_values.items():
        qpos[optimizer.robot.get_joint_index(name)] = value
    return optimizer.adaptor.forward_qpos(qpos)


def _target_vectors(optimizer, qpos: np.ndarray) -> np.ndarray:
    optimizer.robot.compute_forward_kinematics(qpos)
    body_pos = np.array(
        [
            optimizer.robot.get_link_pose(index)[:3, 3]
            for index in optimizer.computed_link_indices
        ]
    )
    return (
        body_pos[optimizer.task_link_indices.numpy()]
        - body_pos[optimizer.origin_link_indices.numpy()]
    )


def _finite_difference_gradient(objective, qpos: np.ndarray) -> np.ndarray:
    step = 1e-4
    gradient = np.empty_like(qpos)
    for index in range(qpos.size):
        upper = qpos.copy()
        lower = qpos.copy()
        upper[index] += step
        lower[index] -= step
        gradient[index] = (
            objective(upper, np.empty(0)) - objective(lower, np.empty(0))
        ) / (2 * step)
    return gradient


def test_v2_collision_term_detects_crossed_yaw_pose():
    retargeting = _build_v2()
    optimizer = retargeting.optimizer
    open_qpos = _full_qpos(optimizer, {})
    crossed_qpos = _full_qpos(
        optimizer,
        {
            "left_index_yaw_joint": np.deg2rad(15.0),
            "left_middle_yaw_joint": np.deg2rad(-15.0),
        },
    )

    assert optimizer.minimum_collision_distance(open_qpos) > 0.014
    assert optimizer.minimum_collision_distance(crossed_qpos) < 0.002

    optimizer.set_joint_reference(
        {
            "left_index_yaw_joint": 0.0,
            "left_middle_yaw_joint": 0.0,
        },
        {
            "left_index_yaw_joint": 4.0,
            "left_middle_yaw_joint": 4.0,
        },
    )
    target_vector = _target_vectors(optimizer, crossed_qpos)
    target_qpos = crossed_qpos[optimizer.idx_pin2target]
    objective = optimizer.get_objective_function(
        target_vector,
        np.array([]),
        target_qpos,
    )
    gradient = np.zeros_like(target_qpos)
    value = objective(target_qpos, gradient)

    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    assert optimizer.last_objective_terms["collision"] > 0
    assert optimizer.last_objective_terms["joint"] > 0


def test_v2_collision_loss_pushes_an_interpenetrating_target_apart():
    retargeting = _build_v2()
    optimizer = retargeting.optimizer
    crossed_qpos = _full_qpos(
        optimizer,
        {
            "left_index_yaw_joint": np.deg2rad(15.0),
            "left_middle_yaw_joint": np.deg2rad(-15.0),
        },
    )
    target_vector = _target_vectors(optimizer, crossed_qpos)
    optimizer.set_joint_reference(
        {
            "left_index_yaw_joint": 0.0,
            "left_middle_yaw_joint": 0.0,
        },
        {
            "left_index_yaw_joint": 4.0,
            "left_middle_yaw_joint": 4.0,
        },
    )

    result = optimizer.retarget(
        target_vector,
        fixed_qpos=np.array([]),
        last_qpos=np.zeros(optimizer.opt_dof),
    )
    result_qpos = np.zeros(optimizer.robot.dof, dtype=np.float64)
    result_qpos[optimizer.idx_pin2target] = result
    result_qpos = optimizer.adaptor.forward_qpos(result_qpos)

    assert optimizer.minimum_collision_distance(result_qpos) > 0.012


def test_right_v2_objective_builds_and_reports_all_added_loss_terms():
    retargeting = _build_right_v2()
    optimizer = retargeting.optimizer
    open_qpos = _full_qpos(optimizer, {})
    target_vector = _target_vectors(optimizer, open_qpos)
    optimizer.set_joint_reference(
        {
            "right_index_yaw_joint": 0.0,
            "right_middle_yaw_joint": 0.0,
        },
        {
            "right_index_yaw_joint": 4.0,
            "right_middle_yaw_joint": 4.0,
        },
    )

    result = optimizer.retarget(
        target_vector,
        fixed_qpos=np.array([]),
        last_qpos=np.zeros(optimizer.opt_dof),
    )

    assert np.isfinite(result).all()
    assert optimizer.last_objective_terms.keys() == {
        "total",
        "task",
        "joint",
        "coupling",
        "collision",
        "temporal",
        "minimum_collision_distance_m",
    }
    assert np.isfinite(list(optimizer.last_objective_terms.values())).all()


def test_right_v2_objective_gradient_matches_finite_difference():
    retargeting = _build_right_v2()
    optimizer = retargeting.optimizer
    target_qpos = np.full(optimizer.opt_dof, 0.2, dtype=np.float64)
    full_qpos = np.zeros(optimizer.robot.dof, dtype=np.float64)
    full_qpos[optimizer.idx_pin2target] = target_qpos
    full_qpos = optimizer.adaptor.forward_qpos(full_qpos)
    target_vector = _target_vectors(optimizer, full_qpos)
    last_qpos = np.zeros_like(target_qpos)
    objective = optimizer.get_objective_function(
        target_vector,
        np.array([]),
        last_qpos,
    )

    analytic_gradient = np.empty_like(target_qpos)
    objective(target_qpos, analytic_gradient)
    reported_temporal_loss = optimizer.last_objective_terms["temporal"]
    finite_difference_gradient = _finite_difference_gradient(
        objective,
        target_qpos,
    )

    np.testing.assert_allclose(
        analytic_gradient,
        finite_difference_gradient,
        rtol=1e-4,
        atol=1e-7,
    )
    expected_temporal_loss = optimizer.norm_delta * np.sum(
        (target_qpos - last_qpos) ** 2
    )
    assert np.isclose(reported_temporal_loss, expected_temporal_loss)
