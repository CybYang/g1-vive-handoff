from abc import abstractmethod
from typing import Any, Dict, List, Optional

import nlopt
import numpy as np
import torch

from dex_retargeting.kinematics_adaptor import (
    KinematicAdaptor,
    MimicJointKinematicAdaptor,
)
from dex_retargeting.robot_wrapper import RobotWrapper


class Optimizer:
    retargeting_type = "BASE"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray,
    ):
        self.robot = robot
        self.num_joints = robot.dof

        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(
                    f"Joint {target_joint_name} given does not appear to be in robot XML."
                )
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.idx_pin2target = np.array(idx_pin2target)

        self.idx_pin2fixed = np.array(
            [i for i in range(robot.dof) if i not in idx_pin2target], dtype=int
        )
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        self.opt_dof = len(idx_pin2target)  # This dof includes the mimic joints

        # Target
        self.target_link_human_indices = target_link_human_indices

        # Free joint
        link_names = robot.link_names
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6

        # Kinematics adaptor
        self.adaptor: Optional[KinematicAdaptor] = None

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(
                f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}"
            )
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def get_link_indices(self, target_link_names):
        return [self.robot.get_link_index(link_name) for link_name in target_link_names]

    def set_kinematic_adaptor(self, adaptor: KinematicAdaptor):
        self.adaptor = adaptor

        # Remove mimic joints from fixed joint list
        if isinstance(adaptor, MimicJointKinematicAdaptor):
            fixed_idx = self.idx_pin2fixed
            mimic_idx = adaptor.idx_pin2mimic
            new_fixed_id = np.array(
                [x for x in fixed_idx if x not in mimic_idx], dtype=int
            )
            self.idx_pin2fixed = new_fixed_id

    def retarget(self, ref_value, fixed_qpos, last_qpos):
        """
        Compute the retargeting results using non-linear optimization
        Args:
            ref_value: the reference value in cartesian space as input, different optimizer has different reference
            fixed_qpos: the fixed value (not optimized) in retargeting, consistent with self.fixed_joint_names
            last_qpos: the last retargeting results or initial value, consistent with function return

        Returns: joint position of robot, the joint order and dim is consistent with self.target_joint_names

        """
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )
        objective_fn = self.get_objective_function(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )

        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            print(e)
            return np.array(last_qpos, dtype=np.float32)

    @abstractmethod
    def get_objective_function(
        self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        pass

    @property
    def fixed_joint_names(self):
        joint_names = self.robot.dof_joint_names
        return [joint_names[i] for i in self.idx_pin2fixed]


class PositionOptimizer(Optimizer):
    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check and cache link indices
        self.target_link_indices = self.get_link_indices(target_link_names)

        self.opt.set_ftol_abs(1e-5)

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.target_link_indices
            ]
            body_pos = np.stack(
                [pose[:3, 3] for pose in target_link_poses], axis=0
            )  # (n ,3)

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            temporal_loss = self.norm_delta * float(
                np.sum((x - last_qpos) ** 2)
            )
            result = huber_distance.cpu().detach().item() + temporal_loss

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                # Compute the gradient to the qpos
                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class VectorOptimizer(Optimizer):
    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=1.0,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = scaling

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Cache link indices that will involve in kinematics computation
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            temporal_loss = self.norm_delta * float(
                np.sum((x - last_qpos) ** 2)
            )
            result = huber_distance.cpu().detach().item() + temporal_loss

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class DexPilotOptimizer(Optimizer):
    """Retargeting optimizer using the method proposed in DexPilot

    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135

    Args:
        robot:
        target_joint_names:
        finger_tip_link_names:
        wrist_link_name:
        gamma:
        project_dist:
        escape_dist:
        eta1:
        eta2:
        scaling:
    """

    retargeting_type = "DEXPILOT"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        target_link_human_indices: Optional[np.ndarray] = None,
        huber_delta=0.03,
        norm_delta=4e-3,
        # DexPilot parameters
        # gamma=2.5e-3,
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
        joint_regularization_weight=0.0,
        joint_coupling_terms: Optional[List[Dict[str, Any]]] = None,
        joint_coupling_weight=0.0,
        collision_link_groups: Optional[List[List[str]]] = None,
        collision_min_distance=0.0,
        collision_weight=0.0,
        collision_samples_per_segment=5,
    ):
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"DexPilot optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)

        origin_link_index, task_link_index = self.generate_link_indices(
            self.num_fingers
        )

        if target_link_human_indices is None:
            target_link_human_indices = (
                np.stack([origin_link_index, task_link_index], axis=0) * 4
            ).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]

        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta

        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        # Optional robot-specific objective terms.  They default to zero so
        # every upstream configuration keeps the original DexPilot behavior.
        if joint_regularization_weight < 0:
            raise ValueError("joint_regularization_weight must be non-negative")
        if joint_coupling_weight < 0:
            raise ValueError("joint_coupling_weight must be non-negative")
        if collision_min_distance < 0 or collision_weight < 0:
            raise ValueError("collision parameters must be non-negative")
        if collision_samples_per_segment < 2:
            raise ValueError("collision_samples_per_segment must be at least 2")
        self.joint_regularization_weight = float(joint_regularization_weight)
        self._joint_reference = np.zeros(self.opt_dof, dtype=np.float64)
        self._joint_reference_weights = np.zeros(self.opt_dof, dtype=np.float64)

        self.joint_coupling_weight = float(joint_coupling_weight)
        self.joint_coupling_terms = []
        for term in joint_coupling_terms or []:
            names = list(term["joint_names"])
            coefficients = np.asarray(term["coefficients"], dtype=np.float64)
            if len(names) < 2 or coefficients.shape != (len(names),):
                raise ValueError(
                    "each joint coupling term needs matching joint_names and coefficients"
                )
            try:
                indices = np.asarray(
                    [self.target_joint_names.index(name) for name in names],
                    dtype=int,
                )
            except ValueError as exc:
                raise ValueError(f"unknown joint in coupling term: {names}") from exc
            self.joint_coupling_terms.append(
                (
                    indices,
                    coefficients,
                    float(term.get("target", 0.0)),
                    float(term.get("weight", 1.0)),
                )
            )

        self.collision_link_groups = [
            list(group) for group in (collision_link_groups or [])
        ]
        if self.collision_link_groups and any(
            len(group) < 2 for group in self.collision_link_groups
        ):
            raise ValueError("each collision link group needs at least two links")
        self.collision_min_distance = float(collision_min_distance)
        self.collision_weight = float(collision_weight)
        self.collision_samples_per_segment = int(collision_samples_per_segment)
        self.last_objective_terms: Dict[str, float] = {}

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            dict.fromkeys(
                target_origin_link_names
                + target_task_link_names
                + [name for group in self.collision_link_groups for name in group]
            )
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )
        self.collision_group_indices = [
            torch.tensor(
                [self.computed_link_names.index(name) for name in group],
                dtype=torch.long,
            )
            for group in self.collision_link_groups
        ]

        # Sanity check and cache link indices
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

        # DexPilot cache
        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = self.set_dexpilot_cache(self.num_fingers, eta1, eta2)

    def set_joint_reference(
        self,
        values: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set a soft per-joint reference for the next retargeting calls.

        A zero weight leaves a joint completely under the original DexPilot
        objective.  The reference is deliberately stateful so an online input
        adapter can update it immediately before each optimization step.
        """
        reference = np.zeros(self.opt_dof, dtype=np.float64)
        reference_weights = np.zeros(self.opt_dof, dtype=np.float64)
        for name, value in values.items():
            if name not in self.target_joint_names:
                raise ValueError(f"unknown target joint reference: {name}")
            index = self.target_joint_names.index(name)
            reference[index] = float(value)
            weight = 1.0 if weights is None else float(weights.get(name, 0.0))
            if weight < 0:
                raise ValueError("joint reference weights must be non-negative")
            reference_weights[index] = weight
        self._joint_reference = reference
        self._joint_reference_weights = reference_weights

    def clear_joint_reference(self) -> None:
        self._joint_reference.fill(0.0)
        self._joint_reference_weights.fill(0.0)

    def _sample_collision_polyline(
        self, body_pos: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        control = body_pos[indices, :]
        samples = []
        fractions = torch.linspace(
            0.0,
            1.0,
            self.collision_samples_per_segment,
            dtype=body_pos.dtype,
            device=body_pos.device,
        )
        for segment_index in range(control.shape[0] - 1):
            start = control[segment_index]
            end = control[segment_index + 1]
            segment = (
                start[None, :] * (1.0 - fractions[:, None])
                + end[None, :] * fractions[:, None]
            )
            if segment_index:
                segment = segment[1:, :]
            samples.append(segment)
        return torch.cat(samples, dim=0)

    def _collision_penalty(self, body_pos: torch.Tensor) -> tuple[torch.Tensor, float]:
        if (
            self.collision_weight <= 0
            or self.collision_min_distance <= 0
            or len(self.collision_group_indices) < 2
        ):
            return body_pos.new_tensor(0.0), float("inf")

        polylines = [
            self._sample_collision_polyline(body_pos, indices)
            for indices in self.collision_group_indices
        ]
        penalties = []
        minimum_distance = float("inf")
        # Groups are ordered thumb-to-pinky (or index-to-pinky for G2), so
        # adjacent groups are the pairs that can physically cross first.
        for first, second in zip(polylines[:-1], polylines[1:]):
            distance = torch.cdist(first, second).min()
            minimum_distance = min(minimum_distance, float(distance.detach().cpu()))
            normalized_overlap = torch.relu(
                (self.collision_min_distance - distance)
                / self.collision_min_distance
            )
            penalties.append(normalized_overlap.square())
        return torch.stack(penalties).mean(), minimum_distance

    def minimum_collision_distance(self, robot_qpos: np.ndarray) -> float:
        """Return the sampled centerline distance for the current robot pose."""
        qpos = np.asarray(robot_qpos, dtype=np.float64)
        if qpos.shape != (self.robot.dof,):
            raise ValueError(
                f"robot_qpos must have shape {(self.robot.dof,)}, got {qpos.shape}"
            )
        if len(self.collision_group_indices) < 2:
            return float("inf")
        self.robot.compute_forward_kinematics(qpos)
        body_pos = np.array(
            [
                self.robot.get_link_pose(index)[:3, 3]
                for index in self.computed_link_indices
            ]
        )
        _, distance = self._collision_penalty(torch.as_tensor(body_pos))
        return distance

    @staticmethod
    def generate_link_indices(num_fingers):
        """
        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        """
        origin_link_index = []
        task_link_index = []

        # Add indices for connections between fingers
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)

        # Add indices for connections to the base (0)
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)

        return origin_link_index, task_link_index

    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """
        Example:
        >>> set_dexpilot_cache(4, 0.1, 0.2)
        (array([False, False, False, False, False, False]),
        [1, 2, 2],
        [0, 0, 1],
        array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        """
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        projected_dist = np.array(
            [eta1] * (num_fingers - 1)
            + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2)
        )

        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        # Freeze this frame's human-derived soft targets while NLopt evaluates
        # the objective repeatedly.  The online adapter updates them before
        # the next call to ``retarget``.
        joint_reference = self._joint_reference.copy()
        joint_reference_weights = self._joint_reference_weights.copy()

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][
            target_vec_dist[0:len_s1] < self.project_dist
        ] = True
        self.projected[:len_s1][
            target_vec_dist[0:len_s1] > self.escape_dist
        ] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj],
            target_vec_dist[len_s1:len_proj] <= 0.03,
        )

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate(
                [
                    weight,
                    np.ones(self.num_fingers, dtype=np.float32) * len_proj
                    + self.num_fingers,
                ]
            )
        )

        # Compute reference distance vector
        normal_vec = target_vector * self.scaling  # (10, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)

        # Compute final reference vector
        reference_vec = np.where(
            self.projected[:, None], projected_vec, normal_vec[:len_proj]
        )  # (6, 3)
        reference_vec = np.concatenate(
            [reference_vec, normal_vec[len_proj:]], axis=0
        )  # (10, 3)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            # Different from the original DexPilot, we use huber loss here instead of the squared dist
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            task_loss = (
                self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
                * weight
                / (robot_vec.shape[0])
            ).sum()
            task_loss = task_loss.sum()
            collision_loss, minimum_collision_distance = self._collision_penalty(
                torch_body_pos
            )
            body_loss = task_loss + self.collision_weight * collision_loss

            joint_error = x - joint_reference
            joint_loss = self.joint_regularization_weight * float(
                np.sum(joint_reference_weights * joint_error**2)
            )
            coupling_loss = 0.0
            for indices, coefficients, target, term_weight in self.joint_coupling_terms:
                error = float(np.dot(coefficients, x[indices]) - target)
                coupling_loss += (
                    self.joint_coupling_weight * term_weight * error**2
                )
            temporal_loss = self.norm_delta * float(
                np.sum((x - last_qpos) ** 2)
            )

            task_value = float(task_loss.detach().cpu())
            collision_value = float(collision_loss.detach().cpu())
            result = (
                float(body_loss.detach().cpu())
                + joint_loss
                + coupling_loss
                + temporal_loss
            )
            self.last_objective_terms = {
                "task": task_value,
                "joint": joint_loss,
                "coupling": coupling_loss,
                "collision": self.collision_weight * collision_value,
                "temporal": temporal_loss,
                "minimum_collision_distance_m": minimum_collision_distance,
                "total": result,
            }

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                body_loss.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                # In the original DexPilot, γ = 2.5 × 10−3 is a weight on regularizing the Allegro angles to zero
                # which is equivalent to fully opened the hand
                # In our implementation, we regularize the joint angles to the previous joint angles
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad_qpos += (
                    2
                    * self.joint_regularization_weight
                    * joint_reference_weights
                    * joint_error
                )
                for indices, coefficients, target, term_weight in self.joint_coupling_terms:
                    error = float(np.dot(coefficients, x[indices]) - target)
                    grad_qpos[indices] += (
                        2
                        * self.joint_coupling_weight
                        * term_weight
                        * error
                        * coefficients
                    )

                grad[:] = grad_qpos[:]

            return result

        return objective
