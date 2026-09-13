"""Observations, rewards, and terminations for the MJLab jump task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs import mdp as env_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from .asset import (
    JUMP_JOINT_NAMES,
    QARM_INITIAL_ROOT_HEIGHT,
    TORQUE_LIMIT_NM,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


ROBOT_CFG = SceneEntityCfg(
    "robot", joint_names=JUMP_JOINT_NAMES, preserve_order=True
)
_UP_B = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32)


def _robot(env: ManagerBasedRlEnv) -> Entity:
    return env.scene[ROBOT_CFG.name]


def _root_pos_relative_to_env(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = _robot(env)
    return robot.data.root_link_pos_w - env.scene.env_origins


def root_up(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = _robot(env)
    up_b = _UP_B.to(device=env.device).expand(env.num_envs, -1)
    return quat_apply(robot.data.root_link_quat_w, up_b)


def supported(env: ManagerBasedRlEnv) -> torch.Tensor:
    contact = env.scene["support_contact"]
    found = contact.data.found
    if found is None:
        raise RuntimeError("support_contact must expose the 'found' field")
    return found.reshape(env.num_envs, -1).amax(dim=1) > 0.0


def jump_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the 25-D normalized observation used by the Gymnasium task."""

    robot = _robot(env)
    root_pos = _root_pos_relative_to_env(env)
    root_vel = robot.data.root_link_lin_vel_w
    root_ang_vel = robot.data.root_link_ang_vel_b
    joint_pos = robot.data.joint_pos[:, ROBOT_CFG.joint_ids]
    joint_vel = robot.data.joint_vel[:, ROBOT_CFG.joint_ids]
    half_range = torch.maximum(
        torch.abs(robot.data.joint_pos_limits[:, ROBOT_CFG.joint_ids, 0]),
        torch.abs(robot.data.joint_pos_limits[:, ROBOT_CFG.joint_ids, 1]),
    ).clamp_min(1.0e-6)
    up = root_up(env)
    support = supported(env).to(dtype=torch.float32).unsqueeze(-1)
    last_action = env_mdp.last_action(env)

    return torch.cat(
        (
            torch.cat(
                (
                    root_pos[:, :2],
                    ((root_pos[:, 2] - QARM_INITIAL_ROOT_HEIGHT) / 0.5).unsqueeze(-1),
                ),
                dim=-1,
            ).clamp(-2.0, 2.0),
            torch.tanh(root_vel / 5.0),
            up,
            torch.tanh(root_ang_vel / 10.0),
            (joint_pos / half_range).clamp(-2.0, 2.0),
            torch.tanh(joint_vel / 10.0),
            support,
            last_action,
        ),
        dim=-1,
    )


def height_delta(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _root_pos_relative_to_env(env)[:, 2] - QARM_INITIAL_ROOT_HEIGHT


def height_reward(
    env: ManagerBasedRlEnv, target_height: float = 0.20
) -> torch.Tensor:
    return torch.clamp(height_delta(env) / target_height, min=-1.0, max=2.0)


def height_progress_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Dense approximation of the original per-step height-progress term."""

    return torch.clamp(_robot(env).data.root_link_lin_vel_w[:, 2] / 3.0, -1.0, 1.0)


def supported_vertical_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.where(
        supported(env),
        _robot(env).data.root_link_lin_vel_w[:, 2].clamp(-1.0, 1.0),
        torch.zeros(env.num_envs, device=env.device),
    )


def upright_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    return root_up(env)[:, 2].clamp_min(0.0)


def horizontal_displacement_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.linalg.vector_norm(_root_pos_relative_to_env(env)[:, :2], dim=-1)


def root_angular_velocity_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).data.root_link_ang_vel_b), dim=-1)


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action), dim=-1)


def torque_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = _robot(env)
    return torch.sum(
        torch.square(robot.data.qfrc_actuator[:, ROBOT_CFG.joint_ids])
        / (TORQUE_LIMIT_NM**2),
        dim=-1,
    )


def jump_success(
    env: ManagerBasedRlEnv,
    target_height: float = 0.20,
    upright_threshold: float = 0.45,
) -> torch.Tensor:
    return (
        (height_delta(env) >= target_height)
        & ~supported(env)
        & (root_up(env)[:, 2] > upright_threshold)
    ).to(dtype=torch.float32)


class OneShotJumpBonus:
    """Award the target-height bonus once per environment episode."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        target_height: float = 0.20,
        upright_threshold: float = 0.45,
    ) -> torch.Tensor:
        reached = jump_success(env, target_height, upright_threshold).bool()
        fresh = reached & ~self.awarded
        self.awarded |= reached
        return fresh.to(dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self.awarded[env_ids] = False


def fell(
    env: ManagerBasedRlEnv,
    minimum_height_delta: float = -0.06,
    minimum_up_z: float = 0.15,
    maximum_horizontal_displacement: float = 1.0,
) -> torch.Tensor:
    robot = _robot(env)
    state_finite = torch.isfinite(robot.data.root_link_pose_w).all(dim=-1)
    state_finite &= torch.isfinite(robot.data.root_link_vel_w).all(dim=-1)
    pos = _root_pos_relative_to_env(env)
    return (
        ~state_finite
        | (pos[:, 2] < QARM_INITIAL_ROOT_HEIGHT + minimum_height_delta)
        | (root_up(env)[:, 2] < minimum_up_z)
        | (torch.linalg.vector_norm(pos[:, :2], dim=-1) > maximum_horizontal_displacement)
    )


__all__ = [
    "ROBOT_CFG",
    "OneShotJumpBonus",
    "action_l2",
    "fell",
    "height_delta",
    "height_progress_reward",
    "height_reward",
    "horizontal_displacement_l2",
    "jump_observation",
    "jump_success",
    "root_angular_velocity_l2",
    "root_up",
    "supported",
    "supported_vertical_velocity",
    "torque_l2",
    "upright_reward",
]
