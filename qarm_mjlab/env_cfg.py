"""MJLab manager-based configuration for free-base Qarm jumping."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from . import mdp
from .asset import (
    DEFAULT_MODEL_PATH,
    JUMP_JOINT_NAMES,
    TORQUE_LIMIT_NM,
    get_qarm_robot_cfg,
)


def make_qarm_jump_env_cfg(
    *,
    play: bool = False,
    num_envs: int | None = None,
    model_path=DEFAULT_MODEL_PATH,
    target_height: float = 0.20,
) -> ManagerBasedRlEnvCfg:
    """Create the vectorized free-base jump environment configuration."""

    if num_envs is None:
        num_envs = 16 if play else 4096
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    support_sensor = ContactSensorCfg(
        name="support_contact",
        primary=ContactMatch(
            mode="subtree", pattern=r"^qarm_root$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="maxforce",
        num_slots=1,
    )
    state_term = ObservationTermCfg(func=mdp.jump_observation)
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=2.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_qarm_robot_cfg(model_path)},
            sensors=(support_sensor,),
            extent=1.5,
        ),
        observations={
            "actor": ObservationGroupCfg(
                terms={"state": state_term},
                concatenate_terms=True,
                enable_corruption=not play,
            ),
            "critic": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=mdp.jump_observation)},
                concatenate_terms=True,
                enable_corruption=False,
            ),
        },
        actions={
            "joint_effort": JointEffortActionCfg(
                entity_name="robot",
                actuator_names=JUMP_JOINT_NAMES,
                scale=TORQUE_LIMIT_NM,
                preserve_order=True,
            )
        },
        events={
            "reset_base": EventTermCfg(
                func=env_mdp.reset_root_state_uniform,
                mode="reset",
                params={"pose_range": {}, "velocity_range": {}},
            ),
            "reset_joints": EventTermCfg(
                func=env_mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.02, 0.02),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=JUMP_JOINT_NAMES, preserve_order=True
                    ),
                },
            ),
        },
        rewards={
            "survival": RewardTermCfg(func=env_mdp.is_alive, weight=0.02),
            "height": RewardTermCfg(
                func=mdp.height_reward,
                weight=2.0,
                params={"target_height": target_height},
            ),
            "height_progress": RewardTermCfg(
                func=mdp.height_progress_reward, weight=1.5
            ),
            "upright": RewardTermCfg(func=mdp.upright_reward, weight=0.35),
            "supported_vertical_velocity": RewardTermCfg(
                func=mdp.supported_vertical_velocity, weight=0.25
            ),
            "horizontal_displacement": RewardTermCfg(
                func=mdp.horizontal_displacement_l2, weight=-0.08
            ),
            "root_angular_velocity": RewardTermCfg(
                func=mdp.root_angular_velocity_l2, weight=-0.05
            ),
            "action": RewardTermCfg(func=mdp.action_l2, weight=-0.004),
            "action_rate": RewardTermCfg(
                func=env_mdp.action_rate_l2, weight=-0.002
            ),
            "torque": RewardTermCfg(func=mdp.torque_l2, weight=-0.004),
            "joint_velocity": RewardTermCfg(
                func=env_mdp.joint_vel_l2,
                weight=-0.001,
                params={"asset_cfg": mdp.ROBOT_CFG},
            ),
            "joint_limits": RewardTermCfg(
                func=env_mdp.joint_pos_limits,
                weight=-0.02,
                params={"asset_cfg": mdp.ROBOT_CFG},
            ),
            "jump_bonus": RewardTermCfg(
                func=mdp.OneShotJumpBonus,
                weight=5.0,
                params={"target_height": target_height},
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
            "fell": TerminationTermCfg(func=mdp.fell),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="qarm_root",
            distance=1.5,
            elevation=-15.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=128,
            njmax=512,
            contact_sensor_maxmatch=64,
            mujoco=MujocoCfg(
                timestep=0.001,
                gravity=(0.0, 0.0, -9.80665),
                integrator="implicitfast",
                solver="newton",
                iterations=100,
                ls_iterations=20,
            ),
        ),
        decimation=10,
        episode_length_s=4.0,
        auto_reset=True,
        # The original Gymnasium task returns a per-control-step reward. Keep
        # that scale so existing hyperparameters and target bonuses transfer.
        scale_rewards_by_dt=False,
    )
    return cfg


@dataclass(init=False)
class QarmJumpEnvCfg(ManagerBasedRlEnvCfg):
    """Convenience dataclass matching MJLab's task configuration style."""

    def __init__(self, play: bool = False, num_envs: int | None = None):
        self.__dict__.update(
            make_qarm_jump_env_cfg(play=play, num_envs=num_envs).__dict__
        )


__all__ = ["QarmJumpEnvCfg", "make_qarm_jump_env_cfg"]
