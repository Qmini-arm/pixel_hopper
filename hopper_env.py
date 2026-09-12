"""Gymnasium environment for learning a free-base Qarm jump.

The fixed-base :class:`qarm_sim.env.QArmMujocoEnv` is intentionally kept
separate from this environment.  A jumping policy needs the free joint in
``robot_freejoint.xml`` and should command the four joint torques directly;
there is no M8010 position controller or gravity feed-forward in the loop.

The environment exposes a small, normalized observation and action interface
that is suitable for PPO or another continuous-control algorithm.  It is
deliberately dependency-light: Gymnasium is required only when this module is
used, while Stable-Baselines3 remains an optional training dependency.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as error:  # pragma: no cover - exercised by import users
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _gymnasium_import_error: ModuleNotFoundError | None = error
else:
    _gymnasium_import_error = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUMP_MODEL_PATH = ROOT / "description" / "output_mjcf" / "robot_freejoint.xml"
JUMP_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4")
FREE_JOINT_NAME = "root_free"

_GymEnvBase = gym.Env if gym is not None else object


class QArmJumpEnv(_GymEnvBase):
    """Free-base MuJoCo task in which the Qarm learns to jump vertically.

    Actions are normalized torques in ``[-1, 1]`` for the four arm joints.
    The observation contains root translation/velocity, an upright direction
    vector, arm joint state, floor support, and the previous action.  A jump
    is rewarded through height progress and a one-time target-height bonus;
    tilt, horizontal drift, torque, and action changes are penalized.

    The default reset pose is a compact, supported posture.  This matters for
    the current CAD model: the all-zero arm pose is tall and quickly tips over
    when the root is released.
    """

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | Path = DEFAULT_JUMP_MODEL_PATH,
        *,
        frame_skip: int = 10,
        episode_seconds: float = 4.0,
        target_height_m: float = 0.20,
        initial_joint_qpos: tuple[float, float, float, float] = (0.0, 1.2, 1.2, 0.0),
        reset_noise_scale: float = 0.02,
        render_mode: str | None = None,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError(
                "QArmJumpEnv requires Gymnasium; install requirements-rl.txt "
                "or run '.venv/bin/python -m pip install gymnasium'"
            ) from _gymnasium_import_error
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        if isinstance(frame_skip, bool) or not isinstance(frame_skip, int) or frame_skip <= 0:
            raise ValueError("frame_skip must be a positive integer")
        for name, value in (("episode_seconds", episode_seconds), ("target_height_m", target_height_m)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(reset_noise_scale) or reset_noise_scale < 0.0:
            raise ValueError("reset_noise_scale must be finite and non-negative")

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")
        try:
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        except ValueError as error:
            raise ValueError(f"cannot compile MuJoCo model {self.model_path}: {error}") from error
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.episode_seconds = float(episode_seconds)
        self.target_height_m = float(target_height_m)
        self.reset_noise_scale = float(reset_noise_scale)
        self.render_mode = render_mode
        self.max_episode_steps = max(
            1, math.ceil(self.episode_seconds / (self.model.opt.timestep * self.frame_skip))
        )

        self.root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "qarm_root"
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.free_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, FREE_JOINT_NAME
        )
        if self.root_body_id < 0 or self.floor_geom_id < 0 or self.free_joint_id < 0:
            raise ValueError(
                "jump model must define bodies/geoms/joints 'qarm_root', 'floor', "
                "and 'root_free'"
            )
        if self.model.jnt_type[self.free_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("root_free must be a MuJoCo free joint")
        if self.model.nq != 11 or self.model.nv != 10:
            raise ValueError(
                "robot_freejoint.xml must expose one free joint and four hinges "
                f"(expected nq=11, nv=10; got nq={self.model.nq}, nv={self.model.nv})"
            )

        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in JUMP_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0):
            missing = [
                name for name, joint_id in zip(JUMP_JOINT_NAMES, self.joint_ids) if joint_id < 0
            ]
            raise ValueError(f"jump model is missing arm joints: {missing}")
        if np.any(self.model.jnt_type[self.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("jump arm joints must all be hinge joints")

        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids].copy()
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids].copy()
        self.actuator_ids = self._find_actuators()
        self.torque_scale = self.model.actuator_ctrlrange[self.actuator_ids, 1].astype(
            np.float64, copy=True
        )
        if np.any(self.torque_scale <= 0.0):
            raise ValueError("jump actuators must have positive upper ctrlrange values")

        default_q = np.asarray(initial_joint_qpos, dtype=np.float64)
        if default_q.shape != (len(JUMP_JOINT_NAMES),) or not np.all(np.isfinite(default_q)):
            raise ValueError("initial_joint_qpos must contain four finite joint angles")
        ranges = self.model.jnt_range[self.joint_ids]
        if np.any(default_q < ranges[:, 0]) or np.any(default_q > ranges[:, 1]):
            raise ValueError("initial_joint_qpos violates a MuJoCo joint range")
        self.initial_joint_qpos = default_q
        self.joint_ranges = ranges.copy()
        self.initial_root_position = self.model.qpos0[:3].astype(np.float64, copy=True)
        self.initial_root_height = float(self.initial_root_position[2])

        # All components are bounded or smoothly saturated to keep PPO's input
        # scale stable while retaining raw values in the step info dictionary.
        self.observation_space = spaces.Box(
            low=np.asarray(
                [-2.0, -2.0, -2.0]
                + [-1.0] * 3
                + [-1.0] * 3
                + [-1.0] * 3
                + [-2.0] * 4
                + [-1.0] * 4
                + [0.0]
                + [-1.0] * 4,
                dtype=np.float32,
            ),
            high=np.asarray(
                [2.0, 2.0, 2.0]
                + [1.0] * 3
                + [1.0] * 3
                + [1.0] * 3
                + [2.0] * 4
                + [1.0] * 4
                + [1.0]
                + [1.0] * 4,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(JUMP_JOINT_NAMES),),
            dtype=np.float32,
        )

        self._last_action = np.zeros(len(JUMP_JOINT_NAMES), dtype=np.float64)
        self._elapsed_steps = 0
        self._peak_height = self.initial_root_height
        self._has_left_ground = False
        self._success_awarded = False
        self._closed = False
        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None

    def _find_actuators(self) -> np.ndarray:
        if self.model.nu != len(JUMP_JOINT_NAMES):
            raise ValueError(f"expected four jump actuators, found {self.model.nu}")
        result = np.full(len(JUMP_JOINT_NAMES), -1, dtype=np.int32)
        for joint_index, joint_id in enumerate(self.joint_ids):
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)
            if len(matches) != 1:
                raise ValueError(
                    f"{JUMP_JOINT_NAMES[joint_index]} must have exactly one direct actuator"
                )
            result[joint_index] = int(matches[0])
        if not np.all(self.model.actuator_ctrllimited[result]):
            raise ValueError("all jump actuators must define ctrlrange")
        return result

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def control_timestep(self) -> float:
        return self.timestep * self.frame_skip

    def _root_up(self) -> np.ndarray:
        return np.asarray(self.data.xmat[self.root_body_id], dtype=np.float64).reshape(3, 3)[:, 2]

    def _supported(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if int(contact.geom1) == self.floor_geom_id or int(contact.geom2) == self.floor_geom_id:
                return True
        return False

    def _observation(self) -> np.ndarray:
        root_position = self.data.qpos[:3]
        root_velocity = self.data.qvel[:3]
        root_angular_velocity = self.data.qvel[3:6]
        hinge_position = self.data.qpos[self.qpos_addresses]
        hinge_velocity = self.data.qvel[self.dof_addresses]
        half_range = np.maximum(
            np.maximum(np.abs(self.joint_ranges[:, 0]), np.abs(self.joint_ranges[:, 1])),
            1e-6,
        )
        observation = np.concatenate(
            (
                np.clip(root_position[:2] - self.initial_root_position[:2], -2.0, 2.0),
                np.asarray(
                    [np.clip((root_position[2] - self.initial_root_height) / 0.5, -2.0, 2.0)]
                ),
                np.tanh(root_velocity / 5.0),
                self._root_up(),
                np.tanh(root_angular_velocity / 10.0),
                np.clip(hinge_position / half_range, -2.0, 2.0),
                np.tanh(hinge_velocity / 10.0),
                np.asarray([1.0 if self._supported() else 0.0]),
                self._last_action,
            )
        )
        return observation.astype(np.float32, copy=False)

    def _info(self, *, action_clipped: bool = False, fell: bool = False) -> dict[str, Any]:
        root_position = self.data.qpos[:3].copy()
        root_velocity = self.data.qvel[:3].copy()
        up = self._root_up()
        return {
            "time_s": float(self.data.time),
            "height_m": float(root_position[2]),
            "height_delta_m": float(root_position[2] - self.initial_root_height),
            "peak_height_delta_m": float(self._peak_height - self.initial_root_height),
            "vertical_velocity_m_s": float(root_velocity[2]),
            "horizontal_displacement_m": float(np.linalg.norm(root_position[:2])),
            "root_up_z": float(up[2]),
            "supported": self._supported(),
            "has_left_ground": self._has_left_ground,
            "success": self._success_awarded,
            "fell": fell,
            "action_clipped": action_clipped,
            "torques_nm": self.data.ctrl[self.actuator_ids].astype(float).tolist(),
            "joint_position_rad": self.data.qpos[self.qpos_addresses].astype(float).tolist(),
            "joint_velocity_rad_s": self.data.qvel[self.dof_addresses].astype(float).tolist(),
            "episode_step": self._elapsed_steps,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if self._closed:
            raise RuntimeError("QArmJumpEnv is closed")
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:3] = self.initial_root_position
        self.data.qpos[3:7] = np.asarray((1.0, 0.0, 0.0, 0.0))
        noise = self.np_random.normal(0.0, self.reset_noise_scale, size=4)
        self.data.qpos[self.qpos_addresses] = np.clip(
            self.initial_joint_qpos + noise,
            self.joint_ranges[:, 0] + 1e-5,
            self.joint_ranges[:, 1] - 1e-5,
        )
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_action.fill(0.0)
        self._elapsed_steps = 0
        self._peak_height = float(self.data.qpos[2])
        self._has_left_ground = not self._supported()
        self._success_awarded = False
        observation = self._observation()
        return observation, self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("QArmJumpEnv is closed")
        if self._elapsed_steps >= self.max_episode_steps:
            raise RuntimeError("episode is over; call reset() before step()")
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != self.action_space.shape or not np.all(np.isfinite(action_array)):
            raise ValueError(f"action must be finite with shape {self.action_space.shape}")
        clipped_action = np.clip(action_array, -1.0, 1.0)
        action_was_clipped = not np.array_equal(action_array, clipped_action)
        previous_height = float(self.data.qpos[2])
        previous_action = self._last_action.copy()
        self.data.ctrl[self.actuator_ids] = clipped_action * self.torque_scale
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._elapsed_steps += 1

        root_position = self.data.qpos[:3]
        root_velocity = self.data.qvel[:3]
        up_z = float(self._root_up()[2])
        supported = self._supported()
        self._has_left_ground = self._has_left_ground or not supported
        self._peak_height = max(self._peak_height, float(root_position[2]))
        height_delta = float(root_position[2] - self.initial_root_height)
        height_progress = float(root_position[2] - previous_height)

        # Dense terms make the early policy learn a crouch/extension sequence;
        # the target bonus then selects genuine, upright jumps over arm flailing.
        reward = 0.02
        reward += 2.0 * np.clip(height_delta / self.target_height_m, -1.0, 2.0)
        reward += 1.5 * np.clip(height_progress / 0.03, -1.0, 1.0)
        reward += 0.35 * max(0.0, up_z)
        if supported:
            reward += 0.25 * np.clip(float(root_velocity[2]), -1.0, 1.0)
        reward -= 0.08 * float(np.linalg.norm(root_position[:2]))
        reward -= 0.05 * float(np.linalg.norm(self.data.qvel[3:6]))
        reward -= 0.004 * float(np.mean(np.square(clipped_action)))
        reward -= 0.002 * float(np.mean(np.square(clipped_action - previous_action)))

        target_reached = (
            self._has_left_ground
            and height_delta >= self.target_height_m
            and not supported
            and up_z > 0.45
        )
        if target_reached and not self._success_awarded:
            reward += 5.0
            self._success_awarded = True

        finite_state = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        fell = (
            not finite_state
            or float(root_position[2]) < self.initial_root_height - 0.06
            or up_z < 0.15
            or float(np.linalg.norm(root_position[:2])) > 1.0
        )
        if fell:
            reward -= 5.0
        terminated = fell
        truncated = self._elapsed_steps >= self.max_episode_steps and not terminated
        self._last_action = clipped_action
        observation = self._observation()
        info = self._info(action_clipped=action_was_clipped, fell=fell)
        return observation, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="overview")
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            if self._viewer.is_running():
                self._viewer.sync()
        return None

    def close(self) -> None:
        if self._closed:
            return
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._closed = True

    def model_info(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "mujoco_version": mujoco.__version__,
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "timestep_s": self.timestep,
            "control_timestep_s": self.control_timestep,
            "max_episode_steps": self.max_episode_steps,
            "joint_names": list(JUMP_JOINT_NAMES),
            "joint_range_rad": self.joint_ranges.tolist(),
            "actuator_ctrlrange_nm": self.model.actuator_ctrlrange[self.actuator_ids].tolist(),
            "initial_joint_qpos_rad": self.initial_joint_qpos.tolist(),
            "initial_root_height_m": self.initial_root_height,
            "target_height_m": self.target_height_m,
        }

    def __enter__(self) -> "QArmJumpEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "DEFAULT_JUMP_MODEL_PATH",
    "FREE_JOINT_NAME",
    "JUMP_JOINT_NAMES",
    "QArmJumpEnv",
]
