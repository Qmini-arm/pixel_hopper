"""MJLab entity configuration for the free-base Qarm MJCF."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "description"
    / "output_mjcf"
    / "robot_freejoint.xml"
)
JUMP_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4")
QARM_INITIAL_JOINT_QPOS = (0.0, 1.2, 1.2, 0.0)
QARM_INITIAL_ROOT_HEIGHT = 0.0232
TORQUE_LIMIT_NM = 23.7


def _spec_from_mjcf(model_path: Path) -> mujoco.MjSpec:
    if not model_path.is_file():
        raise FileNotFoundError(f"Qarm MJCF not found: {model_path}")
    spec = mujoco.MjSpec.from_file(str(model_path))
    # MJLab owns scene-level simulation options. Reset the entity's non-default
    # values so MjSpec.attach() does not report fields that cannot be propagated;
    # SimulationCfg restores the intended 1 ms / implicitfast / gravity values.
    spec.option.timestep = 0.002
    spec.option.gravity = (0.0, 0.0, -9.81)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    # The standalone XML includes its own floor. MJLab adds the canonical
    # terrain entity, so retaining both would create coincident collision
    # planes and distort support forces.
    for geom in spec.worldbody.find_all("geom"):
        if geom.name == "floor":
            spec.delete(geom)
            break
    # EntityCfg creates the scene's canonical ``init_state`` keyframe from
    # ``InitialStateCfg``. Keeping the standalone XML keyframe would make the
    # scene carry two keyframes and MJLab would warn while dropping one.
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def get_qarm_robot_cfg(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    initial_joint_qpos: tuple[float, float, float, float] = QARM_INITIAL_JOINT_QPOS,
) -> EntityCfg:
    """Build an MJLab entity around the existing free-joint MJCF.

    The XML already contains four torque motors. ``XmlActuatorCfg`` wraps those
    motors so MJLab can write effort targets without changing the calibrated
    actuator limits or the model's direct-torque semantics.
    """

    path = Path(model_path).expanduser().resolve()
    if len(initial_joint_qpos) != len(JUMP_JOINT_NAMES):
        raise ValueError("initial_joint_qpos must contain four joint angles")

    return EntityCfg(
        spec_fn=lambda: _spec_from_mjcf(path),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, QARM_INITIAL_ROOT_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            joint_pos=dict(zip(JUMP_JOINT_NAMES, initial_joint_qpos, strict=True)),
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=JUMP_JOINT_NAMES),),
            soft_joint_pos_limit_factor=0.98,
        ),
        sort_actuators=True,
    )


__all__ = [
    "DEFAULT_MODEL_PATH",
    "JUMP_JOINT_NAMES",
    "QARM_INITIAL_JOINT_QPOS",
    "QARM_INITIAL_ROOT_HEIGHT",
    "TORQUE_LIMIT_NM",
    "get_qarm_robot_cfg",
]
