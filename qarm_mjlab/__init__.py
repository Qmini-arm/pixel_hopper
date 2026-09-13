"""MJLab task for training the Qarm as a free-base jumper.

The package intentionally keeps the original Gymnasium environment intact.  The
MJLab task uses the same MJCF, action scale, reset posture, episode horizon, and
success criterion, but evaluates all environments in parallel through MuJoCo
Warp.
"""

from .asset import DEFAULT_MODEL_PATH, JUMP_JOINT_NAMES, QARM_INITIAL_JOINT_QPOS
from .env_cfg import QarmJumpEnvCfg, make_qarm_jump_env_cfg

__all__ = [
    "DEFAULT_MODEL_PATH",
    "JUMP_JOINT_NAMES",
    "QARM_INITIAL_JOINT_QPOS",
    "QarmJumpEnvCfg",
    "make_qarm_jump_env_cfg",
]
