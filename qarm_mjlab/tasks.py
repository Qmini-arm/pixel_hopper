"""Optional registration with MJLab's native task registry."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import make_qarm_jump_env_cfg

TASK_ID = "Qarm-FreeBase-Jump"


def register() -> str:
    """Register the task once and return its MJLab task id."""

    from mjlab.tasks.registry import list_tasks

    if TASK_ID not in list_tasks():
        register_mjlab_task(
            TASK_ID,
            env_cfg=make_qarm_jump_env_cfg(play=False),
            play_env_cfg=make_qarm_jump_env_cfg(play=True),
            rl_cfg=RslRlOnPolicyRunnerCfg(
                experiment_name="qarm_jump",
                logger="tensorboard",
                clip_actions=1.0,
            ),
            runner_cls=None,
        )
    return TASK_ID


__all__ = ["TASK_ID", "ManagerBasedRlEnv", "register"]
