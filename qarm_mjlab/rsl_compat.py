"""Compatibility helpers for MJLab's supported RSL-RL generations."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict
from typing import Any

import torch


def uses_legacy_rsl_rl() -> bool:
    """Return whether the older ``rsl_rl.rsl_rl`` package is installed."""

    return importlib.util.find_spec("rsl_rl.rsl_rl") is not None


class LegacyRslVecEnvWrapper:
    """Expose an MJLab manager environment through the RSL-RL 2/3 contract."""

    def __init__(self, env: Any):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.action_manager.total_action_dim
        self.device = torch.device(env.device)
        self.max_episode_length = env.max_episode_length
        self.step_dt = env.step_dt
        self.cfg = env.cfg
        self.env.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def _convert(self, obs: dict[str, torch.Tensor], extras: dict):
        actor_obs = obs["actor"]
        critic_obs = obs.get("critic", actor_obs)
        info = dict(extras)
        info["observations"] = {"critic": critic_obs}
        return actor_obs, info

    def reset(self):
        return self._convert(*self.env.reset())

    def get_observations(self):
        if getattr(self.env, "obs_buf", None) is None:
            return self.reset()
        return self._convert(self.env.obs_buf, self.env.extras)

    def step(self, actions: torch.Tensor):
        obs, reward, terminated, truncated, extras = self.env.step(actions)
        info = dict(extras)
        info["time_outs"] = truncated
        actor_obs, info = self._convert(obs, info)
        return actor_obs, reward, terminated | truncated, info

    def close(self):
        self.env.close()

    def seed(self, seed: int = -1):
        return self.env.seed(seed)


def make_vec_env(env: Any, clip_actions: float | None):
    """Wrap *env* for the installed RSL-RL generation."""

    if uses_legacy_rsl_rl():
        return LegacyRslVecEnvWrapper(env)
    from mjlab.rl import RslRlVecEnvWrapper

    return RslRlVecEnvWrapper(env, clip_actions=clip_actions)


def make_runner_cfg(agent_cfg: Any) -> dict[str, Any]:
    """Convert MJLab's dataclass config to the installed runner's schema."""

    data = asdict(agent_cfg)
    if not uses_legacy_rsl_rl():
        return data

    actor = data["actor"]
    critic = data["critic"]
    distribution = actor.get("distribution_cfg") or {}
    algorithm = data["algorithm"]
    legacy_algorithm = {
        key: algorithm[key]
        for key in (
            "class_name",
            "num_learning_epochs",
            "num_mini_batches",
            "learning_rate",
            "schedule",
            "gamma",
            "lam",
            "entropy_coef",
            "desired_kl",
            "max_grad_norm",
            "value_loss_coef",
            "use_clipped_value_loss",
            "clip_param",
        )
        if key in algorithm
    }
    return {
        "seed": data["seed"],
        "num_steps_per_env": data["num_steps_per_env"],
        "max_iterations": data["max_iterations"],
        "save_interval": data["save_interval"],
        "experiment_name": data["experiment_name"],
        "run_name": data["run_name"],
        "logger": data["logger"],
        "wandb_project": data["wandb_project"],
        "resume": data["resume"],
        "load_run": data["load_run"],
        "load_checkpoint": data["load_checkpoint"],
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": list(actor["hidden_dims"]),
            "critic_hidden_dims": list(critic["hidden_dims"]),
            "activation": actor["activation"],
            "init_noise_std": distribution.get("init_std", 1.0),
            "noise_std_type": distribution.get("std_type", "scalar"),
        },
        "algorithm": legacy_algorithm,
    }


def runner_class():
    """Return a runner compatible with the installed RSL-RL package."""

    if uses_legacy_rsl_rl():
        from rsl_rl.runners import OnPolicyRunner

        return OnPolicyRunner
    from mjlab.rl import MjlabOnPolicyRunner

    return MjlabOnPolicyRunner


__all__ = [
    "LegacyRslVecEnvWrapper",
    "make_runner_cfg",
    "make_vec_env",
    "runner_class",
    "uses_legacy_rsl_rl",
]
