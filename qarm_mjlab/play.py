"""Play a trained MJLab Qarm jump policy."""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from .env_cfg import make_qarm_jump_env_cfg
from .rsl_compat import make_runner_cfg, make_vec_env, runner_class


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a trained MJLab Qarm policy")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--viewer", choices=("none", "native", "viser"), default="none")
    parser.add_argument("--speed", type=float, default=1.0)
    return parser


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _actor_observations(value):
    """Extract actor observations from legacy and current MJLab wrappers."""

    if isinstance(value, tuple):
        # Legacy RSL-RL returns ``(actor_obs, extras)`` from get_observations().
        return _actor_observations(value[0])
    if isinstance(value, Mapping) or hasattr(value, "keys"):
        try:
            return value["actor"]
        except (KeyError, TypeError, IndexError):
            pass
    return value


def play(args: argparse.Namespace) -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlOnPolicyRunnerCfg
    from mjlab.utils.torch import configure_torch_backends

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    device = _select_device(args.device)
    configure_torch_backends()
    env_cfg = make_qarm_jump_env_cfg(
        play=True,
        num_envs=args.num_envs,
        **({"model_path": args.model_path} if args.model_path is not None else {}),
    )
    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=0,
        num_steps_per_env=1,
        max_iterations=1,
        experiment_name="qarm_jump",
        logger="tensorboard",
        clip_actions=1.0,
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array" if args.viewer == "none" else None)
    wrapped_env = make_vec_env(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_class()(
        wrapped_env, make_runner_cfg(agent_cfg), log_dir=None, device=device
    )
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=device)

    def viewer_policy(value):
        return policy(_actor_observations(value))

    try:
        if args.viewer in {"native", "viser"}:
            if args.viewer == "native":
                from mjlab.viewer import NativeMujocoViewer

                NativeMujocoViewer(wrapped_env, viewer_policy).run(num_steps=args.steps)
            else:
                from mjlab.viewer import ViserPlayViewer

                ViserPlayViewer(wrapped_env, viewer_policy).run(num_steps=args.steps)
            return

        obs = _actor_observations(wrapped_env.get_observations())
        next_frame = time.perf_counter()
        for _ in range(args.steps):
            action = policy(_actor_observations(obs))
            obs, _, _, _ = wrapped_env.step(action)
            obs = _actor_observations(obs)
            next_frame += env.step_dt / args.speed
            delay = next_frame - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
    finally:
        wrapped_env.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        play(args)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as error:
        print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
