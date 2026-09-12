"""Train the Qarm jumping policy with Stable-Baselines3 PPO.

Install the optional RL dependencies first::

    .venv/bin/python -m pip install -r requirements-rl.txt

Then run a short smoke training job with::

    .venv/bin/python -m qarm_sim.train_hopper --timesteps 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .hopper_env import DEFAULT_JUMP_MODEL_PATH, QArmJumpEnv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Qarm hopping policy with PPO")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_JUMP_MODEL_PATH)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--target-height", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="Torch device passed to PPO")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "qarm_hopper",
        help="directory for checkpoints, logs, and the final policy",
    )
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--checkpoint", type=Path, default=None, help="resume a .zip PPO policy")
    return parser


def _require_sb3() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PPO training requires Stable-Baselines3 and PyTorch; install "
            "requirements-rl.txt first"
        ) from error
    return PPO, CheckpointCallback, EvalCallback, make_vec_env, Monitor


def train(args: argparse.Namespace) -> Path:
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if args.eval_freq <= 0:
        raise ValueError("--eval-freq must be positive")

    PPO, CheckpointCallback, EvalCallback, make_vec_env, Monitor = _require_sb3()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    best = output / "best"
    logs = output / "eval"
    for directory in (checkpoints, best, logs):
        directory.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "model_path": args.model_path,
        "frame_skip": args.frame_skip,
        "episode_seconds": args.episode_seconds,
        "target_height_m": args.target_height,
    }
    train_env = make_vec_env(
        QArmJumpEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
    )
    eval_base_env = QArmJumpEnv(**env_kwargs)
    eval_env = Monitor(eval_base_env)
    with (output / "environment.json").open("w", encoding="utf-8") as file:
        json.dump(eval_base_env.model_info(), file, indent=2)

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.eval_freq // args.n_envs),
            save_path=str(checkpoints),
            name_prefix="qarm_hopper",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(best),
            log_path=str(logs),
            eval_freq=max(1, args.eval_freq // args.n_envs),
            deterministic=True,
            render=False,
        ),
    ]

    try:
        if args.checkpoint is None:
            policy = PPO(
                "MlpPolicy",
                train_env,
                n_steps=1024,
                batch_size=256,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                clip_range=0.2,
                verbose=1,
                seed=args.seed,
                device=args.device,
                tensorboard_log=str(output / "tensorboard"),
            )
        else:
            policy = PPO.load(str(args.checkpoint), env=train_env, device=args.device)
        policy.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=args.checkpoint is None,
        )
        final_path = output / "qarm_hopper_final"
        policy.save(str(final_path))
    finally:
        train_env.close()
        eval_env.close()
    return final_path.with_suffix(".zip")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        final_path = train(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"saved PPO policy to {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
