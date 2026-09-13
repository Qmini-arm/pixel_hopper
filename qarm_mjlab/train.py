"""Train the free-base Qarm jump policy with MJLab and RSL-RL."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .env_cfg import make_qarm_jump_env_cfg
from .rsl_compat import make_runner_cfg, make_vec_env, runner_class


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the free-base Qarm jump task with MJLab/RSL-RL"
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-steps-per-env", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:N]")
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "qarm_mjlab")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="accepted for parity with MJLab examples; training is headless",
    )
    return parser


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def train(args: argparse.Namespace) -> Path:
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.num_steps_per_env <= 0:
        raise ValueError("--num-steps-per-env must be positive")
    if args.save_interval <= 0:
        raise ValueError("--save-interval must be positive")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlOnPolicyRunnerCfg
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    device = _select_device(args.device)
    configure_torch_backends()
    env_cfg_kwargs = {"num_envs": args.num_envs}
    if args.model_path is not None:
        env_cfg_kwargs["model_path"] = args.model_path
    env_cfg = make_qarm_jump_env_cfg(**env_cfg_kwargs)

    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=args.seed,
        num_steps_per_env=args.num_steps_per_env,
        max_iterations=math.ceil(args.timesteps / (args.num_envs * args.num_steps_per_env)),
        save_interval=args.save_interval,
        experiment_name="qarm_jump",
        run_name=args.run_name,
        logger="tensorboard",
        wandb_project="qarm_mjlab",
        clip_actions=1.0,
    )
    env_cfg.seed = args.seed

    print("[INFO] Task: qarm_jump")
    print("[INFO] Backend: MJLab/MuJoCo-Warp")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Number of environments: {env_cfg.scene.num_envs}")
    print(f"[INFO] Learning iterations: {agent_cfg.max_iterations}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    wrapped_env = make_vec_env(env, clip_actions=agent_cfg.clip_actions)

    run_root = args.output.expanduser().resolve() / "logs" / "rsl_rl" / "qarm_jump"
    run_name = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        run_name += f"_{args.run_name}"
    log_dir = run_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    train_cfg = make_runner_cfg(agent_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", train_cfg)

    runner = runner_class()(
        wrapped_env,
        train_cfg,
        log_dir=str(log_dir),
        device=device,
    )
    try:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint.expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
            runner.load(str(checkpoint))
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        wrapped_env.close()
    return log_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        log_dir = train(args)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"saved MJLab run to {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
