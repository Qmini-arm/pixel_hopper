"""Run a trained Qarm hopping policy in MuJoCo."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from .hopper_env import DEFAULT_JUMP_MODEL_PATH, QArmJumpEnv
from .train_hopper import _require_sb3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained Qarm hopping policy")
    parser.add_argument("checkpoint", type=Path, help="PPO .zip checkpoint")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_JUMP_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--target-height", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="visual playback speed; 1.0 is real time, 0.25 is quarter speed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if not math.isfinite(args.speed) or args.speed <= 0.0:
        raise SystemExit("--speed must be finite and positive")
    if not args.no_render and sys.platform == "darwin":
        import mujoco.viewer

        if mujoco.viewer._MJPYTHON is None:
            raise SystemExit(
                "macOS viewer must run through mjpython; use "
                "'.venv/bin/mjpython -m qarm_sim.play_hopper ...'"
            )
    PPO, _, _, _, _ = _require_sb3()
    env = QArmJumpEnv(
        model_path=args.model_path,
        frame_skip=args.frame_skip,
        episode_seconds=args.episode_seconds,
        target_height_m=args.target_height,
        render_mode=None if args.no_render else "human",
    )
    policy = PPO.load(str(args.checkpoint), device="auto")
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            info = {}
            next_frame = time.perf_counter()
            while not (terminated or truncated):
                action, _ = policy.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                if not args.no_render:
                    env.render()
                    next_frame += env.control_timestep / args.speed
                    delay = next_frame - time.perf_counter()
                    if delay > 0.0:
                        time.sleep(delay)
                    elif delay < -env.control_timestep:
                        # Do not accumulate an ever-growing delay if rendering
                        # or the host OS pauses the process for a while.
                        next_frame = time.perf_counter()
            print(
                f"episode={episode + 1} success={info.get('success', False)} "
                f"peak_height_delta_m={info.get('peak_height_delta_m', 0.0):.3f} "
                f"fell={info.get('fell', False)}"
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
