from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsl_rl.runners import OnPolicyRunner

from pacman_rl.config import load_yaml, make_env_cfg, make_train_cfg, resolve_path
from pacman_rl.vec_env import RslPacmanVecEnv
from pacman_rl.video import record_policy_video, video_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on the ground-truth Pacman environment.")
    parser.add_argument("--config", default="pacman_rl/configs/pacman_ppo.yaml")
    parser.add_argument("--layout", default=None, help="Override env.layout_file.")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-root", default=None, help="Root directory for runs. Final path is <log-root>/<run-name>.")
    parser.add_argument("--run-name", default=None, help="Run folder name. Defaults to current datetime.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", action="store_true", help="Record policy rollout videos under <run>/train/videos.")
    parser.add_argument("--video-every", type=int, default=0, help="Record every N learning iterations; 0 = final only.")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = load_yaml(args.config)
    env_cfg = make_env_cfg(config, layout_file=args.layout, num_envs=args.num_envs, seed=args.seed)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    train_cfg = make_train_cfg(config, run_name=run_name)
    device = args.device or config.get("device", "cuda")
    iterations = int(args.iterations or config.get("iterations", 1000))
    log_root = resolve_path(args.log_root or config.get("log_root", "logs/pacman_rl"))
    run_dir = log_root / run_name
    train_dir = run_dir / "train"
    checkpoint_dir = train_dir / "checkpoints"
    video_dir = train_dir / "videos"
    for path in (train_dir, checkpoint_dir, video_dir, run_dir / "play"):
        path.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"TensorBoard: tensorboard --logdir {train_dir}")

    env = RslPacmanVecEnv(env_cfg, device=device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=str(train_dir), device=device)
    runner.add_git_repo_to_log(str(ROOT))
    _use_total_progress_eta(runner, total_iterations=iterations)
    _use_checkpoint_dir(runner, checkpoint_dir)
    print(f"Training progress will be reported as i/{iterations} with total-run ETA.")

    try:
        if args.video and args.video_every > 0:
            _record(runner, env_cfg, args, 0, device, video_dir)
            completed = 0
            while completed < iterations:
                chunk = min(args.video_every, iterations - completed)
                runner.learn(num_learning_iterations=chunk)
                completed += chunk
                runner.current_learning_iteration = completed
                _record(runner, env_cfg, args, completed, device, video_dir)
        else:
            runner.learn(num_learning_iterations=iterations)
            if args.video:
                _record(runner, env_cfg, args, runner.current_learning_iteration, device, video_dir)
    finally:
        env.close()


def _use_total_progress_eta(runner: OnPolicyRunner, total_iterations: int) -> None:
    original_log = runner.logger.log

    def log_with_total_eta(*args, **kwargs):
        kwargs["start_it"] = 0
        kwargs["total_it"] = total_iterations
        return original_log(*args, **kwargs)

    runner.logger.log = log_with_total_eta


def _use_checkpoint_dir(runner: OnPolicyRunner, checkpoint_dir: Path) -> None:
    original_save = runner.save

    def save_to_checkpoint_dir(path: str, infos: dict | None = None) -> None:
        target = checkpoint_dir / Path(path).name
        original_save(str(target), infos=infos)

    runner.save = save_to_checkpoint_dir


def _record(
    runner: OnPolicyRunner,
    env_cfg: dict,
    args: argparse.Namespace,
    iteration: int,
    device: str,
    default_video_dir: Path,
) -> None:
    policy = runner.get_inference_policy(device=device)
    out = video_path(default_video_dir, "train_rollout", iteration)
    stats = record_policy_video(
        policy,
        env_cfg,
        out,
        device=device,
        seed=int(env_cfg.get("seed", 0)) + 100000 + int(iteration),
        fps=args.video_fps,
    )
    print(f"Saved video to {out} (return={stats['return']:.2f}, length={stats['length']:.0f})")


if __name__ == "__main__":
    main()
