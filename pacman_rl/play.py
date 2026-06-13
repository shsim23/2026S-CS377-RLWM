from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pacman_env import PacmanEnv, RewardConfig
from rsl_rl.runners import OnPolicyRunner

from pacman_rl.config import load_yaml, make_env_cfg, make_train_cfg
from pacman_rl.vec_env import RslPacmanVecEnv
from pacman_rl.video import save_video, video_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Play/evaluate a PPO Pacman policy checkpoint.")
    parser.add_argument("--config", default="pacman_rl/configs/pacman_ppo.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", action="store_true", help="Record episode videos under <run>/play.")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = load_yaml(args.config)
    env_cfg = make_env_cfg(config, layout_file=args.layout, num_envs=1, seed=args.seed)
    train_cfg = make_train_cfg(config)
    device = args.device or config.get("device", "cuda")

    play_dir = _infer_play_dir(Path(args.checkpoint))
    play_dir.mkdir(parents=True, exist_ok=True)
    print(f"Play directory: {play_dir}")

    runner_env = RslPacmanVecEnv(env_cfg, device=device)
    runner = OnPolicyRunner(runner_env, train_cfg, log_dir=None, device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)
    runner_env.close()

    returns = []
    lengths = []
    pellets_remaining = []
    wins = 0
    for ep in range(args.episodes):
        stats = play_episode(policy, env_cfg, device, args.seed + ep, args, ep, play_dir)
        returns.append(stats["return"])
        lengths.append(stats["length"])
        pellets_remaining.append(stats["pellets_remaining"])
        wins += int(stats["won"])
        print(
            f"episode={ep} return={stats['return']:.2f} "
            f"length={stats['length']:.0f} won={bool(stats['won'])} "
            f"pellets_remaining={stats['pellets_remaining']:.0f}"
        )

    print(
        f"mean_return={sum(returns) / max(len(returns), 1):.2f} "
        f"mean_length={sum(lengths) / max(len(lengths), 1):.1f} "
        f"win_rate={wins / max(len(returns), 1):.3f} "
        f"mean_pellets_remaining={sum(pellets_remaining) / max(len(pellets_remaining), 1):.1f}"
    )


def play_episode(
    policy,
    env_cfg: dict,
    device: str,
    seed: int,
    args: argparse.Namespace,
    episode_idx: int,
    play_dir: Path,
) -> dict[str, float]:
    render_mode = "rgb_array" if args.video else (None if args.headless else "human")
    env = make_env(env_cfg, render_mode)
    frames = []
    total_reward = 0.0
    steps = 0
    won = False
    pellets_remaining = 0.0
    obs, _ = env.reset(seed=seed)
    try:
        max_steps = int(env_cfg.get("max_steps", env_cfg.get("episode", {}).get("max_steps", 500)))
        for _ in range(max_steps):
            frame = env.render()
            if args.video and frame is not None:
                frames.append(frame)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            td = TensorDict({"policy": obs_tensor}, batch_size=[1], device=device)
            with torch.inference_mode():
                action = policy(td, stochastic_output=False)
            obs, reward, terminated, truncated, info = env.step(int(action.view(-1)[0].item()))
            total_reward += float(reward)
            steps += 1
            won = bool(info.get("event", {}).get("won", False))
            pellets_remaining = float(info.get("pellets_remaining", pellets_remaining))
            if terminated or truncated:
                frame = env.render()
                if args.video and frame is not None:
                    frames.append(frame)
                break
    finally:
        env.close()

    if args.video:
        out = video_path(play_dir, "play", episode_idx)
        save_video(frames, out, fps=args.video_fps)
        print(f"Saved video to {out}")

    return {
        "return": total_reward,
        "length": float(steps),
        "won": float(won),
        "pellets_remaining": pellets_remaining,
    }


def _infer_play_dir(checkpoint: Path) -> Path:
    resolved = checkpoint.resolve()
    parts = resolved.parts
    if len(parts) >= 4 and parts[-3:] and resolved.parent.name == "checkpoints" and resolved.parent.parent.name == "train":
        return resolved.parent.parent.parent / "play"
    return resolved.parent / "play"


def make_env(env_cfg: dict, render_mode: str | None) -> PacmanEnv:
    reward_cfg = RewardConfig(**env_cfg.get("reward", {}))
    ghost_cfg = env_cfg.get("ghost", {})
    power_cfg = env_cfg.get("power_pellet", {})
    episode_cfg = env_cfg.get("episode", {})
    return PacmanEnv(
        layout_path=env_cfg["layout_file"],
        num_ghosts=int(ghost_cfg.get("num_ghosts", env_cfg.get("num_ghosts", 1))),
        ghost_epsilon=float(ghost_cfg.get("epsilon", env_cfg.get("ghost_epsilon", 0.2))),
        ghost_policy=str(ghost_cfg.get("policy", env_cfg.get("ghost_policy", "chase_stochastic"))),
        ghost_speed_ratio=float(ghost_cfg.get("speed_ratio", env_cfg.get("ghost_speed_ratio", 1.0))),
        power_pellet_enabled=bool(power_cfg.get("enabled", env_cfg.get("power_pellet_enabled", False))),
        frightened_duration=int(power_cfg.get("frightened_duration", env_cfg.get("frightened_duration", 30))),
        max_steps=int(episode_cfg.get("max_steps", env_cfg.get("max_steps", 500))),
        reward_config=reward_cfg,
        render_mode=render_mode,
        randomize_spawn=bool(env_cfg.get("randomize_spawn", True)),
        min_spawn_dist=int(env_cfg.get("min_spawn_dist", 3)),
    )


if __name__ == "__main__":
    main()
