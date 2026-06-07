"""Play a PPO checkpoint inside the Dreamer WM imagined environment and save videos.

This is intentionally separate from pacman_rl/play.py, which evaluates policies in
real PacmanEnv. Here, dynamics come from RslPacmanDreamerVecEnv and frames are
rendered from the WM-decoded 901-d state vectors.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsl_rl.runners import OnPolicyRunner

from pacman_env import GameState
from pacman_env.constants import MAX_GHOSTS, MAX_GRID_H, MAX_GRID_W
from pacman_rl.config import load_yaml, make_env_cfg, make_train_cfg, make_world_model_cfg, resolve_path
from pacman_rl.play import make_env
from pacman_rl.video import save_video, video_path
from pacman_rl.wm_env import RslPacmanDreamerVecEnv

PAC_SLICE = slice(0, 2)
GHOST_SLICE = slice(2, 18)
FOOD_SLICE = slice(18, 459)
WALL_SLICE = slice(459, 900)
POWER_SLICE = slice(900, 901)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play/evaluate a PPO checkpoint inside the Dreamer WM dreamed env.")
    parser.add_argument("--config", default="pacman_rl/configs/pacman_ppo.yaml")
    parser.add_argument("--checkpoint", required=True, help="PPO checkpoint to play.")
    parser.add_argument("--layout", default=None, help="Override env.layout_file.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None, help="Directory for dreamed videos; default is <run>/play_wm.")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None, help="Override rollout horizon for playback only.")
    parser.add_argument("--deterministic-wm", action="store_true", help="Use Dreamer latent modes instead of samples.")
    parser.add_argument("--no-uncertainty", action="store_true", help="Disable uncertainty truncation during dreamed playback.")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = load_yaml(args.config)
    env_cfg = make_env_cfg(config, layout_file=args.layout, num_envs=1, seed=args.seed)
    if args.max_steps is not None:
        env_cfg["max_steps"] = int(args.max_steps)
    wm_cfg = make_world_model_cfg(config)
    wm_cfg["use_wm"] = True
    if args.deterministic_wm:
        wm_cfg["deterministic_latent"] = True
    if args.no_uncertainty:
        wm_cfg["use_uncertainty_aware_methods"] = False
    else:
        wm_cfg.setdefault("use_uncertainty_aware_methods", True)

    train_cfg = make_train_cfg(config)
    device = args.device or config.get("device", "cuda")
    out_dir = resolve_path(args.out_dir) if args.out_dir else _infer_play_wm_dir(Path(args.checkpoint))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dream playback directory: {out_dir}")

    env = RslPacmanDreamerVecEnv(env_cfg, wm_cfg, device=device)
    try:
        runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
        runner.load(args.checkpoint, map_location=device)
        policy = runner.get_inference_policy(device=device)

        returns = []
        lengths = []
        for ep in range(args.episodes):
            stats = play_dream_episode(policy, env, env_cfg, device, args.seed + ep, ep, out_dir, args.video_fps)
            returns.append(stats["return"])
            lengths.append(stats["length"])
            print(
                f"episode={ep} dreamed_return={stats['return']:.2f} "
                f"length={stats['length']:.0f} final_pellets={stats['pellets_remaining']:.0f} "
                f"mean_confidence={stats['mean_confidence']:.3f}"
            )
    finally:
        env.close()

    if returns:
        print(
            f"mean_dreamed_return={sum(returns) / len(returns):.2f} "
            f"mean_length={sum(lengths) / len(lengths):.1f}"
        )


@torch.no_grad()
def play_dream_episode(policy, env: RslPacmanDreamerVecEnv, env_cfg: dict, device: str, seed: int,
                       episode_idx: int, out_dir: Path, fps: int) -> dict[str, float]:
    env._reset_one(0, seed=seed)
    render_env = make_env(env_cfg, render_mode="rgb_array")
    render_env.reset(seed=seed)
    frames = []
    states = []
    total_reward = 0.0
    confidences = []
    max_steps = int(env_cfg.get("max_steps", env_cfg.get("episode", {}).get("max_steps", 500)))

    try:
        for step in range(max_steps):
            obs = env.get_observations()
            state = obs["policy"][0].detach().cpu().numpy()
            frames.append(render_state_with_pacman_renderer(render_env, state, step))
            states.append(state)
            action = policy(obs, stochastic_output=False)
            _, reward, done, extras = env.step(action)
            total_reward += float(reward[0].item())
            if "wm_confidence" in extras:
                confidences.append(float(extras["wm_confidence"][0].item()))
            if bool(done[0].item()):
                final_state = env.get_observations()["policy"][0].detach().cpu().numpy()
                frames.append(render_state_with_pacman_renderer(render_env, final_state, step + 1))
                states.append(final_state)
                break
    finally:
        render_env.close()

    out = video_path(out_dir, "wm_dream_play", episode_idx)
    save_video(frames, out, fps=fps)
    print(f"Saved dreamed video to {out}")
    final = states[-1]
    pellets_remaining = float((final[FOOD_SLICE] > 0.5).sum())
    return {
        "return": total_reward,
        "length": float(len(frames) - 1),
        "pellets_remaining": pellets_remaining,
        "mean_confidence": float(np.mean(confidences)) if confidences else 1.0,
    }


def render_state_with_pacman_renderer(render_env, state: np.ndarray, step_count: int) -> np.ndarray:
    render_env.game_state = state_vector_to_game_state(state, render_env, step_count)
    frame = render_env.render()
    if frame is None:
        raise RuntimeError("PacmanEnv renderer returned no frame; render_mode must be 'rgb_array'.")
    return frame


def state_vector_to_game_state(state: np.ndarray, render_env, step_count: int) -> GameState:
    layout = render_env.layout
    pacman_pos = _entity_pos(state[0], state[1], layout)

    ghost_slots = state[GHOST_SLICE].reshape(MAX_GHOSTS, 4)
    ghost_positions = []
    ghost_alive = []
    for i in range(render_env.num_ghosts):
        gx, gy, alive, valid = ghost_slots[i]
        ghost_positions.append(_entity_pos(gx, gy, layout))
        ghost_alive.append(bool(alive > 0.5 and valid > 0.5))

    food_full = state[FOOD_SLICE].reshape(MAX_GRID_H, MAX_GRID_W) > 0.5
    food_mask = food_full[: layout.height, : layout.width].copy()
    food_mask[layout.walls] = False

    power_mode_timer = int(round(float(np.clip(state[POWER_SLICE][0], 0.0, 1.0)) * 30.0))
    return GameState(
        pacman_pos=pacman_pos,
        ghost_positions=ghost_positions,
        ghost_alive=ghost_alive,
        food_mask=food_mask,
        power_mode_timer=power_mode_timer,
        step_count=int(step_count),
        done=False,
    )


def _entity_pos(x_norm: float, y_norm: float, layout) -> tuple[int, int]:
    x = int(round((float(x_norm) + 1.0) * (MAX_GRID_W - 1) / 2.0))
    y = int(round((float(y_norm) + 1.0) * (MAX_GRID_H - 1) / 2.0))
    x = int(np.clip(x, 0, layout.width - 1))
    y = int(np.clip(y, 0, layout.height - 1))
    if layout.walls[y, x]:
        return _nearest_walkable(x, y, layout)
    return x, y


def _nearest_walkable(x: int, y: int, layout) -> tuple[int, int]:
    walk_y, walk_x = np.where(~layout.walls)
    if len(walk_x) == 0:
        return x, y
    dist = np.abs(walk_x - x) + np.abs(walk_y - y)
    idx = int(dist.argmin())
    return int(walk_x[idx]), int(walk_y[idx])


def _infer_play_wm_dir(checkpoint: Path) -> Path:
    resolved = checkpoint.resolve()
    if resolved.parent.name == "checkpoints" and resolved.parent.parent.name == "train":
        return resolved.parent.parent.parent / "play_wm"
    return resolved.parent / "play_wm"


if __name__ == "__main__":
    main()
