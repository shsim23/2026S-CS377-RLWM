from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from pacman_env import PacmanEnv, RewardConfig


def video_path(video_arg: str | Path, stem: str, iteration: int | None = None) -> Path:
    path = Path(video_arg)
    suffix = f"_{iteration:06d}" if iteration is not None else ""
    if path.suffix.lower() == ".mp4":
        if iteration is None:
            return path
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path / f"{stem}{suffix}_{timestamp}.mp4"


def save_video(frames: list, path: str | Path, fps: int = 10) -> Path:
    if not frames:
        raise ValueError("No frames were captured; refusing to write an empty video.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.mimsave(out, frames, fps=fps)
    return out


def record_policy_video(
    policy,
    env_cfg: dict[str, Any],
    path: str | Path,
    device: str = "cpu",
    seed: int = 0,
    max_steps: int | None = None,
    fps: int = 10,
) -> dict[str, float]:
    env = _make_render_env(env_cfg)
    frames = []
    total_reward = 0.0
    steps = 0
    obs, _ = env.reset(seed=seed)
    try:
        horizon = int(max_steps or env_cfg.get("max_steps", env_cfg.get("episode", {}).get("max_steps", 500)))
        for _ in range(horizon):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            td = TensorDict({"policy": obs_tensor}, batch_size=[1], device=device)
            with torch.inference_mode():
                action = policy(td, stochastic_output=False)
            obs, reward, terminated, truncated, _ = env.step(int(action.view(-1)[0].item()))
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                break
    finally:
        env.close()

    save_video(frames, path, fps=fps)
    return {"return": total_reward, "length": float(steps)}


def _make_render_env(env_cfg: dict[str, Any]) -> PacmanEnv:
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
        render_mode="rgb_array",
        randomize_spawn=bool(env_cfg.get("randomize_spawn", True)),
        min_spawn_dist=int(env_cfg.get("min_spawn_dist", 3)),
    )
