from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from pacman_env import PacmanEnv, RewardConfig, STATE_DIM


class RslPacmanVecEnv(VecEnv):
    """Synchronous rsl_rl VecEnv adapter over several ground-truth PacmanEnv instances."""

    def __init__(self, cfg: dict[str, Any], device: str = "cpu", render_mode: str | None = None) -> None:
        self.cfg = dict(cfg)
        self.device = torch.device(device)
        self.num_envs = int(self.cfg.get("num_envs", 16))
        self.num_actions = 1
        self.action_space_n = int(self.cfg.get("action_space_n", 5))
        self.max_episode_length = int(self.cfg.get("max_steps", self.cfg.get("max_episode_length", 500)))
        self.step_dt = 1.0
        self.unwrapped = self

        base_seed = int(self.cfg.get("seed", 0))
        self.envs = [self._make_env(render_mode=render_mode) for _ in range(self.num_envs)]
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._obs = torch.zeros(self.num_envs, STATE_DIM, dtype=torch.float32, device=self.device)
        self._episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=base_seed + i)
            self._obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self._obs.clone()}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_ids = actions.detach().to("cpu").long().view(self.num_envs, -1)[:, 0].numpy()
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        timeout_failures = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        wins = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        deaths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        pellets_remaining = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_wins: list[float] = []
        completed_deaths: list[float] = []
        completed_timeouts: list[float] = []
        completed_pellets_remaining: list[float] = []

        for i, (env, action) in enumerate(zip(self.envs, action_ids, strict=True)):
            action_int = int(np.clip(action, 0, self.action_space_n - 1))
            obs, reward, terminated, truncated, info = env.step(action_int)

            event = info.get("event", {})
            won = bool(event.get("won", False))
            died = bool(event.get("died", False))
            timeout_fail = bool(truncated and not terminated and not won)

            if timeout_fail:
                reward += float(self.cfg.get("timeout_penalty", -75.0))

            done = bool(terminated or truncated)

            rewards[i] = float(reward)
            dones[i] = done
            # For Pacman, max-step truncation without winning is a failed episode,
            # not a successful survival timeout to bootstrap through.
            time_outs[i] = False
            timeout_failures[i] = float(timeout_fail)
            self._episode_returns[i] += float(reward)
            self.episode_length_buf[i] += 1

            wins[i] = float(won)
            deaths[i] = float(died)
            pellets_remaining[i] = float(info.get("pellets_remaining", 0))

            if done:
                completed_returns.append(float(self._episode_returns[i].item()))
                completed_lengths.append(int(self.episode_length_buf[i].item()))
                completed_wins.append(float(won))
                completed_deaths.append(float(died))
                completed_timeouts.append(float(timeout_fail))
                completed_pellets_remaining.append(float(info.get("pellets_remaining", 0)))
                obs, _ = env.reset()
                self._episode_returns[i] = 0.0
                self.episode_length_buf[i] = 0

            self._obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        extras: dict[str, Any] = {
            "time_outs": time_outs,
            "log": {
                "/pacman/win_rate": wins,
                "/pacman/death_rate": deaths,
                "/pacman/timeout_failure_rate": timeout_failures,
                "/pacman/pellets_remaining": pellets_remaining,
            },
        }
        if completed_returns:
            extras["episode"] = {
                "return": torch.tensor(completed_returns, dtype=torch.float32, device=self.device),
                "length": torch.tensor(completed_lengths, dtype=torch.float32, device=self.device),
                "win": torch.tensor(completed_wins, dtype=torch.float32, device=self.device),
                "death": torch.tensor(completed_deaths, dtype=torch.float32, device=self.device),
                "timeout": torch.tensor(completed_timeouts, dtype=torch.float32, device=self.device),
                "pellets_remaining": torch.tensor(
                    completed_pellets_remaining, dtype=torch.float32, device=self.device
                ),
            }

        return self.get_observations(), rewards, dones, extras

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _make_env(self, render_mode: str | None = None) -> PacmanEnv:
        reward_cfg = RewardConfig(**self.cfg.get("reward", {}))
        ghost_cfg = self.cfg.get("ghost", {})
        power_cfg = self.cfg.get("power_pellet", {})
        episode_cfg = self.cfg.get("episode", {})
        return PacmanEnv(
            layout_path=self.cfg["layout_file"],
            num_ghosts=int(ghost_cfg.get("num_ghosts", self.cfg.get("num_ghosts", 1))),
            ghost_epsilon=float(ghost_cfg.get("epsilon", self.cfg.get("ghost_epsilon", 0.2))),
            ghost_policy=str(ghost_cfg.get("policy", self.cfg.get("ghost_policy", "chase_stochastic"))),
            ghost_speed_ratio=float(ghost_cfg.get("speed_ratio", self.cfg.get("ghost_speed_ratio", 1.0))),
            power_pellet_enabled=bool(power_cfg.get("enabled", self.cfg.get("power_pellet_enabled", False))),
            frightened_duration=int(power_cfg.get("frightened_duration", self.cfg.get("frightened_duration", 30))),
            max_steps=int(episode_cfg.get("max_steps", self.cfg.get("max_steps", 500))),
            reward_config=reward_cfg,
            render_mode=render_mode,
            randomize_spawn=bool(self.cfg.get("randomize_spawn", True)),
            min_spawn_dist=int(self.cfg.get("min_spawn_dist", 3)),
        )
