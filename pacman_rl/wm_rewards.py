from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pacman_env import RewardComputer, RewardConfig, StepEvent
from pacman_env.constants import MAX_GHOSTS, MAX_GRID_H, MAX_GRID_W
from world_model.dreamer.rssm import FOOD_SLICE, GHOST_SLICE, PAC_SLICE, POWER_SLICE


@dataclass
class DecodedRewardResult:
    reward: torch.Tensor
    win: torch.Tensor
    death: torch.Tensor
    ate_pellet: torch.Tensor
    ate_power: torch.Tensor
    ate_ghosts: torch.Tensor
    remaining_pellets: torch.Tensor


class DecodedStateRewardComputer:
    """Compute Pac-Man rewards from decoded WM state transitions.

    This mirrors RewardComputer semantics while deriving StepEvent fields from
    (previous decoded state, predicted next decoded state). It is intended only
    for WM imagined rollouts, not for GT PacmanEnv stepping.
    """

    def __init__(self, env_cfg: dict[str, Any]) -> None:
        self.reward_cfg = RewardConfig(**env_cfg.get("reward", {}))
        self.reward_computer = RewardComputer(self.reward_cfg)
        power_cfg = env_cfg.get("power_pellet", {})
        self.power_enabled = bool(power_cfg.get("enabled", env_cfg.get("power_pellet_enabled", False)))

    @torch.no_grad()
    def compute(
        self,
        prev_obs: torch.Tensor,
        pred_obs: torch.Tensor,
        episode_ended: torch.Tensor,
        total_pellets: torch.Tensor,
        initial_power: torch.Tensor | None = None,
    ) -> DecodedRewardResult:
        device = pred_obs.device
        B = pred_obs.shape[0]
        curr_food = prev_obs[:, FOOD_SLICE].reshape(B, -1) > 0.5
        pred_food = pred_obs[:, FOOD_SLICE].reshape(B, -1) > 0.5
        removed = curr_food & ~pred_food
        pred_pac = _xy_cells(pred_obs[:, PAC_SLICE])
        pac_idx = (pred_pac[:, 1].clamp(0, MAX_GRID_H - 1) * MAX_GRID_W + pred_pac[:, 0].clamp(0, MAX_GRID_W - 1)).long()
        removed_at_pac = removed.gather(1, pac_idx.unsqueeze(1)).squeeze(1)
        curr_count = curr_food.float().sum(dim=1)
        remaining = pred_food.float().sum(dim=1)
        ate_pellet = removed_at_pac & (remaining < curr_count)

        prev_power = prev_obs[:, POWER_SLICE].squeeze(-1)
        pred_power = pred_obs[:, POWER_SLICE].squeeze(-1)
        power_active = prev_power > 1e-3
        ate_power = torch.zeros(B, dtype=torch.bool, device=device)
        if self.power_enabled and initial_power is not None:
            power_here = initial_power.to(device).bool().gather(1, pac_idx.unsqueeze(1)).squeeze(1)
            ate_power = power_here & (prev_power <= 1e-3) & (pred_power > prev_power + 1e-3)

        pred_ghosts = pred_obs[:, GHOST_SLICE].reshape(B, MAX_GHOSTS, 4)
        prev_ghosts = prev_obs[:, GHOST_SLICE].reshape(B, MAX_GHOSTS, 4)
        pred_g_xy = _xy_cells(pred_ghosts[..., :2].reshape(-1, 2)).reshape(B, MAX_GHOSTS, 2)
        overlap = (pred_g_xy == pred_pac[:, None, :]).all(dim=-1)
        pred_alive_valid = (pred_ghosts[..., 2] > 0.5) & (pred_ghosts[..., 3] > 0.5)
        prev_alive_valid = (prev_ghosts[..., 2] > 0.5) & (prev_ghosts[..., 3] > 0.5)

        death = (overlap & pred_alive_valid).any(dim=1) & ~power_active
        eaten_ghost_mask = overlap & prev_alive_valid & (pred_ghosts[..., 2] <= 0.5) & power_active[:, None]
        ate_ghosts = eaten_ghost_mask.sum(dim=1).long()
        win = (remaining <= 0.0) & ~death

        rewards = torch.zeros(B, dtype=torch.float32, device=device)
        for i in range(B):
            event = StepEvent(
                ate_pellet=bool(ate_pellet[i].item()),
                ate_power=bool(ate_power[i].item()),
                ate_ghosts=int(ate_ghosts[i].item()),
                died=bool(death[i].item()),
                won=bool(win[i].item()),
                remaining_pellets=int(remaining[i].item()),
                total_pellets=int(total_pellets[i].item()),
                episode_ended=bool(episode_ended[i].item() or death[i].item() or win[i].item()),
            )
            rewards[i] = float(self.reward_computer.compute(event))

        return DecodedRewardResult(
            reward=rewards,
            win=win.float(),
            death=death.float(),
            ate_pellet=ate_pellet.float(),
            ate_power=ate_power.float(),
            ate_ghosts=ate_ghosts.float(),
            remaining_pellets=remaining.float(),
        )


def _xy_cells(xy_norm: torch.Tensor) -> torch.Tensor:
    x = torch.round((xy_norm[..., 0] + 1.0) * (MAX_GRID_W - 1) / 2.0).long()
    y = torch.round((xy_norm[..., 1] + 1.0) * (MAX_GRID_H - 1) / 2.0).long()
    return torch.stack([x, y], dim=-1)
