from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pacman_env.constants import ACTION_DELTAS, MAX_GHOSTS, MAX_GRID_H, MAX_GRID_W, Action
from world_model.dreamer.rssm import FOOD_SLICE, GHOST_SLICE, PAC_SLICE, WALL_SLICE


@dataclass
class ReliabilityStats:
    u_rule: torch.Tensor
    u_rule_norm: torch.Tensor
    u_prior: torch.Tensor | None
    u_prior_norm: torch.Tensor | None
    u_total: torch.Tensor
    confidence: torch.Tensor
    rule_truncate: torch.Tensor
    secondary_truncate: torch.Tensor


class RunningMean:
    def __init__(self, momentum: float = 0.99, initial: float = 1.0, device: torch.device | str = "cpu") -> None:
        self.momentum = float(momentum)
        self.value = torch.tensor(float(initial), dtype=torch.float32, device=device)
        self.initialized = False

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.detach().mean().clamp_min(1e-8)
        if not self.initialized:
            self.value = mean
            self.initialized = True
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * mean
        return x / (self.value + 1e-8)


class RuleBasedTransitionScorer:
    """Hard Pac-Man transition checks used as WM rollout reliability signal."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.wall_penalty = float(cfg.get("wall_violation_penalty", 1.0))
        self.bounds_penalty = float(cfg.get("bounds_violation_penalty", 1.0))
        self.food_appearance_penalty = float(cfg.get("food_appearance_penalty", 1.0))
        self.invalid_food_removal_penalty = float(cfg.get("invalid_food_removal_penalty", 1.0))
        self.enable_collision_done_check = bool(cfg.get("enable_collision_done_check", True))
        self.collision_done_penalty = float(cfg.get("collision_done_penalty", 1.0))

    @torch.no_grad()
    def score(
        self,
        current: torch.Tensor,
        predicted: torch.Tensor,
        actions: torch.Tensor,
        done_prob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = current.device
        B = current.shape[0]
        score = torch.zeros(B, dtype=torch.float32, device=device)
        actions = actions.long().view(B)

        curr_pac = self._xy_cells(current[:, PAC_SLICE])
        pred_pac = self._xy_cells(predicted[:, PAC_SLICE])
        walls = current[:, WALL_SLICE].reshape(B, MAX_GRID_H, MAX_GRID_W) > 0.5

        expected = curr_pac.clone()
        for action in Action:
            mask = actions == int(action)
            if not mask.any():
                continue
            dx, dy = ACTION_DELTAS[action]
            cand = curr_pac[mask] + torch.tensor([dx, dy], dtype=torch.long, device=device)
            legal = self._is_in_bounds(cand) & ~self._wall_at(walls[mask], cand)
            expected[mask] = torch.where(legal.unsqueeze(-1), cand, curr_pac[mask])

        score += self._manhattan(pred_pac, expected)
        score += torch.relu(self._manhattan(pred_pac, curr_pac) - 1.0)
        score += self._bounds_penalty(pred_pac)
        score += self._wall_penalty(walls, pred_pac)

        curr_ghosts = current[:, GHOST_SLICE].reshape(B, MAX_GHOSTS, 4)
        pred_ghosts = predicted[:, GHOST_SLICE].reshape(B, MAX_GHOSTS, 4)
        curr_g_xy = self._xy_cells(curr_ghosts[..., :2].reshape(-1, 2)).reshape(B, MAX_GHOSTS, 2)
        pred_g_xy = self._xy_cells(pred_ghosts[..., :2].reshape(-1, 2)).reshape(B, MAX_GHOSTS, 2)
        valid_alive = (current[:, GHOST_SLICE].reshape(B, MAX_GHOSTS, 4)[..., 2] > 0.5) & (
            current[:, GHOST_SLICE].reshape(B, MAX_GHOSTS, 4)[..., 3] > 0.5
        )
        ghost_speed = torch.relu(self._manhattan(pred_g_xy, curr_g_xy) - 1.0) * valid_alive.float()
        score += ghost_speed.sum(dim=1)
        ghost_bounds = self._bounds_penalty(pred_g_xy.reshape(-1, 2)).reshape(B, MAX_GHOSTS) * valid_alive.float()
        score += ghost_bounds.sum(dim=1)
        ghost_walls = self._wall_penalty(
            walls.repeat_interleave(MAX_GHOSTS, dim=0), pred_g_xy.reshape(-1, 2)
        ).reshape(B, MAX_GHOSTS) * valid_alive.float()
        score += ghost_walls.sum(dim=1)

        curr_food = current[:, FOOD_SLICE].reshape(B, -1) > 0.5
        pred_food = predicted[:, FOOD_SLICE].reshape(B, -1) > 0.5
        curr_count = curr_food.float().sum(dim=1)
        pred_count = pred_food.float().sum(dim=1)
        score += self.food_appearance_penalty * torch.relu(pred_count - curr_count)

        removed = curr_food & ~pred_food
        removed_count = removed.float().sum(dim=1)
        score += torch.relu(removed_count - 1.0)
        pac_idx = (pred_pac[:, 1].clamp(0, MAX_GRID_H - 1) * MAX_GRID_W + pred_pac[:, 0].clamp(0, MAX_GRID_W - 1)).long()
        removed_at_pac = removed.gather(1, pac_idx.unsqueeze(1)).squeeze(1)
        invalid_removed = removed_count - removed_at_pac.float()
        score += self.invalid_food_removal_penalty * torch.relu(invalid_removed)

        if self.enable_collision_done_check and done_prob is not None:
            collision = (pred_g_xy == pred_pac[:, None, :]).all(dim=-1) & valid_alive
            collision_any = collision.any(dim=1).float()
            score += self.collision_done_penalty * collision_any * (1.0 - done_prob.clamp(0.0, 1.0))

        return score

    @staticmethod
    def _xy_cells(xy_norm: torch.Tensor) -> torch.Tensor:
        x = torch.round((xy_norm[..., 0] + 1.0) * (MAX_GRID_W - 1) / 2.0).long()
        y = torch.round((xy_norm[..., 1] + 1.0) * (MAX_GRID_H - 1) / 2.0).long()
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _manhattan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).abs().float().sum(dim=-1)

    @staticmethod
    def _is_in_bounds(xy: torch.Tensor) -> torch.Tensor:
        return (xy[..., 0] >= 0) & (xy[..., 0] < MAX_GRID_W) & (xy[..., 1] >= 0) & (xy[..., 1] < MAX_GRID_H)

    def _bounds_penalty(self, xy: torch.Tensor) -> torch.Tensor:
        return (~self._is_in_bounds(xy)).float() * self.bounds_penalty

    def _wall_penalty(self, walls: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        in_bounds = self._is_in_bounds(xy)
        return (self._wall_at(walls, xy) & in_bounds).float() * self.wall_penalty

    @staticmethod
    def _wall_at(walls: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        in_bounds = RuleBasedTransitionScorer._is_in_bounds(xy)
        x = xy[..., 0].clamp(0, MAX_GRID_W - 1).long()
        y = xy[..., 1].clamp(0, MAX_GRID_H - 1).long()
        batch = torch.arange(walls.shape[0], device=walls.device)
        wall = walls[batch, y, x]
        return wall | ~in_bounds

