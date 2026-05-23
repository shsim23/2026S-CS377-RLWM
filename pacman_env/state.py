from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .constants import MAX_GHOSTS, MAX_GRID_H, MAX_GRID_W, STATE_DIM
from .layout import Layout


@dataclass
class GameState:
    pacman_pos:       Tuple[int, int]
    ghost_positions:  List[Tuple[int, int]]
    ghost_alive:      List[bool]
    food_mask:        np.ndarray      # (H, W) bool — actual layout size
    power_mode_timer: int             # 0 = inactive
    step_count:       int
    done:             bool

    def copy(self) -> "GameState":
        return GameState(
            pacman_pos=self.pacman_pos,
            ghost_positions=list(self.ghost_positions),
            ghost_alive=list(self.ghost_alive),
            food_mask=self.food_mask.copy(),
            power_mode_timer=self.power_mode_timer,
            step_count=self.step_count,
            done=self.done,
        )


def _normalize(x: int, y: int) -> Tuple[float, float]:
    x_norm = 2.0 * x / (MAX_GRID_W - 1) - 1.0
    y_norm = 2.0 * y / (MAX_GRID_H - 1) - 1.0
    return float(x_norm), float(y_norm)


class StateBuilder:
    def __init__(self, layout: Layout, num_ghosts: int):
        self.layout = layout
        self.num_ghosts = num_ghosts
        padded = layout.to_padded_arrays()
        self._wall_mask_flat = padded["walls"].flatten().astype(np.float32)  # (441,)

    def build(self, gs: GameState) -> np.ndarray:
        """Returns (STATE_DIM,) float32 in [-1, 1]."""
        vec = np.empty(STATE_DIM, dtype=np.float32)
        offset = 0

        # 1. Pac-Man (2)
        px, py = _normalize(*gs.pacman_pos)
        vec[0] = px
        vec[1] = py
        offset = 2

        # 2. Ghost slots (MAX_GHOSTS * 4 = 16)
        for i in range(MAX_GHOSTS):
            if i < self.num_ghosts:
                gx, gy = _normalize(*gs.ghost_positions[i])
                alive = 1.0 if gs.ghost_alive[i] else 0.0
                valid = 1.0
            else:
                gx = gy = 0.0
                alive = 0.0
                valid = 0.0
            vec[offset]     = gx
            vec[offset + 1] = gy
            vec[offset + 2] = alive
            vec[offset + 3] = valid
            offset += 4

        # 3. Food mask (441) — padded to MAX_GRID_H x MAX_GRID_W
        food_padded = np.zeros((MAX_GRID_H, MAX_GRID_W), dtype=np.float32)
        food_padded[: self.layout.height, : self.layout.width] = gs.food_mask.astype(np.float32)
        vec[offset: offset + 441] = food_padded.flatten()
        offset += 441

        # 4. Wall mask (441)
        vec[offset: offset + 441] = self._wall_mask_flat
        offset += 441

        # 5. Power timer (1) — normalized to [0, 1]
        # frightened_duration is not stored here; just clamp to [0,1] if provided
        max_timer = 30
        vec[offset] = float(np.clip(gs.power_mode_timer / max_timer, 0.0, 1.0))
        offset += 1

        assert offset == STATE_DIM
        return vec
