from __future__ import annotations
from collections import deque
from typing import List, Tuple

import numpy as np


class GhostController:
    def __init__(self, epsilon: float, np_random: np.random.Generator):
        self.epsilon = epsilon
        self.rng = np_random

    def step(
        self,
        ghost_pos: Tuple[int, int],
        pacman_pos: Tuple[int, int],
        walls: np.ndarray,
    ) -> Tuple[int, int]:
        if self.rng.random() < self.epsilon:
            return self._random_legal_step(ghost_pos, walls)
        return self._bfs_chase_step(ghost_pos, pacman_pos, walls)

    # ------------------------------------------------------------------ #
    def _random_legal_step(
        self, pos: Tuple[int, int], walls: np.ndarray
    ) -> Tuple[int, int]:
        x, y = pos
        neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        legal = [
            (nx, ny)
            for nx, ny in neighbors
            if 0 <= ny < walls.shape[0] and 0 <= nx < walls.shape[1] and not walls[ny, nx]
        ]
        if not legal:
            return pos
        idx = int(self.rng.integers(len(legal)))
        return legal[idx]

    def _bfs_chase_step(
        self,
        ghost_pos: Tuple[int, int],
        pacman_pos: Tuple[int, int],
        walls: np.ndarray,
    ) -> Tuple[int, int]:
        if ghost_pos == pacman_pos:
            return ghost_pos

        H, W = walls.shape
        visited = {ghost_pos}
        # BFS queue: (position, first_step)
        queue: deque = deque()
        x, y = ghost_pos
        for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
            if 0 <= ny < H and 0 <= nx < W and not walls[ny, nx]:
                first = (nx, ny)
                if first == pacman_pos:
                    return first
                visited.add(first)
                queue.append((first, first))

        while queue:
            pos, first_step = queue.popleft()
            cx, cy = pos
            for nx, ny in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)]:
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if walls[ny, nx] or (nx, ny) in visited:
                    continue
                if (nx, ny) == pacman_pos:
                    return first_step
                visited.add((nx, ny))
                queue.append(((nx, ny), first_step))

        # Pac-Man unreachable — fall back to random
        return self._random_legal_step(ghost_pos, walls)
