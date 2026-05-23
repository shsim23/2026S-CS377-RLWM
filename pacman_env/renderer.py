from __future__ import annotations
from typing import Optional, Union

import numpy as np

from .layout import Layout
from .state import GameState


class Renderer:
    def __init__(self, layout: Layout, mode: Optional[str] = None):
        self.layout = layout
        self.mode = mode
        self._surface = None
        self._clock = None

        if mode == "human":
            self._init_pygame()

    # ------------------------------------------------------------------ #
    def render(self, game_state: GameState) -> Optional[Union[str, np.ndarray]]:
        if self.mode is None:
            return None
        if self.mode == "ansi":
            return self._render_ansi(game_state)
        return self._render_pygame(game_state)

    def close(self) -> None:
        if self._surface is not None:
            import pygame
            pygame.quit()
            self._surface = None

    # ------------------------------------------------------------------ #
    def _render_ansi(self, gs: GameState) -> str:
        H, W = self.layout.height, self.layout.width
        grid = [list(row) for row in self._base_grid()]

        # Food
        for y in range(H):
            for x in range(W):
                if gs.food_mask[y, x]:
                    grid[y][x] = '.'

        # Pac-Man
        px, py = gs.pacman_pos
        grid[py][px] = 'P'

        # Ghosts
        for i, (gx, gy) in enumerate(gs.ghost_positions):
            if gs.ghost_alive[i]:
                grid[gy][gx] = 'G'

        return '\n'.join(''.join(row) for row in grid)

    def _base_grid(self):
        H, W = self.layout.height, self.layout.width
        grid = []
        for y in range(H):
            row = []
            for x in range(W):
                row.append('%' if self.layout.walls[y, x] else ' ')
            grid.append(row)
        return grid

    # ------------------------------------------------------------------ #
    CELL = 24  # pixels per cell

    def _init_pygame(self):
        import pygame
        pygame.init()
        H, W = self.layout.height, self.layout.width
        self._surface = pygame.display.set_mode((W * self.CELL, H * self.CELL))
        pygame.display.set_caption("Pac-Man")
        self._clock = pygame.time.Clock()

    def _render_pygame(self, gs: GameState) -> Optional[np.ndarray]:
        import pygame
        if self._surface is None:
            self._init_pygame()

        CELL = self.CELL
        H, W = self.layout.height, self.layout.width
        surf = self._surface if self._surface else pygame.Surface((W * CELL, H * CELL))

        # Background
        surf.fill((0, 0, 0))

        for y in range(H):
            for x in range(W):
                rect = (x * CELL, y * CELL, CELL, CELL)
                if self.layout.walls[y, x]:
                    pygame.draw.rect(surf, (0, 0, 180), rect)
                elif gs.food_mask[y, x]:
                    cx, cy = x * CELL + CELL // 2, y * CELL + CELL // 2
                    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), 3)

        # Pac-Man
        px, py = gs.pacman_pos
        pygame.draw.circle(
            surf, (255, 255, 0),
            (px * CELL + CELL // 2, py * CELL + CELL // 2),
            CELL // 2 - 2,
        )

        # Ghosts
        ghost_color = (255, 100, 100) if gs.power_mode_timer == 0 else (0, 100, 255)
        for i, (gx, gy) in enumerate(gs.ghost_positions):
            if gs.ghost_alive[i]:
                pygame.draw.circle(
                    surf, ghost_color,
                    (gx * CELL + CELL // 2, gy * CELL + CELL // 2),
                    CELL // 2 - 2,
                )

        if self.mode == "human":
            pygame.display.flip()
            self._clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

        if self.mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

        return None
