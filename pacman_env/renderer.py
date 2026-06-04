from __future__ import annotations
import math
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
        self._last_pacman_pos = None
        self._last_step_count = None
        self._pacman_facing = 0.0

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
        self._last_pacman_pos = None
        self._last_step_count = None
        self._pacman_facing = 0.0

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
    BG_COLOR = (0, 0, 0)
    WALL_FACE = (20, 20, 184)
    WALL_EDGE = (61, 61, 255)
    FOOD_COLOR = (255, 217, 179)
    PAC_COLOR = (255, 224, 0)
    PAC_EDGE = (202, 168, 0)
    EYE_COLOR = (34, 34, 34)
    GHOST_COLORS = (
        (255, 0, 0),
        (255, 184, 255),
        (0, 255, 255),
        (255, 184, 82),
    )
    FRIGHTENED_GHOST = (0, 100, 255)
    FRIGHTENED_PUPIL = (255, 217, 179)

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

        surf.fill(self.BG_COLOR)
        if (
            self._last_step_count is not None
            and (gs.step_count < self._last_step_count or (gs.step_count == 0 and self._last_step_count > 0))
        ):
            self._last_pacman_pos = None
            self._pacman_facing = 0.0

        for y in range(H):
            for x in range(W):
                rect = (x * CELL, y * CELL, CELL, CELL)
                if self.layout.walls[y, x]:
                    pygame.draw.rect(surf, self.WALL_FACE, rect)
                    pygame.draw.rect(surf, self.WALL_EDGE, rect, 1)
                elif gs.food_mask[y, x]:
                    cx, cy = x * CELL + CELL // 2, y * CELL + CELL // 2
                    pygame.draw.circle(surf, self.FOOD_COLOR, (cx, cy), max(2, CELL // 9))

        self._draw_pacman(pygame, surf, gs.pacman_pos)

        for i, (gx, gy) in enumerate(gs.ghost_positions):
            if gs.ghost_alive[i]:
                color = (
                    self.FRIGHTENED_GHOST
                    if gs.power_mode_timer > 0
                    else self.GHOST_COLORS[i % len(self.GHOST_COLORS)]
                )
                self._draw_ghost(pygame, surf, (gx, gy), color, gs.pacman_pos, gs.power_mode_timer > 0)

        self._last_step_count = gs.step_count

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

    def _cell_center(self, pos: tuple[int, int]) -> tuple[int, int]:
        x, y = pos
        return x * self.CELL + self.CELL // 2, y * self.CELL + self.CELL // 2

    def _pacman_angle(self, pos: tuple[int, int]) -> float:
        if self._last_pacman_pos is not None:
            dx = pos[0] - self._last_pacman_pos[0]
            dy = pos[1] - self._last_pacman_pos[1]
            if abs(dx) + abs(dy) == 1:
                self._pacman_facing = math.atan2(dy, dx)
            elif abs(dx) + abs(dy) > 1:
                self._pacman_facing = 0.0
        self._last_pacman_pos = pos
        return self._pacman_facing

    def _draw_pacman(self, pygame, surf, pos: tuple[int, int]) -> None:
        cx, cy = self._cell_center(pos)
        radius = self.CELL // 2 - 2
        facing = self._pacman_angle(pos)
        mouth = math.radians(32.0)
        points = [(cx, cy)]
        steps = 26
        start = facing + mouth
        end = facing + (2.0 * math.pi) - mouth
        for i in range(steps + 1):
            a = start + (end - start) * i / steps
            points.append((int(round(cx + radius * math.cos(a))), int(round(cy + radius * math.sin(a)))))
        pygame.draw.polygon(surf, self.PAC_COLOR, points)
        pygame.draw.lines(surf, self.PAC_EDGE, False, points[1:], 1)

        eye_x = cx + 0.10 * self.CELL * math.cos(facing) - 0.18 * self.CELL * math.sin(facing)
        eye_y = cy + 0.10 * self.CELL * math.sin(facing) + 0.18 * self.CELL * math.cos(facing)
        pygame.draw.circle(surf, self.EYE_COLOR, (int(round(eye_x)), int(round(eye_y))), max(1, self.CELL // 14))

    def _draw_ghost(
        self,
        pygame,
        surf,
        pos: tuple[int, int],
        color: tuple[int, int, int],
        pacman_pos: tuple[int, int],
        frightened: bool,
    ) -> None:
        cx, cy = self._cell_center(pos)
        radius = self.CELL // 2 - 2
        points = []
        for i in range(15):
            angle = math.pi - math.pi * i / 14
            points.append((cx + radius * math.cos(angle), cy - radius * math.sin(angle)))
        points.append((cx + radius, cy + 0.25 * radius))
        for i in range(9):
            x = cx + radius - 2 * radius * i / 8
            y = cy + (radius if i % 2 == 0 else 0.68 * radius)
            points.append((x, y))
        points.append((cx - radius, cy + 0.25 * radius))
        pygame.draw.polygon(surf, color, [(int(round(x)), int(round(y))) for x, y in points])

        dx = pacman_pos[0] - pos[0]
        dy = pacman_pos[1] - pos[1]
        length = max(1.0, math.hypot(dx, dy))
        pupil_dx = int(round((dx / length) * self.CELL * 0.05))
        pupil_dy = int(round((dy / length) * self.CELL * 0.05))
        pupil_color = self.FRIGHTENED_PUPIL if frightened else self.WALL_FACE
        for sx in (-1, 1):
            eye = (int(round(cx + sx * self.CELL * 0.17)), int(round(cy - self.CELL * 0.12)))
            pygame.draw.circle(surf, (255, 255, 255), eye, max(2, self.CELL // 8))
            pygame.draw.circle(
                surf,
                pupil_color,
                (eye[0] + pupil_dx, eye[1] + pupil_dy),
                max(1, self.CELL // 16),
            )
