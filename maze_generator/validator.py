"""Stage 5 — validation: connectivity / symmetry / no dead-end."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .constants import (
    GATE_OUTSIDE_NEIGHBOR,
    GATE_POS,
    GHOST_HOUSE_INTERIOR_TILES,
    HEIGHT,
    PACMAN_START,
    SYMMETRY_AXIS_COL,
    TILE_GATE,
    TILE_INTERIOR,
    TILE_PATH,
    TILE_WALL,
    WIDTH,
)


Coord = Tuple[int, int]


@dataclass
class ValidationReport:
    ok: bool
    errors: List[str]


def _walkable_for_pacman(grid: np.ndarray, r: int, c: int) -> bool:
    return grid[r, c] == TILE_PATH


def _walkable_for_ghost(grid: np.ndarray, r: int, c: int) -> bool:
    return grid[r, c] in (TILE_PATH, TILE_GATE, TILE_INTERIOR)


def _bfs_reachable(
    grid: np.ndarray,
    start: Coord,
    walkable_fn,
) -> set:
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < HEIGHT and 0 <= nc < WIDTH):
                continue
            if (nr, nc) in seen:
                continue
            if not walkable_fn(grid, nr, nc):
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def validate(grid: np.ndarray) -> ValidationReport:
    errors: List[str] = []

    # 1. Connectivity of all PATH cells (Pacman-walkable)
    path_cells = set(map(tuple, np.argwhere(grid == TILE_PATH).tolist()))
    if path_cells:
        reachable = _bfs_reachable(grid, PACMAN_START, _walkable_for_pacman)
        missing = path_cells - reachable
        if missing:
            errors.append(
                f"{len(missing)} PATH cells unreachable from pacman start "
                f"(e.g., {next(iter(missing))})"
            )

    # Pacman → gate-outside neighbor reachable
    if GATE_OUTSIDE_NEIGHBOR not in _bfs_reachable(
        grid, PACMAN_START, _walkable_for_pacman
    ):
        errors.append(
            f"Pacman start {PACMAN_START} cannot reach gate-outside neighbor "
            f"{GATE_OUTSIDE_NEIGHBOR}"
        )

    # Ghost in ghost house can exit through gate
    interior_reach = _bfs_reachable(grid, GHOST_HOUSE_INTERIOR_TILES[0], _walkable_for_ghost)
    if GATE_OUTSIDE_NEIGHBOR not in interior_reach:
        errors.append("Ghost house interior cannot reach outer maze via gate")

    # 2. Symmetry
    flipped = grid[:, ::-1]
    if not np.array_equal(grid, flipped):
        diff = np.argwhere(grid != flipped)
        errors.append(
            f"Map is not left/right symmetric ({len(diff)} mismatching tiles, "
            f"first at {tuple(diff[0])})"
        )

    # 3. No Pacman-visible dead-ends among PATH cells
    for (r, c) in path_cells:
        nb = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and grid[nr, nc] == TILE_PATH:
                nb += 1
        if nb < 2:
            errors.append(f"Dead-end PATH cell at ({r}, {c}) (degree {nb})")
            break  # one is enough to fail

    return ValidationReport(ok=not errors, errors=errors)
