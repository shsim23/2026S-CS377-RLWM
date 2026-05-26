"""Stage 4 — dead-end removal; Stage 6 — food placement."""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from .constants import (
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
    is_border,
    is_in_ghost_house_region,
)


Coord = Tuple[int, int]


def _mirror(coord: Coord) -> Coord:
    r, c = coord
    return (r, WIDTH - 1 - c)


def _path_neighbor_count(grid: np.ndarray, r: int, c: int) -> int:
    """Count PATH neighbors (regular path only) for Pacman's perspective."""
    count = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and grid[nr, nc] == TILE_PATH:
            count += 1
    return count


def _find_dead_ends(grid: np.ndarray) -> List[Coord]:
    out = []
    rs, cs = np.where(grid == TILE_PATH)
    for r, c in zip(rs.tolist(), cs.tolist()):
        if (r, c) == PACMAN_START:
            # Pacman start may have any degree; still treat as a real PATH
            # cell. If it ends up as a dead-end, dead-end removal handles it.
            pass
        if _path_neighbor_count(grid, r, c) <= 1:
            out.append((r, c))
    return out


def _breakable_walls(grid: np.ndarray, r: int, c: int) -> List[Coord]:
    """Return adjacent walls that may legally be converted to PATH."""
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < HEIGHT and 0 <= nc < WIDTH):
            continue
        if grid[nr, nc] != TILE_WALL:
            continue
        if is_border(nr, nc):
            continue
        if is_in_ghost_house_region(nr, nc):
            continue
        out.append((nr, nc))
    return out


def remove_dead_ends(
    grid: np.ndarray,
    rng: random.Random,
    max_iters: int = 64,
) -> None:
    """Iteratively break walls to remove dead-ends, preserving symmetry."""
    for _ in range(max_iters):
        dead_ends = _find_dead_ends(grid)
        if not dead_ends:
            return
        progressed = False
        for dead in dead_ends:
            if _path_neighbor_count(grid, *dead) > 1:
                # Already resolved by an earlier break in this pass
                continue
            cands = _breakable_walls(grid, *dead)
            if not cands:
                continue
            wall = rng.choice(cands)
            grid[wall] = TILE_PATH
            mirror_wall = _mirror(wall)
            if mirror_wall != wall:
                # Mirror tile must also be breakable (no border / no ghost
                # house region). Symmetry of the inputs guarantees this.
                if (
                    grid[mirror_wall] != TILE_WALL
                    or is_border(*mirror_wall)
                    or is_in_ghost_house_region(*mirror_wall)
                ):
                    # Reverse the break to keep symmetry; treat as no
                    # progress and let the outer loop / validator retry.
                    grid[wall] = TILE_WALL
                    continue
                grid[mirror_wall] = TILE_PATH
            progressed = True
        if not progressed:
            return  # validator will surface the failure → outer retry


def place_food(grid: np.ndarray) -> List[Coord]:
    """Stage 6 — every regular PATH tile gets food except pacman start.

    Ghost house interior tiles and the gate are not PATH, so they are
    naturally excluded.
    """
    food = []
    rs, cs = np.where(grid == TILE_PATH)
    for r, c in zip(rs.tolist(), cs.tolist()):
        if (r, c) == PACMAN_START:
            continue
        food.append((r, c))
    return food
