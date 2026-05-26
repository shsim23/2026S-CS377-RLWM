"""Stage 2 — randomized DFS carving on the left half (cols 2..10).

The carving grid restricts PATH candidate cells to (row, col) with both
coordinates even. Adjacent candidates are separated by one wall tile; this
structurally guarantees the 1-tile-thick-path / 1-tile-thick-wall property.
"""
from __future__ import annotations

import random
from typing import List, Set, Tuple

import numpy as np

from .constants import (
    CARVING_COLS_LEFT,
    CARVING_ROWS,
    GATE_OUTSIDE_NEIGHBOR,
    PACMAN_START,
    TILE_PATH,
    TILE_WALL,
    is_in_ghost_house_region,
)


Coord = Tuple[int, int]


def _candidate_cells() -> List[Coord]:
    """Even-even cells in the left half, excluding the ghost-house region."""
    cells = []
    for r in CARVING_ROWS:
        for c in CARVING_COLS_LEFT:
            if is_in_ghost_house_region(r, c):
                continue
            cells.append((r, c))
    return cells


def _neighbors(cell: Coord, candidate_set: Set[Coord]) -> List[Coord]:
    r, c = cell
    out = []
    for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        nb = (r + dr, c + dc)
        if nb in candidate_set:
            out.append(nb)
    return out


def carve_left_half(
    grid: np.ndarray,
    rng: random.Random,
    connectivity: float,
) -> None:
    """Carve the left half of `grid` in place using randomized DFS.

    The DFS spans all candidate cells reachable from PACMAN_START; additional
    edges are then opened with probability `connectivity` to introduce cycles.
    The intermediate wall between two reached candidates becomes PATH.
    The mirror step (Stage 3) is handled by the caller.
    """
    candidates = _candidate_cells()
    candidate_set: Set[Coord] = set(candidates)

    if PACMAN_START not in candidate_set:
        raise RuntimeError(
            f"Pacman start {PACMAN_START} is not on the carving grid"
        )

    # Randomized iterative DFS
    visited: Set[Coord] = {PACMAN_START}
    grid[PACMAN_START] = TILE_PATH
    stack: List[Coord] = [PACMAN_START]

    while stack:
        current = stack[-1]
        nbs = [n for n in _neighbors(current, candidate_set) if n not in visited]
        if not nbs:
            stack.pop()
            continue
        rng.shuffle(nbs)
        nxt = nbs[0]
        # Carve the wall between current and next, plus next itself
        wall = ((current[0] + nxt[0]) // 2, (current[1] + nxt[1]) // 2)
        if is_in_ghost_house_region(*wall):
            # Should not happen given candidate filtering, but guard anyway.
            visited.add(nxt)
            continue
        grid[wall] = TILE_PATH
        grid[nxt] = TILE_PATH
        visited.add(nxt)
        stack.append(nxt)

    # Connectivity: open extra walls between adjacent visited candidates.
    # Each undirected pair considered once (favor +row / +col direction).
    if connectivity > 0:
        for cell in candidates:
            if cell not in visited:
                continue
            r, c = cell
            for dr, dc in ((2, 0), (0, 2)):
                nb = (r + dr, c + dc)
                if nb not in visited:
                    continue
                wall = (r + dr // 2, c + dc // 2)
                if is_in_ghost_house_region(*wall):
                    continue
                if grid[wall] == TILE_PATH:
                    continue
                if rng.random() < connectivity:
                    grid[wall] = TILE_PATH

    # Force the tile directly above the gate to be PATH so the ghost house
    # has an external maze connection. This tile is on the carving grid and
    # should already have been visited by the DFS; we assert here to surface
    # bugs early.
    if grid[GATE_OUTSIDE_NEIGHBOR] != TILE_PATH:
        raise RuntimeError(
            f"Gate's outside neighbor {GATE_OUTSIDE_NEIGHBOR} was not "
            f"carved — DFS connectivity bug."
        )
