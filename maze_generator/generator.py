"""Main API for the Pac-Man maze generator (Phase 1).

Pipeline:
  Stage 1 — initialize grid + reserve ghost house + pacman start
  Stage 2 — randomized DFS carving (left half only)
  Stage 3 — mirror left to right
  Stage 4 — remove dead-ends (symmetric)
  Stage 5 — validate
  Stage 6 — place food

See `maze_generator_spec.md` for the full specification.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import carving, post_process, validator
from .constants import (
    GATE_POS,
    GHOST_HOUSE_INTERIOR_TILES,
    GHOST_HOUSE_REGION,
    GHOST_STARTS_BY_NUM,
    HEIGHT,
    MAX_RETRIES,
    PACMAN_START,
    SYMMETRY_AXIS_COL,
    TILE_GATE,
    TILE_INTERIOR,
    TILE_PATH,
    TILE_WALL,
    WIDTH,
)


Coord = Tuple[int, int]


def _initialize_grid() -> np.ndarray:
    """Stage 1 — borders + ghost house + pacman start."""
    grid = np.full((HEIGHT, WIDTH), TILE_WALL, dtype=np.int8)
    # Ghost house interior (3 walkable tiles)
    for r, c in GHOST_HOUSE_INTERIOR_TILES:
        grid[r, c] = TILE_INTERIOR
    # Gate
    grid[GATE_POS] = TILE_GATE
    # Pacman start (guaranteed PATH)
    grid[PACMAN_START] = TILE_PATH
    return grid


def _mirror_left_to_right(grid: np.ndarray) -> None:
    """Stage 3 — copy cols 2..9 to their mirrors (cols 11..18)."""
    for c in range(2, SYMMETRY_AXIS_COL):
        grid[:, WIDTH - 1 - c] = grid[:, c]
    # The border cols (0, 20) and the axis col (10) are unchanged.


def _try_generate(
    seed: Optional[int],
    connectivity: float,
    num_ghosts: int,
) -> Dict:
    rng = random.Random(seed)

    # Stage 1
    grid = _initialize_grid()

    # Stage 2
    carving.carve_left_half(grid, rng, connectivity)

    # Stage 3
    _mirror_left_to_right(grid)

    # Stage 4
    post_process.remove_dead_ends(grid, rng)

    # Stage 5
    report = validator.validate(grid)
    if not report.ok:
        raise RuntimeError(
            "Maze validation failed: " + "; ".join(report.errors)
        )

    # Stage 6
    food = post_process.place_food(grid)

    # --- Build output dict ---
    walls = (grid == TILE_WALL)
    ghost_only_tiles: List[Coord] = list(map(tuple, np.argwhere(grid == TILE_GATE).tolist()))
    ghost_house_interior: List[Coord] = list(map(tuple, np.argwhere(grid == TILE_INTERIOR).tolist()))

    ghost_positions = list(GHOST_STARTS_BY_NUM[num_ghosts])
    ghost_in_house = [True] * num_ghosts

    return {
        # Static map structure
        "walls": walls,
        "ghost_only_tiles": ghost_only_tiles,
        "ghost_house_interior": ghost_house_interior,
        # Agent positions
        "pacman_pos": PACMAN_START,
        "ghost_positions": ghost_positions,
        "ghost_in_house": ghost_in_house,
        # Food
        "food_positions": food,
        "food_count": len(food),
        # Score / done
        "score": 0,
        "done": False,
        # Phase 3+ stubs
        "power_pellet_positions": [],
        "warp_tunnel_pairs": [],
        # Debug / meta
        "seed": seed,
        "width": WIDTH,
        "height": HEIGHT,
        # Internal: int-coded tile grid (useful for visualization / WM input)
        "_tile_grid": grid,
    }


def generate_maze(
    width: int = WIDTH,
    height: int = HEIGHT,
    symmetric: bool = True,
    connectivity: float = 0.3,
    ghost_house: bool = True,
    num_ghosts: int = 1,
    num_warp_tunnels: int = 0,
    num_power_pellets: int = 0,
    seed: Optional[int] = None,
) -> Dict:
    """Generate a Pac-Man maze. See `maze_generator_spec.md` §7 for the API.

    Phase 1 supports: 21x21, symmetric=True, ghost_house=True, 1<=num_ghosts<=3,
    num_warp_tunnels=0, num_power_pellets=0. Other values raise.
    """
    if (width, height) != (WIDTH, HEIGHT):
        raise NotImplementedError(
            f"Phase 1 only supports {WIDTH}x{HEIGHT}; got {width}x{height}"
        )
    if not symmetric:
        raise NotImplementedError("Phase 1 requires symmetric=True")
    if not ghost_house:
        raise NotImplementedError("Phase 1 requires ghost_house=True")
    if num_ghosts not in GHOST_STARTS_BY_NUM:
        raise ValueError(f"num_ghosts must be 1..3; got {num_ghosts}")
    if num_warp_tunnels != 0:
        raise NotImplementedError("Warp tunnels arrive in Phase 3")
    if num_power_pellets != 0:
        raise NotImplementedError("Power pellets arrive in Phase 4")
    if not 0.0 <= connectivity <= 1.0:
        raise ValueError(f"connectivity must be in [0, 1]; got {connectivity}")

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        current_seed = (seed + attempt) if seed is not None else None
        try:
            return _try_generate(
                seed=current_seed,
                connectivity=connectivity,
                num_ghosts=num_ghosts,
            )
        except Exception as e:  # noqa: BLE001 — surfaced below
            last_error = e
            continue

    raise RuntimeError(
        f"Failed to generate valid maze after {MAX_RETRIES} attempts. "
        f"This likely indicates a bug — current constraints should always be "
        f"satisfiable on {WIDTH}x{HEIGHT}. Last error: {last_error}"
    )
