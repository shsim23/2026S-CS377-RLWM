"""Static constants for the Pac-Man maze generator.

All coordinates are (row, col), 0-indexed, with row=0 at the top.
"""
from __future__ import annotations


# --- Grid dimensions ---
HEIGHT = 21
WIDTH = 21
SYMMETRY_AXIS_COL = 10  # vertical mirror axis


# --- Tile codes (int enum stored in numpy int8 grid) ---
TILE_WALL = 0
TILE_PATH = 1
TILE_GATE = 2       # GHOST_ONLY_PATH
TILE_INTERIOR = 3   # GHOST_HOUSE_INTERIOR


# --- Pacman start (on carving grid; both even) ---
PACMAN_START = (14, 10)


# --- Ghost house (fixed for every map) ---
GATE_POS = (9, 10)
GHOST_HOUSE_INTERIOR_TILES = [(10, 9), (10, 10), (10, 11)]
GHOST_HOUSE_REGION = frozenset(
    (r, c) for r in (9, 10, 11) for c in (8, 9, 10, 11, 12)
)

# Tile directly above the gate must be PATH so the ghost house is reachable
# from the outside maze. It happens to be on the carving grid.
GATE_OUTSIDE_NEIGHBOR = (8, 10)


# --- Ghost spawn assignments ---
GHOST_STARTS_BY_NUM = {
    1: [(10, 10)],
    2: [(10, 9), (10, 11)],
    3: [(10, 9), (10, 10), (10, 11)],
}


# --- Carving grid (even rows, even cols) ---
CARVING_ROWS = tuple(range(2, HEIGHT - 1, 2))           # 2, 4, ..., 18
CARVING_COLS_LEFT = tuple(range(2, SYMMETRY_AXIS_COL + 1, 2))  # 2, 4, ..., 10


def is_in_ghost_house_region(r: int, c: int) -> bool:
    return (r, c) in GHOST_HOUSE_REGION


def is_border(r: int, c: int) -> bool:
    return r == 0 or c == 0 or r == HEIGHT - 1 or c == WIDTH - 1


# --- Retry policy ---
MAX_RETRIES = 5
