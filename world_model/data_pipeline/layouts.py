"""Serialize generated mazes to env-loadable ASCII layouts.

`maze_generator.generate_maze` returns a dict with an int-coded `_tile_grid`
and (row, col) entity positions. The Pac-Man env loads layouts as ASCII text
via `pacman_env.layout.LayoutParser.from_string`, using the glyphs in
`pacman_env.constants.Tile`. This module converts between the two.

Coordinate convention: the maze generator uses (row, col); the text grid is
row-major (line index = row, char index = col), which is what the env parser
expects. The env then stores positions internally as (x=col, y=row).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from maze_generator.constants import (
    HEIGHT,
    WIDTH,
    TILE_WALL,
    TILE_PATH,
    TILE_GATE,
    TILE_INTERIOR,
)
from pacman_env.constants import Tile


def maze_dict_to_layout_text(maze: Dict) -> str:
    """Convert a `generate_maze(...)` dict into env-loadable ASCII text.

    Mapping (env has no gate/interior concept — both are walkable empty cells):
        TILE_WALL                 -> '%'
        TILE_PATH (with food)      -> '.'
        TILE_PATH (no food)        -> ' '
        TILE_GATE / TILE_INTERIOR  -> ' '   (walkable)
        pacman_pos                 -> 'P'
        ghost_positions            -> 'G'
    """
    grid = maze["_tile_grid"]
    food_set = set(tuple(p) for p in maze["food_positions"])
    ghost_set = set(tuple(p) for p in maze["ghost_positions"])
    pacman = tuple(maze["pacman_pos"])

    lines = []
    for r in range(HEIGHT):
        row_chars = []
        for c in range(WIDTH):
            pos = (r, c)
            tile = grid[r, c]
            if pos == pacman:
                ch = Tile.PACMAN_START          # 'P'
            elif pos in ghost_set:
                ch = Tile.GHOST_START           # 'G'
            elif tile == TILE_WALL:
                ch = Tile.WALL                  # '%'
            elif tile == TILE_PATH:
                ch = Tile.FOOD if pos in food_set else Tile.EMPTY
            elif tile in (TILE_GATE, TILE_INTERIOR):
                ch = Tile.EMPTY                 # walkable, no food
            else:
                ch = Tile.EMPTY
            row_chars.append(ch)
        lines.append("".join(row_chars))
    return "\n".join(lines) + "\n"


def write_layout(maze: Dict, path: str | Path) -> Path:
    """Serialize `maze` and write it to `path` (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(maze_dict_to_layout_text(maze), encoding="utf-8")
    return path
