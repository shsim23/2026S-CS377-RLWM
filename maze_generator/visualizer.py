"""Visualization of a generated maze (ASCII + matplotlib).

ASCII symbols (per spec §8.2):
  ■   wall
  ' ' path (no food)
  ·   path with food
  P   pacman
  G   ghost
  =   ghost house gate
  H   ghost house interior (without ghost)
"""
from __future__ import annotations

from typing import Dict, Optional

from .constants import (
    HEIGHT,
    TILE_GATE,
    TILE_INTERIOR,
    TILE_PATH,
    TILE_WALL,
    WIDTH,
)


def ascii_render(maze: Dict) -> str:
    grid = maze["_tile_grid"]
    food_set = set(maze["food_positions"])
    ghost_set = set(maze["ghost_positions"])
    pacman = maze["pacman_pos"]

    lines = []
    for r in range(HEIGHT):
        row_chars = []
        for c in range(WIDTH):
            pos = (r, c)
            tile = grid[r, c]
            if pos == pacman:
                ch = "P"
            elif pos in ghost_set:
                ch = "G"
            elif tile == TILE_WALL:
                ch = "■"  # ■
            elif tile == TILE_GATE:
                ch = "="
            elif tile == TILE_INTERIOR:
                ch = "H"
            elif tile == TILE_PATH:
                ch = "·" if pos in food_set else " "
            else:
                ch = "?"
            row_chars.append(ch)
        lines.append("".join(row_chars))
    return "\n".join(lines)


def summary_line(maze: Dict) -> str:
    return (
        f"seed={maze['seed']} "
        f"food={maze['food_count']} "
        f"ghosts={len(maze['ghost_positions'])} "
        f"pacman={maze['pacman_pos']}"
    )


def render_image(
    maze: Dict,
    path: Optional[str] = None,
    cell_px: int = 24,
    title: Optional[str] = None,
):
    """Render the maze as a PNG-style figure.

    Pac-Man-ish color scheme: navy walls, black corridors, dotted pellets,
    yellow Pacman, red ghost, magenta gate, teal ghost-house floor.

    If `path` is given, save to that path and close the figure.
    Otherwise return the figure for interactive use.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    grid = maze["_tile_grid"]
    food_set = set(maze["food_positions"])
    ghost_set = set(maze["ghost_positions"])
    pacman = maze["pacman_pos"]

    fig_w = WIDTH * cell_px / 72  # inches @ 72dpi
    fig_h = HEIGHT * cell_px / 72
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=72)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.invert_yaxis()  # row=0 at top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")

    # Tile fills
    for r in range(HEIGHT):
        for c in range(WIDTH):
            tile = grid[r, c]
            if tile == TILE_WALL:
                color = "#1f1fa8"  # navy
            elif tile == TILE_GATE:
                color = "#d96fb8"  # magenta gate
            elif tile == TILE_INTERIOR:
                color = "#1c5e6f"  # teal interior
            else:
                color = "black"
            if color != "black":
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="none"))

    # Food pellets
    for (r, c) in food_set:
        ax.add_patch(Circle((c + 0.5, r + 0.5), 0.10, facecolor="#f7d9a1", edgecolor="none"))

    # Ghosts (red disk + simple "skirt")
    for (r, c) in ghost_set:
        ax.add_patch(Circle((c + 0.5, r + 0.45), 0.38, facecolor="#e23838", edgecolor="none"))
        ax.add_patch(Rectangle((c + 0.12, r + 0.45), 0.76, 0.40, facecolor="#e23838", edgecolor="none"))
        # eyes
        ax.add_patch(Circle((c + 0.35, r + 0.42), 0.08, facecolor="white", edgecolor="none"))
        ax.add_patch(Circle((c + 0.65, r + 0.42), 0.08, facecolor="white", edgecolor="none"))

    # Pacman
    pr, pc = pacman
    ax.add_patch(Circle((pc + 0.5, pr + 0.5), 0.42, facecolor="#ffd400", edgecolor="none"))
    # mouth wedge (right-facing)
    from matplotlib.patches import Wedge
    ax.add_patch(Wedge((pc + 0.5, pr + 0.5), 0.42, -25, 25, facecolor="black", edgecolor="none"))

    if title:
        ax.set_title(title, fontsize=10)

    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return None
    return fig
