"""Generate and visualize sample mazes.

Usage:
    python -m maze_generator.demo            # print 10 mazes to stdout
    python -m maze_generator.demo --save-dir maze_generator/examples
        # saves PNG + .txt for each seed, plus a contact-sheet grid PNG
"""
from __future__ import annotations

import argparse
import os

from .generator import generate_maze
from .visualizer import ascii_render, render_image, summary_line


def _save_contact_sheet(mazes, out_path):
    """Render all mazes in a single image grid (for quick visual diversity)."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, Wedge

    from .constants import (
        HEIGHT, WIDTH, TILE_WALL, TILE_GATE, TILE_INTERIOR,
    )

    n = len(mazes)
    cols = min(5, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, maze in enumerate(mazes):
        ax = axes[idx]
        grid = maze["_tile_grid"]
        food_set = set(maze["food_positions"])
        ghost_set = set(maze["ghost_positions"])
        pacman = maze["pacman_pos"]

        ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT)
        ax.invert_yaxis(); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("black")

        for r in range(HEIGHT):
            for c in range(WIDTH):
                t = grid[r, c]
                if t == TILE_WALL:
                    ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#1f1fa8", edgecolor="none"))
                elif t == TILE_GATE:
                    ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#d96fb8", edgecolor="none"))
                elif t == TILE_INTERIOR:
                    ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#1c5e6f", edgecolor="none"))
        for (r, c) in food_set:
            ax.add_patch(Circle((c + 0.5, r + 0.5), 0.18, facecolor="#f7d9a1", edgecolor="none"))
        for (r, c) in ghost_set:
            ax.add_patch(Circle((c + 0.5, r + 0.5), 0.45, facecolor="#e23838", edgecolor="none"))
        pr, pc = pacman
        ax.add_patch(Circle((pc + 0.5, pr + 0.5), 0.45, facecolor="#ffd400", edgecolor="none"))
        ax.add_patch(Wedge((pc + 0.5, pr + 0.5), 0.45, -25, 25, facecolor="black", edgecolor="none"))
        ax.set_title(f"seed={maze['seed']} food={maze['food_count']}", fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--connectivity", type=float, default=0.3)
    p.add_argument("--num-ghosts", type=int, default=1)
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If given, save each maze as PNG + .txt to this directory "
             "(also writes a contact-sheet 'all_seeds.png').",
    )
    args = p.parse_args()

    mazes = []
    for i in range(args.count):
        seed = args.start_seed + i
        maze = generate_maze(
            seed=seed,
            connectivity=args.connectivity,
            num_ghosts=args.num_ghosts,
        )
        mazes.append(maze)
        print(f"\n=== {summary_line(maze)} ===")
        print(ascii_render(maze))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for maze in mazes:
            seed = maze["seed"]
            base = os.path.join(args.save_dir, f"maze_seed{seed:03d}")
            render_image(maze, path=base + ".png", title=f"seed={seed}")
            with open(base + ".txt", "w") as f:
                f.write(summary_line(maze) + "\n")
                f.write(ascii_render(maze) + "\n")
        sheet = os.path.join(args.save_dir, "all_seeds.png")
        _save_contact_sheet(mazes, sheet)
        print(f"\n[saved] {args.count} PNG + .txt files to {args.save_dir}")
        print(f"[saved] contact sheet: {sheet}")


if __name__ == "__main__":
    main()
