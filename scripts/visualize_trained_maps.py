"""Gallery of the maps that currently have a trained RL agent — rendered in a
classic Pac-Man style (blue maze, pellets, wedge Pac-Man, ghost sprites).

For each `checkpoints/rl_agents/<layout_id>/optimal.pt` that exists, builds the
layout's env, resets to the initial state, and draws it. Tiles all maps into one
PNG so you can see exactly which mazes the agents (and hence the world-model data)
cover.

Usage
-----
    python scripts/visualize_trained_maps.py
    python scripts/visualize_trained_maps.py --num-ghosts 2 --out logs/wm_eval/trained_maps.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pacman_env import PacmanEnv

WALL_FILL = "#1c1cc8"
WALL_EDGE = "#4d6bff"
PELLET = "#ffe6b3"
PACMAN = "#ffe000"
GHOST_COLORS = ["#ff2b2b", "#ff7bd5", "#19e0ff", "#ff9e3d"]   # Blinky/Pinky/Inky/Clyde
BG = "#000010"


def draw_ghost(ax, x, y, color, r=0.42):
    """Classic ghost silhouette: domed head + wavy skirt + eyes."""
    # dome (upper half) + body rectangle
    ax.add_patch(Wedge((x, y - 0.02), r, 0, 180, facecolor=color, edgecolor="none"))
    ax.add_patch(Rectangle((x - r, y - 0.02), 2 * r, r * 0.95, facecolor=color, edgecolor="none"))
    # wavy bottom (three little arches as a polygon)
    n = 4
    pts = [(x - r, y + r * 0.93)]
    xs = np.linspace(x - r, x + r, 2 * n + 1)
    for i, xi in enumerate(xs):
        dy = (r * 0.18) if i % 2 == 0 else 0.0
        pts.append((xi, y + r * 0.93 - dy))
    pts.append((x + r, y + r * 0.93))
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none"))
    # eyes
    for ex in (-0.16, 0.16):
        ax.add_patch(Circle((x + ex, y - 0.05), 0.13, facecolor="white", edgecolor="none"))
        ax.add_patch(Circle((x + ex + 0.05, y - 0.05), 0.06, facecolor="#1a1a55", edgecolor="none"))


def render_map(ax, walls, food, pac, ghosts, title=None):
    H, W = walls.shape
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)          # invert y so row 0 is on top
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#222255")

    ys, xs = np.nonzero(walls)
    for x, y in zip(xs, ys):
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=WALL_FILL,
                               edgecolor=WALL_EDGE, linewidth=0.6))
    fy, fx = np.nonzero(food)
    for x, y in zip(fx, fy):
        ax.add_patch(Circle((x, y), 0.11, facecolor=PELLET, edgecolor="none"))

    px, py = pac
    ax.add_patch(Wedge((px, py), 0.46, 33, 327, facecolor=PACMAN, edgecolor="none"))
    for i, (gx, gy) in enumerate(ghosts):
        draw_ghost(ax, gx, gy, GHOST_COLORS[i % len(GHOST_COLORS)])

    if title:
        ax.set_title(title, color="white", fontsize=11, pad=4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents-root", default="checkpoints/rl_agents")
    p.add_argument("--layout-pool", default="layouts/wm_pool")
    p.add_argument("--pool-split", default="train", choices=["train", "test"])
    p.add_argument("--num-ghosts", type=int, default=2, help="Ghost count to display.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="logs/wm_eval/trained_maps.png")
    args = p.parse_args()

    pool = json.loads((ROOT / args.layout_pool / "manifest.json").read_text())
    agents_root = ROOT / args.agents_root
    entries = [e for e in pool[args.pool_split]
               if (agents_root / e["layout_id"] / "optimal.pt").exists()]
    if not entries:
        sys.exit(f"No trained agents found under {agents_root}.")
    print(f"Rendering {len(entries)} trained map(s): {[e['layout_id'] for e in entries]}")

    ncols = min(5, len(entries))
    nrows = (len(entries) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.2 * nrows),
                             facecolor=BG, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for k, e in enumerate(entries):
        env = PacmanEnv(layout_path=str(ROOT / e["file"]), num_ghosts=args.num_ghosts,
                        randomize_spawn=True, min_spawn_dist=3, max_steps=500)
        env.reset(seed=args.seed)
        gs = env.game_state
        ax = axes[k // ncols][k % ncols]
        ax.axis("on")
        render_map(ax, env.layout.walls, gs.food_mask, gs.pacman_pos,
                   list(gs.ghost_positions), title=f"{e['layout_id']}  ({e.get('food_count','?')} pellets)")
        env.close()

    fig.suptitle(f"RL-agent training maps ({args.pool_split} pool, {len(entries)} maps)",
                 color="white", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=BG)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
