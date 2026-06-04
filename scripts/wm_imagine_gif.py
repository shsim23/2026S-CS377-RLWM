"""Animated GIF of an open-loop IMAGINATION rollout vs ground truth.

For each example window (single-episode, on the given layout), warm the posterior
on a real context window, then roll the PRIOR forward with the real actions and
decode every step. Renders GT (left) vs imagine (right) frame-by-frame over the
full horizon into a GIF so the imagination flow is inspectable directly.

Usage
-----
    python scripts/wm_imagine_gif.py \
        --checkpoint checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt \
        --test-dataset rl_single_L0 --layout-id 0 --n-examples 3 --fps 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from world_model.dreamer import DreamerWorldModel, WorldModelConfig
from scripts.wm_eval_visualize import (
    SingleEpisodeReplay, rollout_for_viz, draw_frame, _entity_cell_errors,
    GRID_H, GRID_W,
)


def make_gif(wall_flat, preds, gts, rewards, out_path: Path, fps: int, title: str) -> None:
    walls = wall_flat.reshape(GRID_H, GRID_W) > 0.5
    H = len(preds)
    fig, (axg, axi) = plt.subplots(1, 2, figsize=(6.4, 3.4), facecolor="#111")

    def render(k: int):
        axg.clear(); axi.clear()
        gt_prev = gts[k - 1] if k >= 1 else None
        pr_prev = preds[k - 1] if k >= 1 else None
        draw_frame(axg, wall_flat, gts[k], prev_dyn=gt_prev, snap=True)
        draw_frame(axi, wall_flat, preds[k], prev_dyn=pr_prev, snap=True)
        err = _entity_cell_errors(preds[k], gts[k], walls)
        gl1 = np.mean(err["ghost_l1"]) if err["ghost_l1"] else 0.0
        r_pred, r_true = rewards[k]
        axg.set_title("ground truth", color="w", fontsize=11)
        axi.set_title("imagine", color="#ffd9b3", fontsize=11)
        fig.suptitle(f"{title}\nh={k + 1:2d}/{H}   pacΔ={err['pac_l1']}  ghΔ={gl1:.1f}   "
                     f"r={r_true:+.2f}  r̂={r_pred:+.2f}",
                     color="w", fontsize=10)
        return axg, axi

    anim = FuncAnimation(fig, render, frames=H, interval=1000 // max(fps, 1))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt")
    p.add_argument("--test-dataset", default="rl_single_L0")
    p.add_argument("--layout-id", type=int, default=0)
    p.add_argument("--data-root", default="data/replay")
    p.add_argument("--config", default="configs/world_model/dreamer_v3.yaml")
    p.add_argument("--out-dir", default=None,
                   help="Defaults to logs/wm_eval/<dataset>_layout<id>_imagine_gif.")
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    cfg = yaml.safe_load(open(ROOT / args.config))["world_model"]
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "logs" / "wm_eval" / f"{args.test_dataset}_layout{args.layout_id}_imagine_gif")
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute()
                    else args.checkpoint, map_location=device, weights_only=False)
    model = DreamerWorldModel(WorldModelConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"Loaded {args.checkpoint} (step {ck.get('step', '?')}) on {device}")

    window = cfg["context"] + cfg["k_step"]
    replay = SingleEpisodeReplay(str(ROOT / args.data_root / args.test_dataset),
                                 length=window, layout_id=args.layout_id, seed=0)
    print(f"layout_id={args.layout_id}: {replay.valid_starts.size} single-episode windows "
          f"(window={window})")

    title = f"{args.test_dataset} L{args.layout_id} (step {ck.get('step', '?')})"
    examples = list(replay.iter_eval_windows(args.n_examples, device=device, seed=cfg["eval_seed"]))
    written = []
    for ei, batch in enumerate(examples):
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        wall_flat, preds, gts, rewards = rollout_for_viz(
            model, batch, cfg["context"], cfg["k_step"], device)
        fname = out_dir / f"imagine_{ei:02d}.gif"
        make_gif(wall_flat, preds, gts, rewards, fname, args.fps, title + f"  [ex {ei}]")
        written.append(fname.name)
        print(f"  saved {fname.name} ({len(preds)} frames)")

    print(f"\nSaved → {out_dir}/")
    for f in written:
        print(f"  {f}")


if __name__ == "__main__":
    main()
