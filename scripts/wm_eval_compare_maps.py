"""Compare a world model's imagination on a TRAINED map vs an UNSEEN map (cross-map
generalization), with the numbers and the visuals in one organized output folder.

For each target layout (same dataset, single-episode windows): run the imagination
error breakdown (scripts.wm_imagine_analysis.analyze), render a few qualitative
GT-vs-imagine rollouts, and a side-by-side GIF. Then emit ONE combined figure that
overlays the per-horizon curves of all targets, plus a maze-comparison figure.

Usage
-----
    python scripts/wm_eval_compare_maps.py \
        --checkpoint checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt \
        --dataset rl_partial --layouts 0 1 --labels trained:L0 unseen:L1 \
        --n-windows 400 --n-examples 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from world_model.dreamer import DreamerWorldModel, WorldModelConfig
from scripts.wm_eval_visualize import (
    SingleEpisodeReplay, rollout_for_viz, plot_rollout, draw_frame, GRID_H, GRID_W,
)
from scripts.wm_imagine_analysis import analyze
from scripts.wm_imagine_gif import make_gif

KEY_H = [1, 4, 8, 16, 24, 32]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt")
    p.add_argument("--dataset", default="rl_partial")
    p.add_argument("--layouts", type=int, nargs="+", default=[0, 1])
    p.add_argument("--labels", nargs="+", default=None,
                   help="One label per layout (e.g. trained:L0 unseen:L1). Default: L<id>.")
    p.add_argument("--data-root", default="data/replay")
    p.add_argument("--config", default="configs/world_model/dreamer_v3.yaml")
    p.add_argument("--n-windows", type=int, default=400)
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--out-dir", default="logs/wm_eval/compare_maps")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    cfg = yaml.safe_load(open(ROOT / args.config))["world_model"]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = args.labels or [f"L{l}" for l in args.layouts]
    assert len(labels) == len(args.layouts), "need one --labels entry per layout"

    ck = torch.load(ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute()
                    else args.checkpoint, map_location=device, weights_only=False)
    model = DreamerWorldModel(WorldModelConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    step = ck.get("step", "?")
    print(f"Loaded {args.checkpoint} (step {step}); trained on a SINGLE map (layout 0).\n")

    context, horizon = cfg["context"], cfg["k_step"]
    window = context + horizon
    results = {}
    wall_imgs = {}
    for lay, lab in zip(args.layouts, labels):
        replay = SingleEpisodeReplay(str(ROOT / args.data_root / args.dataset),
                                     length=window, layout_id=lay, seed=0)
        nW = min(args.n_windows, replay.valid_starts.size)
        print(f"[{lab}] layout {lay}: {replay.valid_starts.size} windows; analyzing {nW} ...")
        r = analyze(model, replay, context, horizon, nW, device, cfg["eval_seed"])
        results[lab] = r
        # qualitative rollouts + a GIF
        examples = list(replay.iter_eval_windows(args.n_examples, device=device, seed=cfg["eval_seed"]))
        for ei, batch in enumerate(examples):
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}
            wall_flat, preds, gts, rewards = rollout_for_viz(model, batch, context, horizon, device)
            plot_rollout(wall_flat, preds, gts, rewards, KEY_H,
                         f"{lab} (layout {lay}, step {step})",
                         out_dir / f"{lab.replace(':','_')}_rollout_{ei:02d}.png")
            if ei == 0:
                make_gif(wall_flat, preds, gts, rewards,
                         out_dir / f"{lab.replace(':','_')}_imagine.gif", args.fps,
                         f"{lab} (layout {lay})")
                wall_imgs[lab] = wall_flat

    # --- combined numerical table ---
    print(f"\n=== Trained vs unseen map — imagination error at key horizons ===")
    print("metric".ljust(22) + "".join(f"{lab:>16}" for lab in labels))
    for h in KEY_H:
        i = h - 1
        print(f"-- h={h} --")
        for name, key, fmt in [
            ("pac cell-L1 (mean)", "pac_l1_mean", "{:.2f}"),
            ("pac exact %", "pac_exact", "{:.0%}"),
            ("ghost cell-L1 (mean)", "gh_l1_mean", "{:.2f}"),
            ("food IoU", "food_iou", "{:.3f}"),
            ("reward |err| (mean)", "rew_abs_mean", "{:.2f}"),
        ]:
            cells = "".join(f"{fmt.format(results[lab][key][i]):>16}" for lab in labels)
            print(f"  {name:<20}{cells}")

    # --- combined CSV ---
    with open(out_dir / "compare_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        metrics = ["pac_l1_mean", "pac_exact", "pac_l1_persist", "gh_l1_mean",
                   "gh_l1_persist", "food_iou", "rew_abs_mean", "rew_abs_p90"]
        w.writerow(["horizon"] + [f"{lab}/{m}" for lab in labels for m in metrics])
        for k in range(horizon):
            rowv = [k + 1]
            for lab in labels:
                rowv += [float(results[lab][m][k]) for m in metrics]
            w.writerow(rowv)

    # --- combined comparison figure ---
    hs = np.arange(1, horizon + 1)
    colors = ["tab:green", "tab:red", "tab:purple", "tab:brown"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    for ci, lab in enumerate(labels):
        r = results[lab]; c = colors[ci % len(colors)]
        axes[0].plot(hs, r["pac_l1_mean"], "-o", ms=3, color=c, label=lab)
        axes[0].plot(hs, r["pac_l1_persist"], ":", color=c, alpha=.4)
        axes[1].plot(hs, r["gh_l1_mean"], "-o", ms=3, color=c, label=lab)
        axes[1].plot(hs, r["gh_l1_persist"], ":", color=c, alpha=.4)
        axes[2].plot(hs, 100 * r["pac_exact"], "-o", ms=3, color=c, label=lab)
        axes[3].plot(hs, r["rew_abs_mean"], "-o", ms=3, color=c, label=lab)
    axes[0].set_title("pacman cell-L1 (solid=model, dotted=persistence)")
    axes[1].set_title("ghost cell-L1 (solid=model, dotted=persistence)")
    axes[2].set_title("pacman exact-cell %")
    axes[3].set_title("reward |err| (mean)")
    for ax in axes:
        ax.set_xlabel("imagine horizon"); ax.grid(alpha=.3); ax.legend()
    fig.suptitle(f"Imagination: trained vs unseen map — {args.dataset} (model step {step}, trained on layout 0)")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_curves.png", dpi=120)
    plt.close(fig)

    # --- maze comparison figure ---
    if wall_imgs:
        fig, axs = plt.subplots(1, len(wall_imgs), figsize=(3.2 * len(wall_imgs), 3.4), facecolor="#111")
        if len(wall_imgs) == 1:
            axs = [axs]
        for ax, (lab, wf) in zip(axs, wall_imgs.items()):
            zeros = np.zeros(460, dtype=np.float32)
            zeros[2:18] = 0.0  # no entities; just show the maze + (empty) dynamic
            draw_frame(ax, wf, zeros, snap=False)
            ax.set_title(lab, color="w")
        fig.suptitle("maze layouts compared", color="w")
        fig.tight_layout()
        fig.savefig(out_dir / "maps.png", dpi=120, facecolor="#111")
        plt.close(fig)

    (out_dir / "summary.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "step": step, "dataset": args.dataset,
         "layouts": args.layouts, "labels": labels,
         "results": {lab: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in results[lab].items()} for lab in labels}}, indent=2))

    print(f"\nSaved → {out_dir}/")
    for f in sorted(p.name for p in out_dir.iterdir()):
        print(f"  {f}")


if __name__ == "__main__":
    main()
