"""Measure entity-position quality across world-model checkpoints (e.g. a beta_cont
sweep) on a single layout, and plot the 1-step-position vs open-loop-rollout tradeoff.

For each checkpoint, on `layout_id` of `dataset` (in-distribution):
  * teacher-forced posterior recon (== eval.py "one_step" definition): decode the
    dynamic state from the posterior latent and measure PACMAN / GHOST cell-L1
    error and the rate of decoding onto a wall (raw + snapped-off-wall);
  * open-loop k-step rollout metrics via the standard `evaluate()` (reward MSE,
    dyn-recon MSE, continue acc at the final horizon).

Usage
-----
    # beta_cont sweep on layout 0 (default checkpoint name template)
    python scripts/wm_measure_position.py --bc-values 1 5 10 15 20 25 --layout-id 0

    # explicit checkpoints
    python scripts/wm_measure_position.py \
        --checkpoints base=checkpoints/dreamer_wm/rl_partial/best.pt --layout-id 0
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

from world_model.dreamer import DreamerWorldModel, WorldModelConfig, evaluate
from world_model.dreamer.replay import SequenceReplay
from world_model.dreamer.rssm import extract_dynamic
from scripts.wm_eval_visualize import _denorm, snap_off_wall, GRID_H, GRID_W

PAC = slice(0, 2)
GHOST = slice(2, 18)


@torch.no_grad()
def measure_positions(model, replay, context: int, n_windows: int, device, seed: int = 0) -> dict:
    """Teacher-forced posterior-recon position errors (cell-L1) and on-wall rates."""
    pac_err, pac_wall, pac_wall_snap = [], [], []
    gh_err, gh_wall = [], []
    for b in replay.iter_eval_windows(n_windows, device=device, seed=seed):
        b = {k: v.unsqueeze(0).to(device) for k, v in b.items()}
        out = model.observe(b["states"], b["actions"], b["is_first"])
        recon = model.reconstruct_with_pos(
            out["recon"], out.get("position_logits"))[0, context:].cpu().numpy()      # (T,460)
        gt = extract_dynamic(b["states"])[0, context:].cpu().numpy()
        walls = b["states"][0, 0, 459:900].cpu().numpy().reshape(GRID_H, GRID_W) > 0.5
        for pr, g in zip(recon, gt):
            pr_r, pr_c = _denorm(pr[1], GRID_H), _denorm(pr[0], GRID_W)
            gt_r, gt_c = _denorm(g[1], GRID_H), _denorm(g[0], GRID_W)
            sr, sc = snap_off_wall(pr_r, pr_c, walls)
            pac_err.append(abs(pr_r - gt_r) + abs(pr_c - gt_c))
            pac_wall.append(float(walls[pr_r, pr_c]))
            pac_wall_snap.append(abs(sr - gt_r) + abs(sc - gt_c))
            prg, gg = pr[GHOST].reshape(4, 4), g[GHOST].reshape(4, 4)
            for i in range(4):
                if gg[i, 2] > 0.5 and gg[i, 3] > 0.5:           # alive & valid in GT
                    r, c = _denorm(prg[i, 1], GRID_H), _denorm(prg[i, 0], GRID_W)
                    rr, cc = _denorm(gg[i, 1], GRID_H), _denorm(gg[i, 0], GRID_W)
                    gh_err.append(abs(r - rr) + abs(c - cc))
                    gh_wall.append(float(walls[r, c]))
    return {
        "pac_cell_l1": float(np.mean(pac_err)),
        "pac_on_wall": float(np.mean(pac_wall)),
        "pac_cell_l1_snap": float(np.mean(pac_wall_snap)),
        "ghost_cell_l1": float(np.mean(gh_err)),
        "ghost_on_wall": float(np.mean(gh_wall)),
    }


def load_model(path: Path, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = DreamerWorldModel(WorldModelConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, int(ck.get("step", -1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bc-values", type=float, nargs="+", default=None,
                   help="beta_cont values; checkpoints from --name-template.")
    p.add_argument("--name-template", default="checkpoints/dreamer_wm/rl_partial_L0_fast_bc{bc}/latest.pt")
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="Explicit label=path pairs (overrides --bc-values).")
    p.add_argument("--dataset", default="rl_partial")
    p.add_argument("--layout-id", type=int, default=0)
    p.add_argument("--data-root", default="data/replay")
    p.add_argument("--config", default="configs/world_model/dreamer_v3.yaml")
    p.add_argument("--n-windows", type=int, default=128)
    p.add_argument("--out-dir", default="logs/wm_eval/bc_sweep_L0")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    cfg = yaml.safe_load(open(ROOT / args.config))["world_model"]
    context, horizon = cfg["context"], cfg["k_step"]
    window = context + cfg["seq_length"]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build (label, x, path) entries. x = beta_cont for plotting, else index.
    entries = []
    if args.checkpoints:
        for i, kv in enumerate(args.checkpoints):
            label, path = kv.split("=", 1)
            entries.append((label, float(i), Path(path)))
    else:
        bcs = args.bc_values or [1, 5, 10, 15, 20, 25]
        for bc in bcs:
            bc_s = int(bc) if float(bc).is_integer() else bc
            entries.append((f"bc={bc_s}", float(bc),
                            ROOT / args.name_template.format(bc=bc_s)))

    data_dir = ROOT / args.data_root / args.dataset
    replay = SequenceReplay(str(data_dir), length=window, seed=0, layout_id=args.layout_id)
    print(f"dataset={data_dir} layout_id={args.layout_id} | {replay.valid_starts.size} windows "
          f"| n_windows={args.n_windows}\n")

    rows = []
    for label, x, path in entries:
        if not path.exists():
            print(f"[skip] {label}: missing {path}")
            continue
        model, step = load_model(path, device)
        pos = measure_positions(model, replay, context, args.n_windows, device, seed=cfg["eval_seed"])
        ev = evaluate(model, replay, context=context, horizon=horizon,
                      n_windows=cfg["n_eval_windows"], device=device, seed=cfg["eval_seed"])
        row = {"label": label, "beta_cont": x, "step": step, **pos,
               "kstep_rew_mse_f": ev["kstep/reward_mse_final"],
               "kstep_cont_mse_f": ev["kstep/recon_cont_mse_final"],
               "kstep_cont_acc_f": ev["kstep/cont_acc_final"],
               "one_step_rew_mse": ev["one_step/reward_mse"]}
        rows.append(row)
        print(f"[{label:>7} | step {step}] PACMAN L1={pos['pac_cell_l1']:.2f} on-wall={pos['pac_on_wall']:.3f}"
              f" | GHOST L1={pos['ghost_cell_l1']:.2f} on-wall={pos['ghost_on_wall']:.3f}"
              f" || kstep rew_mse(f)={ev['kstep/reward_mse_final']:.2f} cont_acc(f)={ev['kstep/cont_acc_final']:.3f}")

    if not rows:
        sys.exit("No checkpoints measured.")

    # CSV + JSON
    with open(out_dir / "position_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (out_dir / "position_sweep.json").write_text(json.dumps(rows, indent=2))

    # Plot: position quality (left axis) vs rollout stability (right axis) vs beta_cont
    xs = [r["beta_cont"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(xs, [r["pac_cell_l1"] for r in rows], "-o", color="tab:orange", label="pacman")
    axes[0].plot(xs, [r["ghost_cell_l1"] for r in rows], "-o", color="tab:red", label="ghost")
    axes[0].set_ylabel("1-step cell-L1 error"); axes[0].set_xlabel("beta_cont"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(xs, [r["pac_on_wall"] for r in rows], "-o", color="tab:orange", label="pacman")
    axes[1].plot(xs, [r["ghost_on_wall"] for r in rows], "-o", color="tab:red", label="ghost")
    axes[1].set_ylabel("1-step on-wall rate"); axes[1].set_xlabel("beta_cont"); axes[1].legend(); axes[1].grid(alpha=.3)
    axes[2].plot(xs, [r["kstep_rew_mse_f"] for r in rows], "-o", color="tab:blue")
    axes[2].set_ylabel("k-step reward MSE (final h)"); axes[2].set_xlabel("beta_cont"); axes[2].grid(alpha=.3)
    fig.suptitle(f"beta_cont sweep — layout {args.layout_id}: 1-step position vs open-loop rollout")
    fig.tight_layout()
    fig.savefig(out_dir / "position_sweep.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved -> {out_dir}/ (position_sweep.csv/json/png)")


if __name__ == "__main__":
    main()
