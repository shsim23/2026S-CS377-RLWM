"""Sweep intrinsic eval of EVERY DreamerV3 world-model checkpoint version on the
maps it was actually trained on (in-distribution, per layout_id).

The `rl_partial` run was trained on the layouts that already had a PPO agent
(layout_ids present in the dataset — 5 maps: train_000..004, NOT a single map and
NOT the full 30-layout pool). This script:

  1. discovers every checkpoint version in a directory (step_*.pt snapshots plus
     best.pt / latest.pt),
  2. for each version, evaluates each trained layout_id separately
     (`SingleLayoutReplay`, reusing the spec-§10 `evaluate()` metrics), and
  3. writes a per-(version, layout) metrics table (CSV + JSON) and a
     metric-vs-training-step plot averaged across layouts, so you can see which
     version is best and whether the model is still under-fitting.

Usage
-----
    # default: all versions in checkpoints/dreamer_wm/rl_partial, all trained maps
    python scripts/wm_eval_versions.py

    # explicit
    python scripts/wm_eval_versions.py \
        --ckpt-dir checkpoints/dreamer_wm/rl_partial --dataset rl_partial \
        --layout-ids 0 1 2 3 4 --viz
"""
from __future__ import annotations

import argparse
import csv
import json
import re
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
# Reuse the single-layout window sampler (and, for --viz, the renderers).
from scripts.wm_eval_visualize import (
    SingleLayoutReplay,
    rollout_for_viz,
    plot_kstep_curves,
    plot_rollout,
    snap_off_wall,
    _denorm,
    GRID_H,
    GRID_W,
)

# Scalar metrics we surface in the summary table (key -> short column header).
SUMMARY_KEYS = {
    "one_step/reward_mse": "1s_rew_mse",
    "one_step/recon_bin_acc": "1s_bin_acc",
    "one_step/recon_cont_mse": "1s_cont_mse",
    "one_step/cont_acc": "1s_cont_acc",
    "kstep/reward_mse_final": "ks_rew_mse_f",
    "kstep/recon_cont_mse_final": "ks_cont_mse_f",
    "kstep/cont_acc_final": "ks_cont_acc_f",
    "collapse/n_collapsed_groups": "collapsed",
    "collapse/mean_group_entropy": "ent",
}


def discover_versions(ckpt_dir: Path, include_aliases: bool) -> list[tuple[str, Path]]:
    """Return [(label, path)] ordered by training step; step_<N>.pt first, then
    best/latest aliases (labelled with their stored step)."""
    steps = []
    for p in ckpt_dir.glob("step_*.pt"):
        m = re.search(r"step_(\d+)\.pt$", p.name)
        if m:
            steps.append((int(m.group(1)), p))
    steps.sort()
    out = [(f"step_{n}", p) for n, p in steps]
    if include_aliases:
        for alias in ("latest.pt", "best.pt"):
            ap = ckpt_dir / alias
            if ap.exists():
                out.append((alias.replace(".pt", ""), ap))
    return out


@torch.no_grad()
def measure_legality(model, replay, context, horizon, n_windows, device, seed):
    """Open-loop rollout legality diagnostics for predicted PACMAN, per horizon:

      on_wall_rate[h]  — fraction of windows whose decoded pacman lands on a wall
                         cell (an impossible state the WM is hallucinating),
      cell_err_raw[h]  — L1 cell distance to ground-truth pacman (no snapping),
      cell_err_snap[h] — same, after snapping the prediction off walls.

    Returns dict of length-`horizon` numpy arrays (means over windows)."""
    H = horizon
    on_wall = np.zeros(H); err_raw = np.zeros(H); err_snap = np.zeros(H); cnt = np.zeros(H)
    for batch in replay.iter_eval_windows(n_windows, device=device, seed=seed):
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        wall_flat, preds, gts, _ = rollout_for_viz(model, batch, context, horizon, device)
        walls = wall_flat.reshape(GRID_H, GRID_W) > 0.5
        for k in range(len(preds)):
            pr, gt = preds[k], gts[k]
            pr_r, pr_c = _denorm(pr[1], GRID_H), _denorm(pr[0], GRID_W)
            gt_r, gt_c = _denorm(gt[1], GRID_H), _denorm(gt[0], GRID_W)
            sr, sc = snap_off_wall(pr_r, pr_c, walls)
            on_wall[k] += float(walls[pr_r, pr_c])
            err_raw[k] += abs(pr_r - gt_r) + abs(pr_c - gt_c)
            err_snap[k] += abs(sr - gt_r) + abs(sc - gt_c)
            cnt[k] += 1
    cnt = np.maximum(cnt, 1)
    return {"on_wall_rate": on_wall / cnt,
            "cell_err_raw": err_raw / cnt,
            "cell_err_snap": err_snap / cnt}


def load_model(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DreamerWorldModel(WorldModelConfig(**ckpt["cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, int(ckpt.get("step", -1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints/dreamer_wm/rl_partial",
                   help="Directory of checkpoint versions to sweep.")
    p.add_argument("--dataset", default="rl_partial",
                   help="Dataset whose layouts the WM was trained on (in-distribution).")
    p.add_argument("--layout-ids", type=int, nargs="+", default=None,
                   help="Layouts to eval. Default: every layout_id present in the dataset.")
    p.add_argument("--data-root", default="data/replay")
    p.add_argument("--config", default="configs/world_model/dreamer_v3.yaml")
    p.add_argument("--out-dir", default=None,
                   help="Default: logs/wm_eval/<ckpt_dir_name>_versions.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-aliases", action="store_true",
                   help="Skip best.pt / latest.pt; only step_*.pt snapshots.")
    p.add_argument("--viz", action="store_true",
                   help="Also render kstep curves + qualitative rollout for the best version.")
    p.add_argument("--viz-horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = p.parse_args()

    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    cfg = yaml.safe_load(open(ROOT / args.config))["world_model"]
    context, horizon = cfg["context"], cfg["k_step"]
    n_windows, eval_seed = cfg["n_eval_windows"], cfg["eval_seed"]
    window = context + cfg["seq_length"]

    ckpt_dir = ROOT / args.ckpt_dir if not Path(args.ckpt_dir).is_absolute() else Path(args.ckpt_dir)
    data_dir = ROOT / args.data_root / args.dataset
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "logs" / "wm_eval" / f"{ckpt_dir.name}_versions")
    out_dir.mkdir(parents=True, exist_ok=True)

    versions = discover_versions(ckpt_dir, include_aliases=not args.no_aliases)
    if not versions:
        sys.exit(f"No checkpoints found in {ckpt_dir}")

    if args.layout_ids is not None:
        layout_ids = args.layout_ids
    else:
        layout_ids = sorted(np.unique(np.load(data_dir / "layout_ids.npy")).tolist())

    print(f"ckpt_dir : {ckpt_dir}")
    print(f"dataset  : {data_dir}  (in-distribution / trained maps)")
    print(f"versions : {[v[0] for v in versions]}")
    print(f"layouts  : {layout_ids}")
    print(f"eval     : context={context} horizon={horizon} n_windows={n_windows}\n")

    # Pre-build one replay per layout (reused across all versions → sampler is
    # deterministic, so every version sees the identical eval windows).
    replays = {}
    for lid in layout_ids:
        r = SingleLayoutReplay(str(data_dir), length=window, layout_id=lid, seed=0)
        replays[lid] = r
        print(f"  layout_id={lid}: {r.valid_starts.size} valid eval windows")

    # results[label] = {"step": int, "per_layout": {lid: metrics}, "mean": {key: val}}
    results: dict[str, dict] = {}
    rows: list[dict] = []   # flat rows for CSV: one per (version, layout|MEAN)

    def _legality_cols(leg: dict) -> dict:
        return {"onwall_h1": float(leg["on_wall_rate"][0]),
                "onwall_hf": float(leg["on_wall_rate"][-1]),
                "onwall_mean": float(np.mean(leg["on_wall_rate"])),
                "pacerr_raw_hf": float(leg["cell_err_raw"][-1]),
                "pacerr_snap_hf": float(leg["cell_err_snap"][-1])}

    for label, path in versions:
        model, step = load_model(path, device)
        per_layout, per_layout_leg = {}, {}
        for lid in layout_ids:
            m = evaluate(model, replays[lid], context=context, horizon=horizon,
                         n_windows=n_windows, device=device, seed=eval_seed)
            leg = measure_legality(model, replays[lid], context, horizon,
                                   n_windows, device, eval_seed)
            per_layout[lid] = m
            per_layout_leg[lid] = leg
            rows.append({"version": label, "step": step, "layout": lid,
                         **{col: m.get(k, float("nan")) for k, col in SUMMARY_KEYS.items()},
                         **_legality_cols(leg)})
        # mean across layouts
        mean = {k: float(np.mean([per_layout[lid][k] for lid in layout_ids]))
                for k in SUMMARY_KEYS}
        leg_mean = {k: np.mean([per_layout_leg[lid][k] for lid in layout_ids], axis=0)
                    for k in ("on_wall_rate", "cell_err_raw", "cell_err_snap")}
        rows.append({"version": label, "step": step, "layout": "MEAN",
                     **{SUMMARY_KEYS[k]: mean[k] for k in SUMMARY_KEYS},
                     **_legality_cols(leg_mean)})
        results[label] = {"step": step, "per_layout": per_layout, "mean": mean,
                          "legality_mean": {k: v.tolist() for k, v in leg_mean.items()}}

        cols = " ".join(f"{SUMMARY_KEYS[k]}={mean[k]:.4f}" for k in SUMMARY_KEYS)
        lc = _legality_cols(leg_mean)
        print(f"[{label:>11} | step {step:>6}]  (mean over {len(layout_ids)} maps)  {cols}")
        print(f"{'':>22}pacman-on-wall: h1={lc['onwall_h1']:.3f} hfinal={lc['onwall_hf']:.3f} "
              f"mean={lc['onwall_mean']:.3f} | pac cell-err hfinal raw={lc['pacerr_raw_hf']:.2f} "
              f"snap={lc['pacerr_snap_hf']:.2f}")

    # ---- write CSV ----
    csv_path = out_dir / "version_sweep.csv"
    legality_cols = ["onwall_h1", "onwall_hf", "onwall_mean", "pacerr_raw_hf", "pacerr_snap_hf"]
    fieldnames = ["version", "step", "layout"] + list(SUMMARY_KEYS.values()) + legality_cols
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ---- write full JSON (incl. k-step curves) ----
    json_path = out_dir / "version_sweep.json"
    json_path.write_text(json.dumps(
        {"ckpt_dir": str(ckpt_dir), "dataset": args.dataset, "layout_ids": layout_ids,
         "results": results}, indent=2))

    # ---- plot key metrics vs training step (mean across layouts) ----
    # Only step_*.pt snapshots define a clean training-step axis.
    step_versions = [(lbl, results[lbl]) for lbl, _ in versions if lbl.startswith("step_")]
    if step_versions:
        steps = [r["step"] for _, r in step_versions]
        plot_specs = [
            ("kstep/reward_mse_final", "k-step reward MSE (final h)", "tab:red"),
            ("kstep/recon_cont_mse_final", "k-step dyn-recon MSE (final h)", "tab:blue"),
            ("one_step/cont_acc", "1-step continue acc", "tab:green"),
            ("one_step/recon_bin_acc", "1-step food/wall bin acc", "tab:purple"),
        ]
        fig, axes = plt.subplots(1, len(plot_specs), figsize=(5 * len(plot_specs), 4))
        for ax, (key, ylabel, color) in zip(axes, plot_specs):
            ys = [r["mean"][key] for _, r in step_versions]
            ax.plot(steps, ys, "-o", ms=4, color=color)
            ax.set_xlabel("training step"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
        fig.suptitle(f"WM version sweep — {ckpt_dir.name} (mean over {len(layout_ids)} trained maps)")
        fig.tight_layout()
        fig.savefig(out_dir / "metric_vs_step.png", dpi=120)
        plt.close(fig)

        # legality vs training step + on-wall rate vs horizon
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(steps, [r["legality_mean"]["on_wall_rate"][-1] for _, r in step_versions],
                     "-o", ms=4, color="tab:red", label="final horizon")
        axes[0].plot(steps, [np.mean(r["legality_mean"]["on_wall_rate"]) for _, r in step_versions],
                     "-o", ms=4, color="tab:orange", label="horizon mean")
        axes[0].set_xlabel("training step"); axes[0].set_ylabel("pacman-on-wall rate")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(steps, [r["legality_mean"]["cell_err_raw"][-1] for _, r in step_versions],
                     "-o", ms=4, color="tab:gray", label="raw")
        axes[1].plot(steps, [r["legality_mean"]["cell_err_snap"][-1] for _, r in step_versions],
                     "-o", ms=4, color="tab:blue", label="snapped off wall")
        axes[1].set_xlabel("training step"); axes[1].set_ylabel("pacman cell L1 err (final h)")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        for lbl, r in step_versions:           # on-wall vs horizon, per version
            curve = r["legality_mean"]["on_wall_rate"]
            axes[2].plot(range(1, len(curve) + 1), curve, "-", lw=1, alpha=0.8, label=lbl)
        axes[2].set_xlabel("open-loop horizon"); axes[2].set_ylabel("pacman-on-wall rate")
        axes[2].grid(alpha=0.3); axes[2].legend(fontsize=6, ncol=2)
        fig.suptitle(f"Pacman legality — {ckpt_dir.name} (mean over {len(layout_ids)} maps)")
        fig.tight_layout()
        fig.savefig(out_dir / "legality_vs_step.png", dpi=120)
        plt.close(fig)

    # ---- pick best version under two different criteria ----
    # Selection is over step_*.pt snapshots (best.pt/latest.pt are aliases of an
    # existing step, so they'd just duplicate a snapshot).
    snap_labels = [lbl for lbl, _ in versions if lbl.startswith("step_")] or list(results)
    best_1step = min(snap_labels, key=lambda l: results[l]["mean"]["one_step/reward_mse"])
    best_nstep = min(snap_labels, key=lambda l: results[l]["mean"]["kstep/reward_mse_final"])
    print(f"\nBest by 1-step reward MSE : {best_1step} (step {results[best_1step]['step']}) "
          f"-> {results[best_1step]['mean']['one_step/reward_mse']:.4f}")
    print(f"Best by k-step reward MSE : {best_nstep} (step {results[best_nstep]['step']}) "
          f"-> {results[best_nstep]['mean']['kstep/reward_mse_final']:.4f}")

    # ---- optional qualitative viz: render each distinct 'best' separately ----
    if args.viz:
        # tag -> label; de-dupes when the two criteria pick the same checkpoint.
        targets: dict[str, str] = {}
        if best_1step == best_nstep:
            targets[f"best_1step_and_nstep_{best_1step}"] = best_1step
        else:
            targets[f"best_1step_{best_1step}"] = best_1step
            targets[f"best_nstep_{best_nstep}"] = best_nstep

        for tag, label in targets.items():
            sub = out_dir / tag
            sub.mkdir(parents=True, exist_ok=True)
            model, step = load_model(dict(versions)[label], device)
            for lid in layout_ids:
                m = results[label]["per_layout"][lid]
                title = f"{ckpt_dir.name} {label} (step {step}) — layout_id={lid}"
                plot_kstep_curves(m, title, sub / f"kstep_curves_layout{lid}.png")
                batch = next(replays[lid].iter_eval_windows(1, device=device, seed=eval_seed))
                batch = {k: v.unsqueeze(0) for k, v in batch.items()}
                wall_flat, preds, gts, rewards = rollout_for_viz(model, batch, context, horizon, device)
                plot_rollout(wall_flat, preds, gts, rewards, args.viz_horizons, title,
                             sub / f"rollout_layout{lid}.png")
            print(f"  rendered {label} -> {sub}/")

    print(f"\nSaved -> {out_dir}/")
    for f in ("version_sweep.csv", "version_sweep.json", "metric_vs_step.png",
              "legality_vs_step.png"):
        if (out_dir / f).exists():
            print(f"  {f}")


if __name__ == "__main__":
    main()
