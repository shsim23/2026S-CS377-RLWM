"""Numerical breakdown of WHERE open-loop imagination error accumulates / fails.

On single-episode windows of one layout, warm the posterior on the context and roll
the PRIOR forward `horizon` steps with the real actions. Per horizon, report:

  * pacman / ghost cell-L1: mean, median, p90  (tail vs typical);
  * exact-cell hit-rate and within-1-cell rate  (a concrete "still correct" metric);
  * marginal growth Δ per step  (where the curve accelerates);
  * a PERSISTENCE baseline (entity frozen at its context-end cell) — model error far
    below persistence ⇒ the dynamics prior is genuinely tracking motion; model ≈
    persistence ⇒ it is just copying; model flat while persistence keeps rising ⇒
    the residual is the stochastic ceiling (e.g. ε-random ghosts), not divergence;
  * food precision/recall/IoU and reward |err| per horizon.

Usage
-----
    python scripts/wm_imagine_analysis.py \
        --checkpoint checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt \
        --test-dataset rl_single_L0 --layout-id 0 --n-windows 512
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
from world_model.dreamer.rssm import extract_dynamic
from scripts.wm_eval_visualize import (
    SingleEpisodeReplay, rollout_for_viz, _denorm, GRID_H, GRID_W, GHOST_SLICE, FOOD_SLICE,
)


def _pac_cell(dyn):
    return _denorm(dyn[1], GRID_H), _denorm(dyn[0], GRID_W)


def _ghost_cells(dyn):
    """List of (slot, row, col) for ghosts that are alive & valid in `dyn`."""
    g = dyn[GHOST_SLICE].reshape(4, 4)
    out = []
    for i in range(4):
        if g[i, 2] > 0.5 and g[i, 3] > 0.5:
            out.append((i, _denorm(g[i, 1], GRID_H), _denorm(g[i, 0], GRID_W)))
    return out


@torch.no_grad()
def analyze(model, replay, context, horizon, n_windows, device, seed=0):
    H = horizon
    pac = np.full((n_windows, H), np.nan)             # model pacman cell-L1
    pac_p = np.full((n_windows, H), np.nan)           # persistence pacman cell-L1
    gh = [[] for _ in range(H)]; gh_p = [[] for _ in range(H)]
    food_tp = np.zeros(H); food_fp = np.zeros(H); food_fn = np.zeros(H)
    foodp_tp = np.zeros(H); foodp_fp = np.zeros(H); foodp_fn = np.zeros(H)  # persistence food
    rew = np.full((n_windows, H), np.nan)
    w = 0
    for batch in replay.iter_eval_windows(n_windows, device=device, seed=seed):
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        states = batch["states"]
        ref = extract_dynamic(states[:, context - 1])[0].cpu().numpy()   # context-end GT (persistence)
        rp_pac = _pac_cell(ref)
        rp_gh = {i: (r, c) for i, r, c in _ghost_cells(ref)}
        ref_food = ref[FOOD_SLICE] > 0.5
        wall_flat, preds, gts, rewards = rollout_for_viz(model, batch, context, horizon, device)
        H_eff = len(preds)
        for k in range(H_eff):
            pr, gt = preds[k], gts[k]
            # pacman
            ppr, pgt = _pac_cell(pr), _pac_cell(gt)
            pac[w, k] = abs(ppr[0] - pgt[0]) + abs(ppr[1] - pgt[1])
            pac_p[w, k] = abs(rp_pac[0] - pgt[0]) + abs(rp_pac[1] - pgt[1])
            # ghosts (score every GT-alive&valid slot; match by slot)
            pr_g = {i: (r, c) for i, r, c in _ghost_cells(pr)}
            for i, gr, gc in _ghost_cells(gt):
                if i in pr_g:
                    gh[k].append(abs(pr_g[i][0] - gr) + abs(pr_g[i][1] - gc))
                if i in rp_gh:
                    gh_p[k].append(abs(rp_gh[i][0] - gr) + abs(rp_gh[i][1] - gc))
            # food (precision/recall vs GT)
            pf = pr[FOOD_SLICE] > 0.5; gf = gt[FOOD_SLICE] > 0.5
            food_tp[k] += np.sum(pf & gf); food_fp[k] += np.sum(pf & ~gf); food_fn[k] += np.sum(~pf & gf)
            foodp_tp[k] += np.sum(ref_food & gf); foodp_fp[k] += np.sum(ref_food & ~gf); foodp_fn[k] += np.sum(~ref_food & gf)
            # reward
            rew[w, k] = abs(rewards[k][0] - rewards[k][1])
        w += 1
    pac, pac_p, rew = pac[:w], pac_p[:w], rew[:w]

    def col_stats(arr2d):
        return (np.nanmean(arr2d, 0), np.nanmedian(arr2d, 0),
                np.nanpercentile(arr2d, 90, 0))

    def list_stats(lst):
        m = [float(np.mean(x)) if x else 0.0 for x in lst]
        med = [float(np.median(x)) if x else 0.0 for x in lst]
        p90 = [float(np.percentile(x, 90)) if x else 0.0 for x in lst]
        return np.array(m), np.array(med), np.array(p90)

    pac_m, pac_med, pac_p90 = col_stats(pac)
    pacP_m = np.nanmean(pac_p, 0)
    gh_m, gh_med, gh_p90 = list_stats(gh)
    ghP_m, _, _ = list_stats(gh_p)
    pac_exact = np.nanmean(pac == 0, 0); pac_le1 = np.nanmean(pac <= 1, 0)
    gh_exact = np.array([np.mean(np.array(x) == 0) if x else 0.0 for x in gh])
    gh_le1 = np.array([np.mean(np.array(x) <= 1) if x else 0.0 for x in gh])
    food_iou = food_tp / np.maximum(food_tp + food_fp + food_fn, 1)
    food_prec = food_tp / np.maximum(food_tp + food_fp, 1)
    food_rec = food_tp / np.maximum(food_tp + food_fn, 1)
    foodP_iou = foodp_tp / np.maximum(foodp_tp + foodp_fp + foodp_fn, 1)
    rew_m, rew_med, rew_p90 = col_stats(rew)

    return {
        "n_windows": int(w), "horizon": H,
        "pac_l1_mean": pac_m, "pac_l1_med": pac_med, "pac_l1_p90": pac_p90,
        "pac_l1_persist": pacP_m, "pac_exact": pac_exact, "pac_le1": pac_le1,
        "gh_l1_mean": gh_m, "gh_l1_med": gh_med, "gh_l1_p90": gh_p90,
        "gh_l1_persist": ghP_m, "gh_exact": gh_exact, "gh_le1": gh_le1,
        "food_iou": food_iou, "food_prec": food_prec, "food_rec": food_rec,
        "food_iou_persist": foodP_iou,
        "rew_abs_mean": rew_m, "rew_abs_med": rew_med, "rew_abs_p90": rew_p90,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt")
    p.add_argument("--test-dataset", default="rl_single_L0")
    p.add_argument("--layout-id", type=int, default=0)
    p.add_argument("--data-root", default="data/replay")
    p.add_argument("--config", default="configs/world_model/dreamer_v3.yaml")
    p.add_argument("--n-windows", type=int, default=512)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    cfg = yaml.safe_load(open(ROOT / args.config))["world_model"]
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "logs" / "wm_eval" / f"{args.test_dataset}_layout{args.layout_id}_imagine_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute()
                    else args.checkpoint, map_location=device, weights_only=False)
    model = DreamerWorldModel(WorldModelConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    window = cfg["context"] + cfg["k_step"]
    replay = SingleEpisodeReplay(str(ROOT / args.data_root / args.test_dataset),
                                 length=window, layout_id=args.layout_id, seed=0)
    print(f"Loaded {args.checkpoint} (step {ck.get('step', '?')}); "
          f"{replay.valid_starts.size} single-episode windows; analyzing {args.n_windows}.\n")

    r = analyze(model, replay, cfg["context"], cfg["k_step"], args.n_windows, device, cfg["eval_seed"])
    H = r["horizon"]

    # --- console table ---
    print(f"=== Imagination error breakdown ({r['n_windows']} single-episode windows) ===")
    print(f"{'h':>3} | {'pac mean':>8} {'med':>4} {'p90':>4} {'exact%':>6} {'≤1%':>5} {'persist':>7} "
          f"| {'gh mean':>7} {'p90':>4} {'exact%':>6} {'persist':>7} | {'foodIoU':>7} | {'rew|e|':>6} {'p90':>5}")
    for k in range(H):
        print(f"{k+1:>3} | {r['pac_l1_mean'][k]:>8.2f} {r['pac_l1_med'][k]:>4.0f} {r['pac_l1_p90'][k]:>4.0f} "
              f"{100*r['pac_exact'][k]:>6.1f} {100*r['pac_le1'][k]:>5.1f} {r['pac_l1_persist'][k]:>7.2f} "
              f"| {r['gh_l1_mean'][k]:>7.2f} {r['gh_l1_p90'][k]:>4.0f} {100*r['gh_exact'][k]:>6.1f} "
              f"{r['gh_l1_persist'][k]:>7.2f} | {r['food_iou'][k]:>7.3f} | {r['rew_abs_mean'][k]:>6.2f} {r['rew_abs_p90'][k]:>5.2f}")

    # marginal growth (per-step Δ of mean cell-L1)
    pac_d = np.diff(r["pac_l1_mean"], prepend=0.0)
    gh_d = np.diff(r["gh_l1_mean"], prepend=0.0)
    print("\nMarginal Δ (mean cell-L1 added per step):")
    print("  pacman max Δ at h=%d (%.2f); ghost max Δ at h=%d (%.2f)"
          % (int(pac_d.argmax()) + 1, pac_d.max(), int(gh_d.argmax()) + 1, gh_d.max()))

    # --- CSV ---
    keys = [k for k in r if isinstance(r[k], np.ndarray)]
    with open(out_dir / "imagine_analysis.csv", "w", newline="") as f:
        wcsv = csv.writer(f); wcsv.writerow(["horizon"] + keys)
        for k in range(H):
            wcsv.writerow([k + 1] + [float(r[key][k]) for key in keys])
    (out_dir / "imagine_analysis.json").write_text(json.dumps(
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}, indent=2))

    # --- plots: model vs persistence (the key diagnostic) ---
    hs = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(hs, r["pac_l1_mean"], "-o", ms=3, color="tab:orange", label="model mean")
    axes[0].plot(hs, r["pac_l1_p90"], "--", color="tab:orange", alpha=.6, label="model p90")
    axes[0].plot(hs, r["pac_l1_persist"], ":", color="gray", label="persistence")
    axes[0].set_title("pacman cell-L1"); axes[0].legend()
    axes[1].plot(hs, r["gh_l1_mean"], "-o", ms=3, color="tab:red", label="model mean")
    axes[1].plot(hs, r["gh_l1_p90"], "--", color="tab:red", alpha=.6, label="model p90")
    axes[1].plot(hs, r["gh_l1_persist"], ":", color="gray", label="persistence")
    axes[1].set_title("ghost cell-L1"); axes[1].legend()
    axes[2].plot(hs, 100 * r["pac_exact"], "-o", ms=3, color="tab:orange", label="pacman exact %")
    axes[2].plot(hs, 100 * r["pac_le1"], "--", color="tab:orange", alpha=.6, label="pacman ≤1 %")
    axes[2].plot(hs, 100 * r["gh_exact"], "-o", ms=3, color="tab:red", label="ghost exact %")
    axes[2].set_title("exact-cell hit rate (%)"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("imagine horizon"); ax.grid(alpha=.3)
    fig.suptitle(f"Imagination error breakdown — {args.test_dataset} L{args.layout_id} (step {ck.get('step','?')})")
    fig.tight_layout()
    fig.savefig(out_dir / "imagine_analysis.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved → {out_dir}/ (imagine_analysis.csv/json/png)")


if __name__ == "__main__":
    main()
