"""Single-test-map evaluation + visualization for a DreamerV3 world-model checkpoint.

Restricts evaluation to ONE held-out layout (by `layout_id`) from a test-pool
dataset, computes the intrinsic metrics (spec §10), and renders:

  1. `metrics.json`          — one-step + k-step aggregate metrics for the map.
  2. `kstep_curves.png`      — reward-MSE / recon-MSE / continue-accuracy vs horizon.
  3. `rollout_qualitative.png` — ground-truth vs open-loop-predicted game grids at a
                                 ladder of horizons (warm the posterior on a real
                                 context window, then roll the PRIOR forward with the
                                 real actions and decode at each step).

Usage
-----
    python scripts/wm_eval_visualize.py \
        --checkpoint checkpoints/dreamer_wm/main/best.pt \
        --test-dataset main_test --layout-id 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from world_model.dreamer import DreamerWorldModel, WorldModelConfig, SequenceReplay, evaluate
from world_model.dreamer.nn import OneHotCategoricalST
from world_model.dreamer.rssm import extract_dynamic

GRID_H = GRID_W = 21          # MAX_GRID_H / MAX_GRID_W (pacman_env.constants)
PAC_SLICE = slice(0, 2)
GHOST_SLICE = slice(2, 18)
FOOD_SLICE = slice(18, 459)   # within the 460-d DYNAMIC vector
POWER_IDX = 459


# --------------------------------------------------------------------------- #
class SingleLayoutReplay(SequenceReplay):
    """SequenceReplay restricted to windows lying entirely inside one layout_id.

    Reuses the parent sampling API (`iter_eval_windows`) so `evaluate()` works
    unchanged, but only emits windows whose every step shares `layout_id`.
    """

    def __init__(self, dataset_dir: str, length: int, layout_id: int, seed=None):
        super().__init__(dataset_dir, length, seed)
        same = (self.layout_ids == layout_id).astype(np.int64)
        # prefix sums → a length-L window [s, s+L) is all-same iff it sums to L.
        csum = np.concatenate([[0], np.cumsum(same)])
        win_sum = csum[self.length:] - csum[: self.N - self.length + 1]
        self.valid_starts = np.flatnonzero(win_sum == self.length)
        if self.valid_starts.size == 0:
            raise ValueError(
                f"No length-{self.length} window fits inside layout_id={layout_id} "
                f"(it has too few contiguous steps)."
            )

    def iter_eval_windows(self, n_windows: int, device=None, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = min(n_windows, self.valid_starts.size)
        starts = rng.choice(self.valid_starts, size=n, replace=False)
        for s in np.sort(starts):
            idx = np.arange(s, s + self.length)
            yield {
                "states": torch.from_numpy(np.asarray(self.states)[idx]).float(),
                "actions": torch.from_numpy(self.actions[idx]).long(),
                "rewards": torch.from_numpy(self.rewards[idx]).float(),
                "continues": torch.from_numpy(self.continues[idx]).float(),
                "is_first": torch.from_numpy(self.is_first[idx]),
            }


# --------------------------------------------------------------------------- #
def _denorm(coord: float, span: int = GRID_W) -> int:
    """Inverse of pacman_env.state._normalize: [-1,1] → integer cell index.

    Clamped to the valid grid: pacman/ghosts ALWAYS exist, so a predicted coord
    that drifts slightly outside [-1,1] (common for entities near the maze edge in
    open-loop rollout) is snapped to the boundary cell rather than dropped."""
    return int(np.clip(round((float(coord) + 1.0) / 2.0 * (span - 1)), 0, span - 1))


def render_grid(wall_flat: np.ndarray, dyn: np.ndarray) -> np.ndarray:
    """Render one game frame to an (H, W) integer label image.

    Labels: 0 empty, 1 wall, 2 food, 3 ghost, 4 pacman (pacman drawn last/on top).
    `wall_flat` is the static 441-d wall mask; `dyn` is the 460-d dynamic vector.
    """
    img = np.zeros((GRID_H, GRID_W), dtype=np.int64)
    img[wall_flat.reshape(GRID_H, GRID_W) > 0.5] = 1

    food = dyn[FOOD_SLICE].reshape(GRID_H, GRID_W)
    img[(food > 0.5) & (img == 0)] = 2

    ghosts = dyn[GHOST_SLICE].reshape(4, 4)
    for gx, gy, alive, valid in ghosts:
        if valid > 0.5 and alive > 0.5:   # invalid slot / eaten ghost is legitimately absent
            img[_denorm(gy, GRID_H), _denorm(gx, GRID_W)] = 3

    img[_denorm(dyn[1], GRID_H), _denorm(dyn[0], GRID_W)] = 4
    return img


_CMAP = ListedColormap(["#000000", "#1a1aff", "#ffb8b8", "#ff3030", "#ffe000"])  # empty/wall/food/ghost/pac


def plot_kstep_curves(metrics: dict, title: str, out_path: Path) -> None:
    specs = [
        ("kstep/reward_mse_curve", "reward MSE (raw)", "tab:red"),
        ("kstep/recon_cont_mse_curve", "dynamic-state recon MSE", "tab:blue"),
        ("kstep/cont_acc_curve", "continue accuracy", "tab:green"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, ylabel, color) in zip(axes, specs):
        curve = metrics[key]
        ax.plot(range(1, len(curve) + 1), curve, "-o", ms=3, color=color)
        ax.set_xlabel("open-loop horizon (steps)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def rollout_for_viz(model, batch, context, horizon, device):
    """Warm posterior on [0:context], then open-loop the prior `horizon` steps with
    the real actions. Returns per-horizon predicted & ground-truth dynamic states."""
    states = batch["states"].to(device)            # (1, L, 901)
    actions = batch["actions"].to(device)
    L = states.shape[1]
    horizon = min(horizon, L - context)

    e = model.embed_layout(states)
    warm = model.observe(states[:, :context], actions[:, :context], batch["is_first"][:, :context].to(device))
    h, z = warm["h"][:, -1], warm["z"][:, -1]

    preds, gts, rewards = [], [], []
    for k in range(horizon):
        t = context + k
        h = model.seq(h, model._flat(z), actions[:, t], e[:, t])
        z = OneHotCategoricalST(model.prior(h), model.cfg.unimix).sample_st()
        z_flat = model._flat(z)
        recon = model.decoder.reconstruct(model.decoder(h, z_flat))   # (1, 460)
        preds.append(recon[0].cpu().numpy())
        gts.append(extract_dynamic(states[:, t])[0].cpu().numpy())
        rewards.append((
            model.reward_from_logits(model.reward_head(h, z_flat))[0].item(),
            batch["rewards"][0, t].item(),
        ))
    wall_flat = states[0, 0, 459:900].cpu().numpy()
    return wall_flat, preds, gts, rewards


def plot_rollout(wall_flat, preds, gts, rewards, horizons, title, out_path) -> None:
    horizons = [k for k in horizons if k <= len(preds)]
    n = len(horizons)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, k in enumerate(horizons):
        gt_img = render_grid(wall_flat, gts[k - 1])
        pr_img = render_grid(wall_flat, preds[k - 1])
        for row, img, lbl in ((0, gt_img, "GT"), (1, pr_img, "pred")):
            ax = axes[row, j]
            ax.imshow(img, cmap=_CMAP, vmin=0, vmax=4, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(lbl, fontsize=12)
        r_pred, r_true = rewards[k - 1]
        axes[0, j].set_title(f"h={k}\nr={r_true:+.2f}", fontsize=9)
        axes[1, j].set_title(f"r̂={r_pred:+.2f}", fontsize=9)
    fig.suptitle(title + "   (yellow=pacman  red=ghost  pink=food  blue=wall)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/dreamer_wm/main/best.pt")
    p.add_argument("--test-dataset", default="main_test",
                   help="Held-out test-pool dataset name under --data-root.")
    p.add_argument("--layout-id", type=int, default=0,
                   help="Which test layout to evaluate (see layout_ids.npy).")
    p.add_argument("--data-root", default="data/replay")
    p.add_argument("--config", default="configs/world_model/dreamer_v3.yaml")
    p.add_argument("--out-dir", default=None,
                   help="Defaults to logs/wm_eval/<dataset>_layout<id>.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--viz-horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = p.parse_args()

    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    cfg = yaml.safe_load(open(ROOT / args.config))["world_model"]
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "logs" / "wm_eval" / f"{args.test_dataset}_layout{args.layout_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute()
                      else args.checkpoint, map_location=device, weights_only=False)
    model = DreamerWorldModel(WorldModelConfig(**ckpt["cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.checkpoint} (step {ckpt.get('step', '?')}) on {device}")

    window = cfg["context"] + cfg["seq_length"]
    replay = SingleLayoutReplay(str(ROOT / args.data_root / args.test_dataset),
                                length=window, layout_id=args.layout_id, seed=0)
    print(f"layout_id={args.layout_id}: {replay.valid_starts.size} valid eval windows "
          f"(window={window})")

    metrics = evaluate(model, replay, context=cfg["context"], horizon=cfg["k_step"],
                       n_windows=cfg["n_eval_windows"], device=device, seed=cfg["eval_seed"])

    print(f"\n=== Test map (layout_id={args.layout_id}) metrics ===")
    for k in sorted(metrics):
        if not k.endswith("_curve"):
            print(f"  {k}: {metrics[k]:.5f}")
    (out_dir / "metrics.json").write_text(
        json.dumps({"layout_id": args.layout_id, "checkpoint": args.checkpoint,
                    "step": ckpt.get("step"), **metrics}, indent=2))

    title = f"WM eval — {args.test_dataset} layout_id={args.layout_id} (step {ckpt.get('step', '?')})"
    plot_kstep_curves(metrics, title, out_dir / "kstep_curves.png")

    # Qualitative rollout on a single representative window.
    batch = next(replay.iter_eval_windows(1, device=device, seed=cfg["eval_seed"]))
    batch = {k: v.unsqueeze(0) for k, v in batch.items()}
    wall_flat, preds, gts, rewards = rollout_for_viz(
        model, batch, cfg["context"], cfg["k_step"], device)
    plot_rollout(wall_flat, preds, gts, rewards, args.viz_horizons, title,
                 out_dir / "rollout_qualitative.png")

    print(f"\nSaved → {out_dir}/")
    for f in ("metrics.json", "kstep_curves.png", "rollout_qualitative.png"):
        print(f"  {f}")


if __name__ == "__main__":
    main()
