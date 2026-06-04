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
from matplotlib.patches import Wedge, Circle, Rectangle, Polygon
from collections import deque

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


class SingleEpisodeReplay(SingleLayoutReplay):
    """SingleLayoutReplay further restricted to windows lying within ONE episode
    (no is_first reset after the window's first step).

    Open-loop imagination gets no observations and is not reset mid-rollout, so a
    window that crosses an episode boundary is impossible to predict (the next
    episode re-spawns pacman/ghosts at random and re-samples the ghost count).
    Measuring across such a boundary inflates the error and makes the GT vs
    imagine ghost counts disagree — so we evaluate imagination on single-episode
    windows only."""

    def __init__(self, dataset_dir: str, length: int, layout_id: int, seed=None):
        super().__init__(dataset_dir, length, layout_id, seed)
        isf = np.asarray(self.is_first).astype(bool)
        csum = np.concatenate([[0], np.cumsum(isf)])
        s = self.valid_starts
        # resets strictly inside the window (s, s+length): a reset at s itself is
        # fine (the window simply begins at an episode start).
        resets_inside = csum[s + self.length] - csum[s + 1]
        self.valid_starts = s[resets_inside == 0]
        if self.valid_starts.size == 0:
            raise ValueError(
                f"No single-episode length-{self.length} window inside layout_id={layout_id} "
                f"(episodes are shorter than the warm-up+horizon span)."
            )


# --------------------------------------------------------------------------- #
def _denorm(coord: float, span: int = GRID_W) -> int:
    """Inverse of pacman_env.state._normalize: [-1,1] → integer cell index.

    Clamped to the valid grid: pacman/ghosts ALWAYS exist, so a predicted coord
    that drifts slightly outside [-1,1] (common for entities near the maze edge in
    open-loop rollout) is snapped to the boundary cell rather than dropped."""
    return int(np.clip(round((float(coord) + 1.0) / 2.0 * (span - 1)), 0, span - 1))


def snap_off_wall(r: int, c: int, walls: np.ndarray) -> tuple[int, int]:
    """Move a cell off a wall to the nearest non-wall cell (BFS, 4-connected).

    Pacman and ghosts can NEVER legally occupy a wall, but the WM decodes a
    continuous position that — especially in open-loop rollout — sometimes rounds
    onto a wall cell. We snap it to the closest walkable cell instead of drawing
    an impossible state. Returns the input unchanged if no non-wall cell exists."""
    H, W = walls.shape
    if not walls[r, c]:
        return r, c
    seen = {(r, c)}
    q = deque([(r, c)])
    while q:
        cr, cc = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen:
                if not walls[nr, nc]:
                    return nr, nc
                seen.add((nr, nc))
                q.append((nr, nc))
    return r, c


def pacman_cell(dyn: np.ndarray, walls: np.ndarray | None = None) -> tuple[int, int]:
    """Decoded pacman (row, col); snapped off a wall when `walls` is given."""
    r, c = _denorm(dyn[1], GRID_H), _denorm(dyn[0], GRID_W)
    return snap_off_wall(r, c, walls) if walls is not None else (r, c)


def render_grid(wall_flat: np.ndarray, dyn: np.ndarray, snap: bool = True) -> np.ndarray:
    """Render one game frame to an (H, W) integer label image.

    Labels: 0 empty, 1 wall, 2 food, 3 ghost, 4 pacman (pacman drawn last/on top).
    `wall_flat` is the static 441-d wall mask; `dyn` is the 460-d dynamic vector.
    When `snap`, pacman/ghosts that decode onto a wall are moved to the nearest
    walkable cell so an impossible state is never drawn.
    """
    walls = wall_flat.reshape(GRID_H, GRID_W) > 0.5
    img = np.zeros((GRID_H, GRID_W), dtype=np.int64)
    img[walls] = 1

    food = dyn[FOOD_SLICE].reshape(GRID_H, GRID_W)
    img[(food > 0.5) & (img == 0)] = 2

    ghosts = dyn[GHOST_SLICE].reshape(4, 4)
    for gx, gy, alive, valid in ghosts:
        if valid > 0.5 and alive > 0.5:   # invalid slot / eaten ghost is legitimately absent
            gr, gc = _denorm(gy, GRID_H), _denorm(gx, GRID_W)
            if snap:
                gr, gc = snap_off_wall(gr, gc, walls)
            img[gr, gc] = 3

    pr, pc = (pacman_cell(dyn, walls) if snap else
              (_denorm(dyn[1], GRID_H), _denorm(dyn[0], GRID_W)))
    img[pr, pc] = 4
    return img


_CMAP = ListedColormap(["#000000", "#1a1aff", "#ffb8b8", "#ff3030", "#ffe000"])  # empty/wall/food/ghost/pac

# Classic arcade ghost colors (Blinky, Pinky, Inky, Clyde).
GHOST_COLORS = ["#ff0000", "#ffb8ff", "#00ffff", "#ffb852"]
WALL_FACE, WALL_EDGE = "#1414b8", "#3d3dff"
FOOD_COLOR = "#ffd9b3"
PAC_COLOR = "#ffe000"
BG_COLOR = "#000000"


def _facing_angle(prev_dyn, dyn) -> float:
    """Direction pacman faces, in degrees, from previous→current decoded cell.
    Defaults to facing right when there is no motion / no previous frame."""
    if prev_dyn is None:
        return 0.0
    dr = _denorm(dyn[1], GRID_H) - _denorm(prev_dyn[1], GRID_H)
    dc = _denorm(dyn[0], GRID_W) - _denorm(prev_dyn[0], GRID_W)
    if dr == 0 and dc == 0:
        return 0.0
    return float(np.degrees(np.arctan2(dr, dc)))   # y-axis is inverted in draw_frame


def _draw_pacman(ax, r, c, angle_deg, scale=0.46):
    half = 32.0                      # half-mouth opening (deg)
    ax.add_patch(Wedge((c, r), scale, angle_deg + half, angle_deg + 360 - half,
                       facecolor=PAC_COLOR, edgecolor="#caa800", lw=0.5, zorder=8))
    # eye, offset perpendicular to facing
    a = np.radians(angle_deg)
    ex = c + 0.10 * np.cos(a) - 0.18 * np.sin(a)
    ey = r + 0.10 * np.sin(a) + 0.18 * np.cos(a)
    ax.add_patch(Circle((ex, ey), 0.07, facecolor="#222", edgecolor="none", zorder=9))


def _draw_ghost(ax, r, c, color, facing=0.0, scale=0.46):
    R = scale
    pts = []
    n_arc = 14
    for i in range(n_arc + 1):                     # rounded top (semicircle, y-up local)
        ang = np.pi * (1 - i / n_arc)
        pts.append((R * np.cos(ang), R * np.sin(ang)))
    pts.append((R, -0.25 * R))                     # right side down
    n_seg = 8                                       # wavy skirt
    for i in range(n_seg + 1):
        x = R - 2 * R * i / n_seg
        y = -R if i % 2 == 0 else -R + 0.32 * R
        pts.append((x, y))
    pts.append((-R, -0.25 * R))                     # left side up
    verts = [(c + x, r - y) for x, y in pts]        # y-up local → data (axis inverted)
    ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor="none", zorder=5))
    fx, fy = 0.05 * np.cos(np.radians(facing)), 0.05 * np.sin(np.radians(facing))
    for sx in (-1, 1):                              # two eyes, pupils toward facing
        ex, ey = c + sx * 0.17, r - 0.12
        ax.add_patch(Circle((ex, ey), 0.12, facecolor="white", edgecolor="none", zorder=6))
        ax.add_patch(Circle((ex + fx, ey + fy), 0.06, facecolor="#1414b8", edgecolor="none", zorder=7))


def draw_frame(ax, wall_flat, dyn, prev_dyn=None, snap=True):
    """Render one frame in an arcade-Pac-Man style onto `ax` (walls blue, food
    dots, classic ghosts, pacman wedge). Entities decoding onto a wall are snapped
    to the nearest walkable cell when `snap`."""
    walls = wall_flat.reshape(GRID_H, GRID_W) > 0.5
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-0.5, GRID_W - 0.5)
    ax.set_ylim(-0.5, GRID_H - 0.5)
    ax.invert_yaxis()                               # row 0 at top
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    for rr in range(GRID_H):                         # walls
        for cc in range(GRID_W):
            if walls[rr, cc]:
                ax.add_patch(Rectangle((cc - 0.5, rr - 0.5), 1, 1,
                                       facecolor=WALL_FACE, edgecolor=WALL_EDGE, lw=0.4, zorder=1))
    food = dyn[FOOD_SLICE].reshape(GRID_H, GRID_W)   # food pellets
    fr, fc = np.where((food > 0.5) & (~walls))
    for rr, cc in zip(fr, fc):
        ax.add_patch(Circle((cc, rr), 0.09, facecolor=FOOD_COLOR, edgecolor="none", zorder=2))

    for gi, (gx, gy, alive, valid) in enumerate(dyn[GHOST_SLICE].reshape(4, 4)):
        if valid > 0.5 and alive > 0.5:
            gr, gc = _denorm(gy, GRID_H), _denorm(gx, GRID_W)
            if snap:
                gr, gc = snap_off_wall(gr, gc, walls)
            _draw_ghost(ax, gr, gc, GHOST_COLORS[gi % 4])

    pr, pc = pacman_cell(dyn, walls if snap else None)
    _draw_pacman(ax, pr, pc, _facing_angle(prev_dyn, dyn))


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
    the real actions. Returns per-horizon predicted & ground-truth dynamic states.

    Uses `model.decode_state`, which fills entity positions from the two-hot
    PositionHead when the model is in twohot mode (else the regression decoder)."""
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
        recon = model.decode_state(h, z_flat)          # (1, 460) — head-filled positions
        preds.append(recon[0].cpu().numpy())
        gts.append(extract_dynamic(states[:, t])[0].cpu().numpy())
        rewards.append((
            model.reward_from_logits(model.reward_head(h, z_flat))[0].item(),
            batch["rewards"][0, t].item(),
        ))
    wall_flat = states[0, 0, 459:900].cpu().numpy()
    return wall_flat, preds, gts, rewards


def _entity_cell_errors(pred_dyn: np.ndarray, gt_dyn: np.ndarray, walls: np.ndarray) -> dict:
    """Cell-L1 error (Manhattan, in grid cells) between predicted and GT entity
    positions for one frame, plus whether the predicted cell lands on a wall.
    Ghosts are only scored where the GT ghost is alive & valid."""
    pr_r, pr_c = _denorm(pred_dyn[1], GRID_H), _denorm(pred_dyn[0], GRID_W)
    gt_r, gt_c = _denorm(gt_dyn[1], GRID_H), _denorm(gt_dyn[0], GRID_W)
    out = {"pac_l1": abs(pr_r - gt_r) + abs(pr_c - gt_c),
           "pac_wall": float(walls[pr_r, pr_c]),
           "ghost_l1": [], "ghost_wall": []}
    pg = pred_dyn[GHOST_SLICE].reshape(4, 4)
    gg = gt_dyn[GHOST_SLICE].reshape(4, 4)
    for i in range(4):
        if gg[i, 2] > 0.5 and gg[i, 3] > 0.5:          # GT ghost alive & valid
            r, c = _denorm(pg[i, 1], GRID_H), _denorm(pg[i, 0], GRID_W)
            rr, cc = _denorm(gg[i, 1], GRID_H), _denorm(gg[i, 0], GRID_W)
            out["ghost_l1"].append(abs(r - rr) + abs(c - cc))
            out["ghost_wall"].append(float(walls[r, c]))
    return out


@torch.no_grad()
def measure_imagine_error(model, replay, context, horizon, n_windows, device, seed=0) -> dict:
    """Aggregate open-loop imagination error vs GT, per horizon, over many windows.

    Returns per-horizon arrays of pacman / ghost cell-L1, on-wall rate, food-cell
    IoU, and reward |error| — the quantitative companion to the qualitative plot."""
    H = horizon
    pac_l1 = [[] for _ in range(H)]; pac_wall = [[] for _ in range(H)]
    gh_l1 = [[] for _ in range(H)]; gh_wall = [[] for _ in range(H)]
    food_iou = [[] for _ in range(H)]; rew_abs = [[] for _ in range(H)]
    n_used = 0
    for batch in replay.iter_eval_windows(n_windows, device=device, seed=seed):
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        wall_flat, preds, gts, rewards = rollout_for_viz(model, batch, context, horizon, device)
        walls = wall_flat.reshape(GRID_H, GRID_W) > 0.5
        n_used += 1
        for k in range(len(preds)):
            err = _entity_cell_errors(preds[k], gts[k], walls)
            pac_l1[k].append(err["pac_l1"]); pac_wall[k].append(err["pac_wall"])
            gh_l1[k].extend(err["ghost_l1"]); gh_wall[k].extend(err["ghost_wall"])
            pf = (preds[k][FOOD_SLICE] > 0.5); gf = (gts[k][FOOD_SLICE] > 0.5)
            inter = float((pf & gf).sum()); union = float((pf | gf).sum())
            food_iou[k].append(inter / union if union > 0 else 1.0)
            rp, rt = rewards[k]
            rew_abs[k].append(abs(rp - rt))
    mean = lambda L: [float(np.mean(x)) if x else 0.0 for x in L]
    return {
        "n_windows": n_used, "horizon": len(pac_l1),
        "imagine/pac_cell_l1_curve": mean(pac_l1),
        "imagine/pac_on_wall_curve": mean(pac_wall),
        "imagine/ghost_cell_l1_curve": mean(gh_l1),
        "imagine/ghost_on_wall_curve": mean(gh_wall),
        "imagine/food_iou_curve": mean(food_iou),
        "imagine/reward_abs_err_curve": mean(rew_abs),
    }


def plot_imagine_curves(m: dict, title: str, out_path: Path) -> None:
    """Per-horizon imagination error vs GT (cell-L1, on-wall, food IoU, reward err)."""
    hs = list(range(1, m["horizon"] + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hs, m["imagine/pac_cell_l1_curve"], "-o", ms=3, color="tab:orange", label="pacman")
    axes[0].plot(hs, m["imagine/ghost_cell_l1_curve"], "-o", ms=3, color="tab:red", label="ghost")
    axes[0].set_ylabel("position cell-L1 error"); axes[0].legend()
    axes[1].plot(hs, m["imagine/pac_on_wall_curve"], "-o", ms=3, color="tab:orange", label="pacman")
    axes[1].plot(hs, m["imagine/ghost_on_wall_curve"], "-o", ms=3, color="tab:red", label="ghost")
    axes[1].set_ylabel("on-wall rate"); axes[1].legend()
    axes[2].plot(hs, m["imagine/food_iou_curve"], "-o", ms=3, color="tab:green", label="food IoU")
    axes[2].plot(hs, m["imagine/reward_abs_err_curve"], "-o", ms=3, color="tab:blue", label="|reward err|")
    axes[2].set_ylabel("food IoU / |reward err|"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("open-loop (imagine) horizon"); ax.grid(alpha=0.3)
    fig.suptitle(title + " — imagination error vs GT")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rollout(wall_flat, preds, gts, rewards, horizons, title, out_path) -> None:
    horizons = [k for k in horizons if k <= len(preds)]
    n = len(horizons)
    walls = wall_flat.reshape(GRID_H, GRID_W) > 0.5
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.8), facecolor="#111")
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, k in enumerate(horizons):
        # facing comes from the previous decoded frame in the SAME stream
        gt_prev = gts[k - 2] if k >= 2 else None
        pr_prev = preds[k - 2] if k >= 2 else None
        for row, (seq, prev, lbl) in enumerate(((gts, gt_prev, "GT"), (preds, pr_prev, "imagine"))):
            ax = axes[row, j]
            draw_frame(ax, wall_flat, seq[k - 1], prev_dyn=prev, snap=True)
            if j == 0:
                ax.set_ylabel(lbl, fontsize=12, color="w")
        r_pred, r_true = rewards[k - 1]
        err = _entity_cell_errors(preds[k - 1], gts[k - 1], walls)
        gl1 = np.mean(err["ghost_l1"]) if err["ghost_l1"] else 0.0
        axes[0, j].set_title(f"h={k}\nr={r_true:+.2f}", fontsize=9, color="w")
        # pred-frame caption shows reward AND position divergence from GT at this horizon
        axes[1, j].set_title(f"r̂={r_pred:+.2f}\npacΔ{err['pac_l1']} ghΔ{gl1:.1f}",
                             fontsize=8, color="#ffd9b3")
    fig.suptitle(title + "   (GT top vs imagine bottom; Δ = cell-L1; pacman snapped off walls)",
                 fontsize=10, color="w")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="#111")
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
    p.add_argument("--n-examples", type=int, default=4,
                   help="How many example imagination rollouts to render (one PNG each).")
    p.add_argument("--measure-windows", type=int, default=None,
                   help="Windows for the quantitative imagine-vs-GT error curves "
                        "(default: config n_eval_windows).")
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

    # Imagination is evaluated on SINGLE-EPISODE windows: warm-up + horizon must
    # lie inside one episode so the open-loop rollout never crosses an unobservable
    # episode boundary. Window length = context + k_step (the warm+imagine span).
    window = cfg["context"] + cfg["k_step"]
    replay = SingleEpisodeReplay(str(ROOT / args.data_root / args.test_dataset),
                                 length=window, layout_id=args.layout_id, seed=0)
    print(f"layout_id={args.layout_id}: {replay.valid_starts.size} single-episode eval windows "
          f"(window={window} = context {cfg['context']} + horizon {cfg['k_step']})")

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

    # --- Quantitative imagination error vs GT, per horizon (cell-L1 etc.). ---
    mw = args.measure_windows or cfg["n_eval_windows"]
    imag = measure_imagine_error(model, replay, cfg["context"], cfg["k_step"],
                                 n_windows=mw, device=device, seed=cfg["eval_seed"])
    print(f"\n=== Imagination (open-loop) error vs GT — {imag['n_windows']} windows ===")
    for h in (1, 2, 4, 8, 16, cfg["k_step"]):
        if h <= imag["horizon"]:
            i = h - 1
            print(f"  h={h:2d}: pacman cell-L1 {imag['imagine/pac_cell_l1_curve'][i]:.2f} "
                  f"(on-wall {imag['imagine/pac_on_wall_curve'][i]:.2f}) | "
                  f"ghost {imag['imagine/ghost_cell_l1_curve'][i]:.2f} "
                  f"(on-wall {imag['imagine/ghost_on_wall_curve'][i]:.2f}) | "
                  f"food IoU {imag['imagine/food_iou_curve'][i]:.2f} | "
                  f"|rew err| {imag['imagine/reward_abs_err_curve'][i]:.2f}")
    metrics.update(imag)
    (out_dir / "metrics.json").write_text(
        json.dumps({"layout_id": args.layout_id, "checkpoint": args.checkpoint,
                    "step": ckpt.get("step"), **metrics}, indent=2))
    plot_imagine_curves(imag, title, out_dir / "imagine_error_curves.png")

    # --- Qualitative: several example imagination rollouts (GT vs imagine). ---
    written = ["metrics.json", "kstep_curves.png", "imagine_error_curves.png"]
    examples = list(replay.iter_eval_windows(args.n_examples, device=device, seed=cfg["eval_seed"]))
    for ei, batch in enumerate(examples):
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        wall_flat, preds, gts, rewards = rollout_for_viz(
            model, batch, cfg["context"], cfg["k_step"], device)
        fname = f"rollout_example_{ei:02d}.png"
        plot_rollout(wall_flat, preds, gts, rewards, args.viz_horizons,
                     title + f"  [example {ei}]", out_dir / fname)
        written.append(fname)

    print(f"\nSaved → {out_dir}/")
    for f in written:
        print(f"  {f}")


if __name__ == "__main__":
    main()
