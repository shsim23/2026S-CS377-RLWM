"""Visualize world-model IMAGINATION trajectories.

Each scene renders two rows of K+1 frames:
    Top    : ground-truth state at time t
    Bottom : world-model's imagined state at time t
             (z_t from autoregressive rollout, then DynamicStateHead(z) decoded)

Walls are taken from the seed state (episode-constant; the model is not
required to predict them).

Reward and done are also plotted (pred vs true) below the frames.

Usage:
    python scripts/imagine_trajectory.py \
        --checkpoint checkpoints/pacman_classic_v10c/best.pt \
        --data-dir   data/replay/pacman_classic \
        --k-step 10 --n-scenes 4 --burnin 5 \
        --out-dir viz/imagination
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_env.constants import MAX_GHOSTS, MAX_GRID_H, MAX_GRID_W
from world_model import EnsembleWorldModel, SequenceReplayBuffer

# state vector layout (matches StateBuilder)
STATE_PAC    = slice(0, 2)
STATE_GHOSTS = slice(2, 2 + MAX_GHOSTS * 4)
STATE_FOOD   = slice(18, 18 + 441)
STATE_WALL   = slice(459, 459 + 441)
STATE_POWER  = 900

# DynamicStateHead output layout (460-D, no walls)
OUT_PAC    = slice(0, 2)
OUT_GHOSTS = slice(2, 18)
OUT_FOOD   = slice(18, 459)
OUT_POWER  = slice(459, 460)


def _denorm(xn, yn):
    x = int(round((xn + 1) * (MAX_GRID_W - 1) / 2))
    y = int(round((yn + 1) * (MAX_GRID_H - 1) / 2))
    return x, y


def decode_true(vec: np.ndarray):
    walls = vec[STATE_WALL].reshape(MAX_GRID_H, MAX_GRID_W) > 0.5
    food  = vec[STATE_FOOD].reshape(MAX_GRID_H, MAX_GRID_W) > 0.5
    pac   = _denorm(vec[0], vec[1])
    ghosts = []
    g = vec[STATE_GHOSTS].reshape(MAX_GHOSTS, 4)
    for i in range(MAX_GHOSTS):
        gx, gy, alive, valid = g[i]
        if valid > 0.5 and alive > 0.5:
            ghosts.append(_denorm(gx, gy))
    power = float(vec[STATE_POWER])
    return walls, food, pac, ghosts, power


def decode_pred(pred_460: np.ndarray, walls: np.ndarray):
    """Decode DynamicStateHead output. Food slice is logits (apply sigmoid)."""
    food_logits = pred_460[OUT_FOOD]
    food = 1.0 / (1.0 + np.exp(-food_logits))
    food = food.reshape(MAX_GRID_H, MAX_GRID_W) > 0.5

    pac = _denorm(pred_460[0], pred_460[1])
    ghosts = []
    g = pred_460[OUT_GHOSTS].reshape(MAX_GHOSTS, 4)
    for i in range(MAX_GHOSTS):
        gx, gy, alive, valid = g[i]
        if valid > 0.5 and alive > 0.5:
            ghosts.append(_denorm(gx, gy))
    power = float(pred_460[OUT_POWER][0])
    return walls, food, pac, ghosts, power


def _bounds(walls, food):
    """Bounding box of the playable area for cleaner cropping."""
    occupied_cols = np.any(~walls | food, axis=0)
    occupied_rows = np.any(~walls | food, axis=1)
    last_col = int(np.where(occupied_cols)[0].max()) + 1 if occupied_cols.any() else walls.shape[1]
    last_row = int(np.where(occupied_rows)[0].max()) + 1 if occupied_rows.any() else walls.shape[0]
    return last_row, last_col


def draw_frame(ax, walls, food, pac, ghosts, power, title, bounds):
    last_row, last_col = bounds
    ax.set_xlim(-0.5, last_col - 0.5)
    ax.set_ylim(last_row - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    ax.set_facecolor("black")

    # walls
    for y in range(last_row):
        for x in range(last_col):
            if walls[y, x]:
                ax.add_patch(patches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1, color="#2233aa"))
    # food
    ys, xs = np.where(food[:last_row, :last_col])
    ax.scatter(xs, ys, s=6, c="white", edgecolors="none")
    # ghosts
    ghost_color = "#ff5555" if power < 0.5 else "#5588ff"
    for gx, gy in ghosts:
        if gx < last_col and gy < last_row:
            ax.add_patch(patches.Circle((gx, gy), 0.4, color=ghost_color))
    # pacman
    px, py = pac
    if 0 <= px < last_col and 0 <= py < last_row:
        ax.add_patch(patches.Circle((px, py), 0.42, color="#ffd400"))


def imagine_rollout(ensemble, traj, K, burnin):
    """Returns lists indexed 0..K_used with imagined state predictions.

    item 0 corresponds to the (real) seed state — at t=0 the imagined state is
    DynamicStateHead(encoder(seed)), so any reconstruction error there is the
    encoder/decoder gap, not dynamics rollout error.
    """
    device = next(ensemble.parameters()).device
    states  = torch.from_numpy(traj["states"]).float().to(device)
    actions = torch.from_numpy(traj["actions"]).long().to(device)
    rewards = traj["rewards"]
    dones   = traj["dones"]
    T = len(states)
    if T < burnin + K + 1:
        return None

    # GRU warmup
    if burnin > 0:
        z, h = ensemble.warmup_h(states[: burnin + 1].unsqueeze(0),
                                 actions[: burnin + 1].unsqueeze(0))
        start = burnin
    else:
        z, h = ensemble.encode(states[0:1])
        start = 0

    member = ensemble.members[0]   # K=1
    pred_states = []                # decoded state vectors per step
    r_pred, r_true = [], []
    d_pred, d_true = [], []
    lat_mse = []
    end_step = K

    with torch.no_grad():
        # t=0 — seed encoding decoded straight away
        ds0 = member.dynamic_state_head(z).squeeze(0).cpu().numpy()
        pred_states.append(ds0)

        for t in range(K):
            a = actions[start + t: start + t + 1]
            out = ensemble.imagine_step(z, h, a)
            z_next, h_next = out["z_next"], out["h_next"]

            ds = member.dynamic_state_head(z_next).squeeze(0).cpu().numpy()
            pred_states.append(ds)

            r_pred.append(out["reward"].item())
            r_true.append(float(rewards[start + t]))
            d_pred.append(out["done"].item())
            d_true.append(float(dones[start + t]))

            z_true, _ = ensemble.encode(states[start + t + 1: start + t + 2])
            lat_mse.append(((z_next - z_true) ** 2).mean().item())

            if dones[start + t]:
                end_step = t + 1
                break
            z, h = z_next, h_next

    return {
        "true_states": states.cpu().numpy(),
        "pred_states": pred_states,
        "start": start,
        "K_used": end_step,
        "r_true": np.array(r_true),
        "r_pred": np.array(r_pred),
        "d_true": np.array(d_true),
        "d_pred": np.array(d_pred),
        "lat_mse": np.array(lat_mse),
    }


def make_scene_figure(ensemble, traj, scene_idx, K, burnin, out_path,
                      max_frames: int = 6):
    res = imagine_rollout(ensemble, traj, K, burnin)
    if res is None:
        return False

    K_used = res["K_used"]
    start = res["start"]

    # walls from seed (episode-constant)
    walls_true, food0_true, pac0, ghosts0, power0 = decode_true(res["true_states"][start])
    walls = walls_true
    bounds = _bounds(walls, food0_true)

    # pick frame indices: subsample to max_frames if K+1 too many
    all_idx = list(range(0, K_used + 1))     # 0..K_used
    if len(all_idx) > max_frames:
        step = (len(all_idx) - 1) / (max_frames - 1)
        all_idx = [int(round(i * step)) for i in range(max_frames)]
    n_frames = len(all_idx)

    fig = plt.figure(figsize=(2.0 * n_frames + 1, 6.5))
    gs = fig.add_gridspec(3, n_frames, height_ratios=[1.0, 1.0, 0.9],
                          hspace=0.40, wspace=0.18)

    for col, ti in enumerate(all_idx):
        # top row: true
        ax_t = fig.add_subplot(gs[0, col])
        w, f, p, gs_, pw = decode_true(res["true_states"][start + ti])
        draw_frame(ax_t, w, f, p, gs_, pw,
                   f"TRUE  t={ti}", bounds)

        # middle row: imagined
        ax_p = fig.add_subplot(gs[1, col])
        w_p, f_p, p_p, gs_p, pw_p = decode_pred(res["pred_states"][ti], walls)
        draw_frame(ax_p, w_p, f_p, p_p, gs_p, pw_p,
                   f"IMAG  t={ti}", bounds)

    # bottom row: reward / done / latent MSE in 3 cells
    if n_frames >= 3:
        ax_r = fig.add_subplot(gs[2, :max(1, n_frames // 3)])
        xs = np.arange(1, K_used + 1)
        ax_r.plot(xs, res["r_true"], "-o", color="#22aa22", markersize=4, label="true")
        ax_r.plot(xs, res["r_pred"], "--s", color="#cc4444", markersize=4, label="pred")
        ax_r.set_title(f"reward  MSE={((res['r_true']-res['r_pred'])**2).mean():.3f}", fontsize=9)
        ax_r.set_xlabel("step"); ax_r.legend(fontsize=7); ax_r.grid(alpha=0.3)

        ax_d = fig.add_subplot(gs[2, max(1, n_frames // 3):max(2, 2 * n_frames // 3)])
        ax_d.plot(xs, res["d_true"], "-o", color="#22aa22", markersize=4, label="true")
        ax_d.plot(xs, res["d_pred"], "--s", color="#cc4444", markersize=4, label="pred")
        ax_d.set_ylim(-0.05, 1.05)
        ax_d.set_title(f"done  |err|={np.abs(res['d_true']-res['d_pred']).mean():.3f}", fontsize=9)
        ax_d.set_xlabel("step"); ax_d.legend(fontsize=7); ax_d.grid(alpha=0.3)

        ax_l = fig.add_subplot(gs[2, max(2, 2 * n_frames // 3):])
        ax_l.plot(xs, res["lat_mse"], "-o", color="#444", markersize=4)
        ax_l.set_title(f"latent MSE per step  mean={res['lat_mse'].mean():.3f}", fontsize=9)
        ax_l.set_xlabel("step"); ax_l.grid(alpha=0.3)

    fig.suptitle(
        f"scene {scene_idx}  —  burnin={burnin}, K_used={K_used}, "
        f"ep_len={len(res['true_states'])}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data/replay/pacman_classic")
    p.add_argument("--split", default="val")
    p.add_argument("--k-step", type=int, default=10)
    p.add_argument("--burnin", type=int, default=5,
                   help="GRU warmup steps before starting rollout")
    p.add_argument("--n-scenes", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=6,
                   help="If K+1 frames too many, subsample to this")
    p.add_argument("--out-dir", default="viz/imagination")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ensemble = EnsembleWorldModel.load(args.checkpoint)
    ensemble.eval()
    buf = SequenceReplayBuffer(args.data_dir, split=args.split)

    np.random.seed(args.seed)
    trajs = buf.sample_trajectories(min(args.n_scenes * 5, len(buf.episode_files)),
                                    seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for traj in trajs:
        if made >= args.n_scenes:
            break
        path = out_dir / f"imagine_{made:02d}.png"
        if make_scene_figure(ensemble, traj, made, args.k_step, args.burnin,
                             path, max_frames=args.max_frames):
            print(f"saved {path}  (ep_len={len(traj['states'])})")
            made += 1

    print(f"\nWrote {made} figures to {out_dir}/")


if __name__ == "__main__":
    main()
