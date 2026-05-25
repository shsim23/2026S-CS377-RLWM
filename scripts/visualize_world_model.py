"""Per-scene visualization of world-model K-step rollout vs ground truth.

For each scene we render:
  - Top row: ground-truth Pac-Man frames at t = 0, K/2, K
  - Bottom row: reward(true vs pred), done(true vs pred), per-step latent MSE,
    ensemble sigma across the rollout.
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

STATE_PAC = slice(0, 2)
STATE_GHOSTS = slice(2, 2 + MAX_GHOSTS * 4)
STATE_FOOD = slice(18, 18 + 441)
STATE_WALL = slice(459, 459 + 441)
STATE_POWER = 900


def decode_state(vec: np.ndarray):
    """Recover walls, food mask, pacman pos, ghost list from a state vector."""
    walls = vec[STATE_WALL].reshape(MAX_GRID_H, MAX_GRID_W) > 0.5
    food = vec[STATE_FOOD].reshape(MAX_GRID_H, MAX_GRID_W) > 0.5

    def denorm(xn, yn):
        x = int(round((xn + 1) * (MAX_GRID_W - 1) / 2))
        y = int(round((yn + 1) * (MAX_GRID_H - 1) / 2))
        return x, y

    pac = denorm(vec[0], vec[1])
    ghosts = []
    g = vec[STATE_GHOSTS].reshape(MAX_GHOSTS, 4)
    for i in range(MAX_GHOSTS):
        gx, gy, alive, valid = g[i]
        if valid > 0.5 and alive > 0.5:
            ghosts.append(denorm(gx, gy))
    return walls, food, pac, ghosts, float(vec[STATE_POWER])


def draw_frame(ax, vec: np.ndarray, title: str):
    walls, food, pac, ghosts, power = decode_state(vec)
    H, W = walls.shape

    # crop empty padding columns/rows on right/bottom for cleaner view
    occupied_cols = np.any(~walls | food, axis=0)
    occupied_rows = np.any(~walls | food, axis=1)
    last_col = int(np.where(occupied_cols)[0].max()) + 1 if occupied_cols.any() else W
    last_row = int(np.where(occupied_rows)[0].max()) + 1 if occupied_rows.any() else H

    ax.set_xlim(-0.5, last_col - 0.5)
    ax.set_ylim(last_row - 0.5, -0.5)  # invert y
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)

    # walls
    for y in range(last_row):
        for x in range(last_col):
            if walls[y, x]:
                ax.add_patch(patches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1, color="#2233aa"))

    # food
    ys, xs = np.where(food[:last_row, :last_col])
    ax.scatter(xs, ys, s=8, c="white", edgecolors="none")

    # ghosts
    ghost_color = "#ff5555" if power < 0.5 else "#5588ff"
    for gx, gy in ghosts:
        ax.add_patch(patches.Circle((gx, gy), 0.4, color=ghost_color))

    # pacman
    ax.add_patch(patches.Circle(pac, 0.42, color="#ffd400"))
    ax.set_facecolor("black")


def rollout_metrics(ensemble, traj, K, burnin):
    """Run K-step rollout from traj. Return per-step arrays."""
    device = next(ensemble.parameters()).device
    states = torch.from_numpy(traj["states"]).float().to(device)
    actions = torch.from_numpy(traj["actions"]).long().to(device)
    rewards = traj["rewards"]
    dones = traj["dones"]
    T = len(states)
    if T < burnin + K + 1:
        return None

    # optional GRU warmup with first `burnin` steps
    if burnin > 0:
        prefix_s = states[: burnin + 1].unsqueeze(0)
        prefix_a = actions[: burnin + 1].unsqueeze(0)
        z, h = ensemble.warmup_h(prefix_s, prefix_a)
        start = burnin
    else:
        z, h = ensemble.encode(states[0:1])
        start = 0

    r_true, r_pred = [], []
    d_true, d_pred = [], []
    lat_mse, sigma = [], []
    end_step = K

    with torch.no_grad():
        for t in range(K):
            a = actions[start + t: start + t + 1]
            out = ensemble.imagine_step(z, h, a)
            z_true, _ = ensemble.encode(states[start + t + 1: start + t + 2])

            r_pred.append(out["reward"].item())
            r_true.append(float(rewards[start + t]))
            d_pred.append(out["done"].item())
            d_true.append(float(dones[start + t]))
            lat_mse.append(((out["z_next"] - z_true) ** 2).mean().item())
            sigma.append(out["sigma"].item())

            if dones[start + t]:
                end_step = t + 1
                break
            z, h = out["z_next"], out["h_next"]

    return {
        "states": states.cpu().numpy(),
        "start": start,
        "K_used": end_step,
        "r_true": np.array(r_true),
        "r_pred": np.array(r_pred),
        "d_true": np.array(d_true),
        "d_pred": np.array(d_pred),
        "lat_mse": np.array(lat_mse),
        "sigma": np.array(sigma),
    }


def make_scene_figure(ensemble, traj, scene_idx, K, burnin, out_path):
    m = rollout_metrics(ensemble, traj, K, burnin)
    if m is None:
        return False

    K_used = m["K_used"]
    start = m["start"]
    frame_idxs = [start, start + K_used // 2, start + K_used]

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.28)

    for col, ti in enumerate(frame_idxs):
        ax = fig.add_subplot(gs[0, col])
        draw_frame(ax, m["states"][ti], f"true frame t={ti - start}")

    xs = np.arange(1, K_used + 1)

    ax_r = fig.add_subplot(gs[1, 0])
    ax_r.plot(xs, m["r_true"], "-o", label="true",  color="#22aa22", markersize=4)
    ax_r.plot(xs, m["r_pred"], "--s", label="pred", color="#cc4444", markersize=4)
    ax_r.set_title(f"reward — MSE={((m['r_true']-m['r_pred'])**2).mean():.3f}", fontsize=10)
    ax_r.set_xlabel("rollout step"); ax_r.legend(fontsize=8); ax_r.grid(alpha=0.3)

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(xs, m["d_true"], "-o", label="true",  color="#22aa22", markersize=4)
    ax_d.plot(xs, m["d_pred"], "--s", label="pred", color="#cc4444", markersize=4)
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.set_title(f"done — |err|={np.abs(m['d_true']-m['d_pred']).mean():.3f}", fontsize=10)
    ax_d.set_xlabel("rollout step"); ax_d.legend(fontsize=8); ax_d.grid(alpha=0.3)

    ax_l = fig.add_subplot(gs[1, 2])
    ax_l.plot(xs, m["lat_mse"], "-o", color="#444444", markersize=4, label="latent MSE")
    ax_l.plot(xs, m["sigma"],   "--s", color="#aa6600", markersize=4, label="ens sigma")
    ax_l.set_title("latent MSE  &  ensemble sigma", fontsize=10)
    ax_l.set_xlabel("rollout step"); ax_l.legend(fontsize=8); ax_l.grid(alpha=0.3)

    fig.suptitle(
        f"scene {scene_idx} — burnin={burnin}, K_used={K_used}, ep_len={len(m['states'])}",
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
    p.add_argument("--burnin", type=int, default=0,
                   help="GRU warmup steps before starting rollout (eval default 0)")
    p.add_argument("--n-scenes", type=int, default=6)
    p.add_argument("--out-dir", default="viz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ensemble = EnsembleWorldModel.load(args.checkpoint)
    ensemble.eval()
    buffer = SequenceReplayBuffer(args.data_dir, split=args.split)

    np.random.seed(args.seed)
    # sample enough trajectories — many will be too short for K
    trajs = buffer.sample_trajectories(min(args.n_scenes * 5, len(buffer.episode_files)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for i, traj in enumerate(trajs):
        if made >= args.n_scenes:
            break
        path = out_dir / f"scene_{made:02d}.png"
        if make_scene_figure(ensemble, traj, made, args.k_step, args.burnin, path):
            print(f"saved {path}  (ep_len={len(traj['states'])})")
            made += 1

    print(f"\nWrote {made} scene figures to {out_dir}/")


if __name__ == "__main__":
    main()
