"""Visualize the v0 vs v1 divergence analysis.

Two figures:
  - viz/divergence_eval_trace.png:  K-step eval metrics across training for v0 vs v1
  - viz/divergence_loss_components.png: loss-component contribution under each config
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model import EnsembleWorldModel, SequenceReplayBuffer
from world_model.loss import compute_world_model_loss


# --------------------------------------------------------------------------- #
# Eval trace data (parsed from training stdout)
V0 = {
    "step":   [ 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000],
    "latent": [0.038, 0.012, 0.017, 0.015, 0.253, 0.041, 0.392, 0.729, 0.773, 0.960, 0.662, 0.877],
    "reward": [0.219, 0.203, 0.204, 0.232, 0.224, 0.228, 0.235, 0.217, 0.705, 0.256, 0.201, 0.241],
    "done":   [0.175, 0.125, 0.123, 0.082, 0.107, 0.099, 0.143, 0.155, 0.115, 0.148, 0.183, 0.132],
}

V1 = {
    "step":   [ 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000],
    "latent": [0.054, 0.147, 0.333, 0.419, 0.280, 0.721, 0.405, 0.405, 0.372, 0.285, 0.324, 0.282, 0.425, 0.286, 0.451, 0.394],
    "reward": [0.281, 0.226, 0.224, 0.224, 0.263, 0.242, 0.240, 0.238, 0.235, 0.265, 0.267, 0.237, 0.247, 0.253, 0.261, 0.232],
    "done":   [0.207, 0.317, 0.034, 0.013, 0.030, 0.004, 0.013, 0.014, 0.020, 0.059, 0.032, 0.044, 0.027, 0.041, 0.015, 0.010],
}


# --------------------------------------------------------------------------- #
def measure_components(ckpt_path: str, beta_reward, beta_done, beta_var, pos_weight, n_batches=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buf = SequenceReplayBuffer("data/replay/pacman_classic", split="train")
    ens = EnsembleWorldModel.load(ckpt_path).to(device)
    ens.eval()
    m = ens.members[0]
    keys = ["L_latent", "L_reward", "L_done", "L_var"]
    sums = {k: 0.0 for k in keys}
    for i in range(n_batches):
        batch = buf.sample_sequence(batch_size=64, seq_length=50, bootstrap_seed=20000 + i)
        s = batch["states"].to(device); a = batch["actions"].to(device)
        r = batch["rewards"].to(device); d = batch["dones"].to(device)
        with torch.no_grad():
            out = m.forward_sequence(s, a, burnin=5)
            _, log = compute_world_model_loss(
                out, r, d,
                beta_reward=beta_reward, beta_done=beta_done, beta_var=beta_var,
                pos_weight_done=pos_weight,
            )
        for k in keys:
            sums[k] += log[k] / n_batches
    return sums


def figure_eval_trace(out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # latent_mse — log scale
    ax = axes[0]
    ax.plot(V0["step"], V0["latent"], "-o", color="#3366cc", label="v0 (β=1, pw=1, K=5)", markersize=4)
    ax.plot(V1["step"], V1["latent"], "-s", color="#cc4444", label="v1 (β=3, pw=20, K=1)", markersize=4)
    ax.axhline(0.05, color="green", linestyle=":", linewidth=1, label="threshold 0.05")
    ax.set_yscale("log"); ax.set_xlabel("train step"); ax.set_title("K-step latent_mse")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # reward_mse
    ax = axes[1]
    ax.plot(V0["step"], V0["reward"], "-o", color="#3366cc", label="v0", markersize=4)
    ax.plot(V1["step"], V1["reward"], "-s", color="#cc4444", label="v1", markersize=4)
    ax.axhline(0.10, color="green", linestyle=":", linewidth=1, label="threshold 0.10")
    ax.set_xlabel("train step"); ax.set_title("K-step reward_mse")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # done_err
    ax = axes[2]
    ax.plot(V0["step"], V0["done"], "-o", color="#3366cc", label="v0", markersize=4)
    ax.plot(V1["step"], V1["done"], "-s", color="#cc4444", label="v1", markersize=4)
    ax.axhline(0.10, color="green", linestyle=":", linewidth=1, label="threshold 0.10")
    ax.set_xlabel("train step"); ax.set_title("K-step done_err")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("v0 vs v1: K-step eval metrics across training\n"
                 "v1 diverges on latent immediately while perfecting done — head pressure trade-off",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out_path)


def figure_loss_components(out_path: str):
    # Measure each ckpt under its own training config + the proposed v2 config
    print("measuring v0 best.pt under v0 training settings (β=1, pw=1)...")
    v0_at_v0 = measure_components(
        "checkpoints/pacman_classic/best.pt",
        beta_reward=1.0, beta_done=1.0, beta_var=0.01, pos_weight=1.0,
    )

    print("measuring v1 best.pt under v1 training settings (β=3, pw=20)...")
    v1_at_v1 = measure_components(
        "checkpoints/pacman_classic_v1_freshK1/best.pt",
        beta_reward=3.0, beta_done=3.0, beta_var=0.1, pos_weight=20.0,
    )

    print("measuring v1 final.pt under v1 training settings (β=3, pw=20)...")
    v1_final = measure_components(
        "checkpoints/pacman_classic_v1_freshK1/final.pt",
        beta_reward=3.0, beta_done=3.0, beta_var=0.1, pos_weight=20.0,
    )

    print("measuring v1 best.pt under proposed v2 settings (β=1, pw=5)...")
    v1_at_v2 = measure_components(
        "checkpoints/pacman_classic_v1_freshK1/best.pt",
        beta_reward=1.0, beta_done=1.0, beta_var=0.1, pos_weight=5.0,
    )

    # Convert to loss contributions (β * raw)
    def contrib(d, br, bd, bv, _pw):
        # pw is folded into the raw L_done value via weighted BCE; here we just
        # report the post-β contribution (the raw already reflects pw).
        return {
            "L_latent":  d["L_latent"],
            "L_reward":  br * d["L_reward"],
            "L_done":    bd * d["L_done"],
            "L_var":     bv * d["L_var"],
        }

    configs = [
        ("v0 (β=1, pw=1)\nat best ckpt",         v0_at_v0, 1.0, 1.0, 0.01),
        ("v1 (β=3, pw=20)\nat step-500 (best)",  v1_at_v1, 3.0, 3.0, 0.1),
        ("v1 (β=3, pw=20)\nat step-8000 (final)",v1_final, 3.0, 3.0, 0.1),
        ("proposed v2 (β=1, pw=5)\nat v1 step-500 ckpt", v1_at_v2, 1.0, 1.0, 0.1),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Stacked bar
    ax = axes[0]
    keys = ["L_latent", "L_reward", "L_done", "L_var"]
    colors = ["#3366cc", "#22aa22", "#cc4444", "#aa6600"]
    labels = [c[0] for c in configs]
    bottoms = np.zeros(len(configs))
    for k, color in zip(keys, colors):
        vals = np.array([contrib(c[1], c[2], c[3], c[4], None)[k] for c in configs])
        ax.bar(labels, vals, bottom=bottoms, label=k, color=color)
        for i, v in enumerate(vals):
            if v > 0.03:
                ax.text(i, bottoms[i] + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += vals
    for i, total in enumerate(bottoms):
        ax.text(i, total + 0.05, f"Σ={total:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("contribution to total loss"); ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Loss-component contribution (stacked)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Ratio L_done / L_latent
    ax = axes[1]
    ratios = []
    for label, d, br, bd, bv in configs:
        c = contrib(d, br, bd, bv, None)
        ratios.append(c["L_done"] / max(c["L_latent"], 1e-6))
    bars = ax.bar(labels, ratios, color=["#3366cc", "#cc4444", "#cc4444", "#22aa22"])
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, r + max(ratios) * 0.02,
                f"{r:.1f}×", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="parity")
    ax.axhline(2.0, color="green", linestyle=":", linewidth=1, label="recommended ≤ 2×")
    ax.set_ylabel("L_done / L_latent  (gradient dominance)")
    ax.set_title("Head-vs-dynamics loss balance")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y", which="both")

    fig.suptitle("Loss-component analysis — measured on train data with each ckpt+config\n"
                 "v1 starts with L_done ≈ 55× L_latent: optimizer prioritises done, warps latent",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out_path)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    Path("viz").mkdir(exist_ok=True)
    figure_eval_trace("viz/divergence_eval_trace.png")
    figure_loss_components("viz/divergence_loss_components.png")
