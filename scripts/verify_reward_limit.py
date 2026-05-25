"""Verify the structural reward-limit hypothesis for v8.2.

Tests on val data:
  (A) Per-component dyn aux loss (L_dyn_food BCE, pos MSE, ghost MSE, power MSE)
      — from encoder path (dyn_state(z_all)).
  (B) Same from dynamics path (dyn_state(z_preds)) — that's what reward depends on.
  (C) Sigmoid-sum vs true food count (encoder + dynamics paths).
      If sum systematically biased / noisy, this is the SNR problem.
  (D) Predicted food_eaten (count_t - count_{t+1}, clamped) vs true food_eaten.
      Also: signed delta vs true (to see how often clamp(0) kills a real +1).
  (E) Predicted reward vs true reward (symlog), per food-eaten class.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model import EnsembleWorldModel, SequenceReplayBuffer
from world_model.utils import symlog


@torch.no_grad()
def collect(member, val_buf, device, n_trajs=100, seed=0):
    """For each transition (t -> t+1) in val_buf, collect:
       count_t_true, count_t_enc, count_next_true, count_next_dyn,
       food_eaten_true, food_eaten_pred, food_BCE_enc, food_BCE_dyn,
       pos_err_enc, pos_err_dyn, reward_true_symlog, reward_pred_symlog
    """
    rows = []
    for traj in val_buf.sample_trajectories(n_trajs, seed=seed):
        states  = torch.from_numpy(traj["states"]).float().to(device)
        actions = torch.from_numpy(traj["actions"]).long().to(device)
        rewards = torch.from_numpy(traj["rewards"]).float().to(device)
        T = states.shape[0]
        if T < 2:
            continue

        # encoder path on all states
        z_all = member.encoder(states)                       # (T, latent)
        dyn_enc = member.dynamic_state_head(z_all)           # (T, 460)

        # dynamics path: simulate from t=0 onwards using true z_t as input
        # (this matches the training loss; not full autoregressive rollout)
        h = torch.zeros(1, member.gru_hidden, device=device)
        z_preds_list = []
        for t in range(T - 1):
            a_emb = member.action_embedder(actions[t:t+1])
            z_next, h = member.dynamics(z_all[t:t+1], a_emb, h)
            z_preds_list.append(z_next.squeeze(0))
        z_preds = torch.stack(z_preds_list, dim=0)           # (T-1, latent)
        dyn_dyn = member.dynamic_state_head(z_preds)         # (T-1, 460)

        # true food mask
        food_true_t  = (states[:, 18:459] > 0.5).float()     # (T, 441)
        # predicted food sigmoid (encoder + dynamics paths)
        food_sig_enc = torch.sigmoid(dyn_enc[:, 18:459])     # (T, 441)
        food_sig_dyn = torch.sigmoid(dyn_dyn[:, 18:459])     # (T-1, 441)

        # counts (soft = current behaviour; hard = STE-style threshold)
        count_true        = food_true_t.sum(dim=-1)              # (T,)
        count_enc         = food_sig_enc.sum(dim=-1)             # (T,)
        count_dyn         = food_sig_dyn.sum(dim=-1)             # (T-1,)
        food_hard_enc     = (food_sig_enc > 0.5).float()         # (T, 441)
        food_hard_dyn     = (food_sig_dyn > 0.5).float()         # (T-1, 441)
        count_enc_hard    = food_hard_enc.sum(dim=-1)            # (T,)
        count_dyn_hard    = food_hard_dyn.sum(dim=-1)            # (T-1,)

        # BCE per step
        bce_enc = F.binary_cross_entropy_with_logits(
            dyn_enc[:, 18:459], food_true_t, reduction="none"
        ).mean(dim=-1)                                       # (T,)
        bce_dyn = F.binary_cross_entropy_with_logits(
            dyn_dyn[:, 18:459], food_true_t[1:], reduction="none"
        ).mean(dim=-1)                                       # (T-1,)

        # pos MSE
        pos_err_enc = ((dyn_enc[:, 0:2] - states[:, 0:2]) ** 2).mean(dim=-1)   # (T,)
        pos_err_dyn = ((dyn_dyn[:, 0:2] - states[1:, 0:2]) ** 2).mean(dim=-1)  # (T-1,)

        # food_eaten (true) — number of cells that disappeared between t and t+1
        food_eaten_true = (food_true_t[:-1] * (1 - food_true_t[1:])).sum(dim=-1)  # (T-1,)

        # predicted food_eaten from deterministic reward formula
        # (matches world_model.single._deterministic_reward_symlog)
        delta_signed = count_enc[:-1]      - count_dyn            # soft
        delta_clamp  = delta_signed.clamp(min=0.0)
        delta_hard_s = count_enc_hard[:-1] - count_dyn_hard       # hard (STE-style)
        delta_hard_c = delta_hard_s.clamp(min=0.0)

        # rewards (true; soft path; hard path)
        r_raw_true  = rewards[:-1]
        r_symlog_true     = symlog(r_raw_true)
        r_raw_pred_soft   = -0.01 + 1.0 * delta_clamp
        r_symlog_pred_soft = symlog(r_raw_pred_soft)
        r_raw_pred_hard   = -0.01 + 1.0 * delta_hard_c
        r_symlog_pred_hard = symlog(r_raw_pred_hard)

        for t in range(T - 1):
            rows.append((
                count_true[t].item(),
                count_enc[t].item(),
                count_enc_hard[t].item(),
                count_true[t+1].item(),
                count_dyn[t].item(),
                count_dyn_hard[t].item(),
                food_eaten_true[t].item(),
                delta_signed[t].item(),
                delta_clamp[t].item(),
                delta_hard_s[t].item(),
                delta_hard_c[t].item(),
                bce_enc[t].item(),
                bce_dyn[t].item(),
                pos_err_enc[t].item(),
                pos_err_dyn[t].item(),
                r_raw_true[t].item(),
                r_symlog_true[t].item(),
                r_symlog_pred_soft[t].item(),
                r_symlog_pred_hard[t].item(),
            ))

    cols = ["count_t_true", "count_t_enc", "count_t_enc_hard",
            "count_next_true", "count_next_dyn", "count_next_dyn_hard",
            "food_eaten_true",
            "delta_signed", "delta_clamp",
            "delta_hard_s", "delta_hard_c",
            "bce_enc", "bce_dyn",
            "pos_err_enc", "pos_err_dyn",
            "r_raw_true", "r_symlog_true",
            "r_symlog_pred_soft", "r_symlog_pred_hard"]
    arr = np.array(rows)
    return {c: arr[:, i] for i, c in enumerate(cols)}, arr.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/pacman_classic_v82/best.pt")
    p.add_argument("--data-dir",   default="data/replay/pacman_classic")
    p.add_argument("--n-trajs",    type=int, default=100)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = EnsembleWorldModel.load(args.checkpoint).to(device)
    ensemble.eval()
    member = ensemble.members[0]
    print(f"loaded {args.checkpoint}  (members={len(ensemble.members)})")

    val_buf = SequenceReplayBuffer(args.data_dir, split="val")
    data, n = collect(member, val_buf, device, n_trajs=args.n_trajs)
    print(f"\ncollected {n} transitions from {args.n_trajs} val trajectories\n")

    # ----- (A)/(B) per-component aux loss -----
    print("=== (A) Dyn aux components (encoder path: dyn_state(z_all)) ===")
    print(f"  food_BCE      : {data['bce_enc'].mean():.4f}")
    print(f"  pos MSE       : {data['pos_err_enc'].mean():.6f}")
    print("\n=== (B) Dyn aux components (dynamics path: dyn_state(z_preds)) ===")
    print(f"  food_BCE      : {data['bce_dyn'].mean():.4f}")
    print(f"  pos MSE       : {data['pos_err_dyn'].mean():.6f}")

    # ----- (C) Sigmoid-sum vs true count -----
    print("\n=== (C) Food count: sigmoid-sum vs true ===")
    def stats(true, pred, label):
        bias  = (pred - true).mean()
        rmse  = np.sqrt(((pred - true) ** 2).mean())
        corr  = np.corrcoef(true, pred)[0, 1]
        print(f"  {label:22s}  true mean={true.mean():.2f}  pred mean={pred.mean():.2f}  "
              f"bias={bias:+.2f}  rmse={rmse:.2f}  corr={corr:.4f}")
    stats(data["count_t_true"],    data["count_t_enc"],        "soft enc  count_t")
    stats(data["count_t_true"],    data["count_t_enc_hard"],   "hard enc  count_t")
    stats(data["count_next_true"], data["count_next_dyn"],     "soft dyn  count_t+1")
    stats(data["count_next_true"], data["count_next_dyn_hard"],"hard dyn  count_t+1")

    # ----- (D) food_eaten prediction -----
    print("\n=== (D) food_eaten signal recovery (soft vs hard threshold) ===")
    fe_true   = data["food_eaten_true"]
    print(f"  true food_eaten distribution:")
    for v in [0, 1, 2, 3]:
        cnt = (fe_true == v).sum()
        if cnt > 0:
            print(f"    food_eaten = {v}  : {cnt:>5d}  ({cnt/len(fe_true)*100:.1f}%)")

    m0 = fe_true == 0
    m1 = fe_true == 1

    def report_delta(delta_s, delta_c, label):
        print(f"\n  [{label}]")
        print(f"    fe=0 (n={m0.sum()}):  signed mean/std = {delta_s[m0].mean():+.3f} / {delta_s[m0].std():.3f}    clamped mean = {delta_c[m0].mean():.3f}    FP rate (>0.5) = {(delta_s[m0] > 0.5).mean():.3f}")
        print(f"    fe=1 (n={m1.sum()}):  signed mean/std = {delta_s[m1].mean():+.3f} / {delta_s[m1].std():.3f}    clamped mean = {delta_c[m1].mean():.3f}    sign-err = {(delta_s[m1] < 0).mean():.3f}   FN rate (<0.5) = {(delta_s[m1] < 0.5).mean():.3f}")
    report_delta(data["delta_signed"], data["delta_clamp"],  "SOFT path (current behaviour)")
    report_delta(data["delta_hard_s"], data["delta_hard_c"], "HARD path (sigmoid>0.5 threshold)")

    # ----- (E) reward error decomposition -----
    print("\n=== (E) Reward MSE breakdown (symlog space, 1-step) ===")
    err_soft = (data["r_symlog_pred_soft"] - data["r_symlog_true"]) ** 2
    err_hard = (data["r_symlog_pred_hard"] - data["r_symlog_true"]) ** 2
    r_true = data["r_symlog_true"]
    baseline_mean = ((r_true - r_true.mean()) ** 2).mean()
    baseline_zero = (r_true ** 2).mean()

    print(f"  {'split':28s}  {'SOFT MSE':>10s}  {'HARD MSE':>10s}")
    print(f"  {'overall':28s}  {err_soft.mean():>10.4f}  {err_hard.mean():>10.4f}")
    print(f"  {'fe=0':28s}  {err_soft[m0].mean():>10.4f}  {err_hard[m0].mean():>10.4f}    n={m0.sum()}")
    print(f"  {'fe=1':28s}  {err_soft[m1].mean():>10.4f}  {err_hard[m1].mean():>10.4f}    n={m1.sum()}")
    m2plus = fe_true >= 2
    if m2plus.sum() > 0:
        print(f"  {'fe>=2':28s}  {err_soft[m2plus].mean():>10.4f}  {err_hard[m2plus].mean():>10.4f}    n={m2plus.sum()}")
    print(f"\n  baseline constant-mean MSE: {baseline_mean:.4f}")
    print(f"  baseline constant-zero MSE: {baseline_zero:.4f}")
    print(f"  threshold target          : 0.10")

    # ----- summary verdict -----
    print("\n=== Verdict ===")
    food_bce = data["bce_dyn"].mean()
    improv  = err_soft.mean() - err_hard.mean()
    rel     = improv / err_soft.mean() * 100 if err_soft.mean() > 0 else 0.0
    print(f"  food_BCE (dynamics): {food_bce:.4f}")
    print(f"  SOFT  reward_mse  : {err_soft.mean():.4f}")
    print(f"  HARD  reward_mse  : {err_hard.mean():.4f}")
    print(f"  Δ from threshold  : {improv:+.4f}  ({rel:+.1f}% relative)")
    if rel > 30:
        print("  → STRONG signal: hard threshold removes most of the sum noise.")
        print("    Training with STE should break the current plateau.")
    elif rel > 10:
        print("  → MODERATE signal: hard threshold helps, but other noise remains.")
    else:
        print("  → WEAK signal: thresholding alone insufficient; other bottleneck dominates.")


if __name__ == "__main__":
    main()
