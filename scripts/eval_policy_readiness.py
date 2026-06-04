"""Policy-readiness eval for a WM checkpoint.

Beyond the standard k-step MSE: measures whether predicted reward preserves
ORDER (the only thing a policy actually needs from a critic-free planner).

Metrics
-------
  - reward MSE / latent MSE / done err     (sanity, matches eval_world_model.py)
  - sign accuracy                          (sign(pred) vs true)
  - Pearson / Spearman correlation         (pred_r vs true_r over all (traj,t))
  - per-class mean reward                  (food_eaten=0 vs food_eaten=1 buckets)
  - food-eaten ROC-AUC                     (pred_r as score for true food event)
  - GRU-warmup variant                     (eval with P real burn-in steps)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model import EnsembleWorldModel, SequenceReplayBuffer

_PELLET_VALUE = 1.0
_FOOD_EATEN_THRESHOLD = 0.5   # true_r > threshold * pellet ⇒ food eaten


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(); y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return _pearson(rx.astype(np.float64), ry.astype(np.float64))


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via rank statistic (Mann-Whitney U)."""
    pos = labels > 0.5
    neg = ~pos
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def eval_policy_readiness(
    ensemble: EnsembleWorldModel,
    buffer: SequenceReplayBuffer,
    K: int = 10,
    N: int = 100,
    seed: int = 0,
    warmup: int = 0,
) -> dict:
    device = next(ensemble.parameters()).device
    latent_errs, reward_errs, done_errs = [], [], []
    pred_r_all, true_r_all, true_done_all = [], [], []

    for traj in buffer.sample_trajectories(N, seed=seed):
        states  = torch.from_numpy(traj["states"]).float().to(device)
        actions = torch.from_numpy(traj["actions"]).long().to(device)
        rewards = traj["rewards"]
        dones   = traj["dones"]
        T = len(states)
        if T < warmup + K + 1:
            continue

        # ---- warmup ----
        if warmup > 0:
            z, h = ensemble.warmup_h(
                states[:warmup + 1].unsqueeze(0),
                actions[:warmup + 1].unsqueeze(0),
            )
        else:
            z, h = ensemble.encode(states[0:1])

        # rollout K steps from position `warmup`
        offset = warmup
        for t in range(K):
            a = actions[offset + t: offset + t + 1]
            out = ensemble.imagine_step(z, h, a)

            z_true, _ = ensemble.encode(states[offset + t + 1: offset + t + 2])
            latent_errs.append(((out["z_next"] - z_true) ** 2).mean().item())
            pred_r = out["reward"].item()
            true_r = float(rewards[offset + t])
            true_d = float(dones[offset + t])
            reward_errs.append((pred_r - true_r) ** 2)
            done_errs.append(abs(out["done"].item() - true_d))

            pred_r_all.append(pred_r)
            true_r_all.append(true_r)
            true_done_all.append(true_d)

            if out["done"].item() > 0.5:
                break
            z, h = out["z_next"], out["h_next"]

    pred_r = np.array(pred_r_all)
    true_r = np.array(true_r_all)
    food_eaten = (true_r > _STEP_PENALTY + _FOOD_EATEN_THRESHOLD).astype(np.float64)
    death      = (true_r < -1.0).astype(np.float64)   # -10.01 / -9.01

    # "positive event" = reward clearly above 0 (food eaten or large bonus)
    pred_pos = (pred_r > 0.0).astype(np.float64)
    true_pos = (true_r > 0.0).astype(np.float64)
    sign_acc = float((pred_pos == true_pos).mean())

    # per-class buckets
    cls0 = food_eaten < 0.5
    cls1 = ~cls0
    res = {
        "k_step_latent_mse": float(np.mean(latent_errs)),
        "k_step_reward_mse": float(np.mean(reward_errs)),
        "k_step_done_err":   float(np.mean(done_errs)),
        "n_samples":         int(len(pred_r)),
        "n_food_eaten":      int(cls1.sum()),
        "n_death":           int(death.sum()),
        "food_eaten_frac":   float(cls1.mean()),
        "sign_accuracy":     sign_acc,            # pred>0 vs true>0
        "pearson_r":         _pearson(pred_r, true_r),
        "spearman_r":        _spearman(pred_r, true_r),
        "food_eaten_auc":    _roc_auc(pred_r, food_eaten),
        "death_recall_low":  (float((pred_r[death > 0.5] < pred_r[death < 0.5].mean()).mean())
                              if death.any() and (1 - death).any() else float("nan")),
        "mean_pred_r_cls0":  float(pred_r[cls0].mean()) if cls0.any() else float("nan"),
        "mean_true_r_cls0":  float(true_r[cls0].mean()) if cls0.any() else float("nan"),
        "mean_pred_r_cls1":  float(pred_r[cls1].mean()) if cls1.any() else float("nan"),
        "mean_true_r_cls1":  float(true_r[cls1].mean()) if cls1.any() else float("nan"),
        "separation_pred":   (float(pred_r[cls1].mean() - pred_r[cls0].mean())
                              if cls0.any() and cls1.any() else float("nan")),
        "separation_true":   (float(true_r[cls1].mean() - true_r[cls0].mean())
                              if cls0.any() and cls1.any() else float("nan")),
    }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data/replay/pacman_classic")
    p.add_argument("--split", default="val")
    p.add_argument("--k-step", type=int, default=10)
    p.add_argument("--n-trajs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=0,
                   help="Real burn-in steps for GRU before rollout")
    args = p.parse_args()

    ensemble = EnsembleWorldModel.load(args.checkpoint)
    ensemble.eval()
    buf = SequenceReplayBuffer(args.data_dir, split=args.split)

    print(f"\n=== Policy-Readiness Eval (warmup={args.warmup}) ===")
    res = eval_policy_readiness(
        ensemble, buf, K=args.k_step, N=args.n_trajs,
        seed=args.seed, warmup=args.warmup,
    )
    for k, v in res.items():
        if isinstance(v, int):
            print(f"  {k:22s}: {v}")
        else:
            print(f"  {k:22s}: {v:.4f}")


if __name__ == "__main__":
    main()
