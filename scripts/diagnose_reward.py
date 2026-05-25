"""Reward-prediction diagnostics for a trained world model.

Sections:
  Phase 1.1 — baseline reward_mse vs trivial predictors
  Phase 1.2 — per-step (1..K) reward error decomposition
  Phase 1.3 — reward-class breakdown
  Phase 2.1 — linear probe on frozen latent
              (does the latent encode the info needed for reward?)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model import EnsembleWorldModel, SequenceReplayBuffer


# --------------------------------------------------------------------------- #
def load_all_transitions(buf: SequenceReplayBuffer, max_trajs: int | None = None):
    """Return concatenated arrays of (state, action, reward, done, next_state)."""
    trajs = buf.sample_trajectories(max_trajs or len(buf.episode_files), seed=0)
    Ss, As, Rs, Ds, Sn = [], [], [], [], []
    for traj in trajs:
        s = traj["states"]; a = traj["actions"]; r = traj["rewards"]; d = traj["dones"]
        if len(s) < 2:
            continue
        Ss.append(s[:-1]); As.append(a[:-1]); Rs.append(r[:-1])
        Ds.append(d[:-1]); Sn.append(s[1:])
    return (np.concatenate(Ss),  np.concatenate(As),
            np.concatenate(Rs),  np.concatenate(Ds), np.concatenate(Sn))


@torch.no_grad()
def encode_states(member, states_np: np.ndarray, device, batch=1024):
    out = []
    for i in range(0, len(states_np), batch):
        s = torch.from_numpy(states_np[i:i + batch]).float().to(device)
        z = member.encoder(s)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
def phase_1_1_baselines(R_train: np.ndarray, R_val: np.ndarray, model_mse: float):
    print("\n=== Phase 1.1 — Baselines on val set ===")
    print(f"  val n_transitions       : {len(R_val)}")
    print(f"  val reward mean         : {R_val.mean():+.4f}")
    print(f"  val reward std          : {R_val.std():.4f}")
    print(f"  val reward min / max    : {R_val.min():+.2f} / {R_val.max():+.2f}")
    print()
    print(f"  baseline constant-zero  : {((R_val - 0.0) ** 2).mean():.4f}")
    print(f"  baseline constant-mean  : {((R_val - R_train.mean()) ** 2).mean():.4f}  (mean={R_train.mean():+.4f})")
    print(f"  baseline median         : {((R_val - np.median(R_train)) ** 2).mean():.4f}  (median={np.median(R_train):+.4f})")
    print(f"  our v4 model (K-step)   : {model_mse:.4f}")


@torch.no_grad()
def phase_1_2_per_step(ensemble, val_buf, K: int = 10, N: int = 100, seed: int = 0):
    print("\n=== Phase 1.2 — Per-step reward_mse (k=1..K) ===")
    device = next(ensemble.parameters()).device
    per_step_errs = [[] for _ in range(K)]
    per_step_true = [[] for _ in range(K)]
    per_step_pred = [[] for _ in range(K)]

    for traj in val_buf.sample_trajectories(N, seed=seed):
        states  = torch.from_numpy(traj["states"]).float().to(device)
        actions = torch.from_numpy(traj["actions"]).long().to(device)
        rewards = traj["rewards"]
        T = len(states)
        if T < K + 1:
            continue
        z, h = ensemble.encode(states[0:1])
        for t in range(K):
            a = actions[t: t + 1]
            out = ensemble.imagine_step(z, h, a)
            r_pred = out["reward"].item()
            r_true = float(rewards[t])
            per_step_errs[t].append((r_pred - r_true) ** 2)
            per_step_true[t].append(r_true)
            per_step_pred[t].append(r_pred)
            z, h = out["z_next"], out["h_next"]

    for k in range(K):
        errs = per_step_errs[k]
        if errs:
            mean_err = np.mean(errs)
            pred = np.array(per_step_pred[k])
            true = np.array(per_step_true[k])
            print(f"  k={k+1:2d}  mse={mean_err:.4f}   n={len(errs):>4}   "
                  f"true_std={true.std():.3f}  pred_std={pred.std():.3f}  "
                  f"pred_mean={pred.mean():+.3f}")


@torch.no_grad()
def phase_1_3_reward_class(ensemble, val_buf, N: int = 100, seed: int = 0):
    print("\n=== Phase 1.3 — Error by reward class (1-step only) ===")
    device = next(ensemble.parameters()).device

    pairs = []  # (true_reward, pred_reward)
    for traj in val_buf.sample_trajectories(N, seed=seed):
        states  = torch.from_numpy(traj["states"]).float().to(device)
        actions = torch.from_numpy(traj["actions"]).long().to(device)
        rewards = traj["rewards"]
        T = len(states)
        for t in range(T - 1):
            z, h = ensemble.encode(states[t:t + 1])
            a = actions[t: t + 1]
            out = ensemble.imagine_step(z, h, a)
            pairs.append((float(rewards[t]), out["reward"].item()))

    pairs = np.array(pairs)
    true, pred = pairs[:, 0], pairs[:, 1]

    # define classes based on actual reward values present
    classes = {
        "death (≤ -5)":      true <= -5,
        "step penalty (-0.5,0)": (true > -0.5) & (true < 0),
        "food (0.5,1.5)":    (true > 0.5) & (true < 1.5),
        "ghost eaten (5,15)": (true > 5) & (true < 15),
        "win (>20)":         true > 20,
    }
    print(f"  total 1-step transitions: {len(true)}")
    print(f"  {'class':22s} {'count':>6s}  {'true_mean':>10s}  {'pred_mean':>10s}  {'mse':>8s}")
    for name, mask in classes.items():
        if mask.sum() == 0:
            continue
        t = true[mask]; p = pred[mask]
        print(f"  {name:22s} {mask.sum():>6d}  {t.mean():>+10.3f}  {p.mean():>+10.3f}  {((p - t) ** 2).mean():>8.4f}")


@torch.no_grad()
def measure_food_count_head(member, S_va, device, batch=1024):
    """If the model has a FoodCountHead, measure its accuracy on val states."""
    if not hasattr(member, "food_count_head"):
        print("\n=== Phase 2.0 — FoodCountHead accuracy ===")
        print("  (model has no food_count_head — skipping)")
        return
    print("\n=== Phase 2.0 — FoodCountHead accuracy on val states ===")
    preds, trues = [], []
    for i in range(0, len(S_va), batch):
        s = torch.from_numpy(S_va[i:i + batch]).float().to(device)
        z = member.encoder(s)
        p = member.food_count_head(z).cpu().numpy()
        t = ((s[:, 18:459] > 0.5).float().sum(dim=-1) / 100.0).cpu().numpy()
        preds.append(p); trues.append(t)
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    mse = ((preds - trues) ** 2).mean()
    print(f"  count predictions (normalized by /100):")
    print(f"    true mean / std    : {trues.mean():.4f} / {trues.std():.4f}")
    print(f"    pred mean / std    : {preds.mean():.4f} / {preds.std():.4f}")
    print(f"    pred MSE           : {mse:.6f}")
    print(f"    pred max-abs error : {np.max(np.abs(preds - trues)):.4f}")


@torch.no_grad()
def measure_dynamic_state_head(member, S_va, device, batch=1024):
    """If the model has a DynamicStateHead, decode val states and report
    per-component accuracy. Especially the food mask BCE — that's the key
    signal for whether the encoder actually preserves food positions."""
    if not hasattr(member, "dynamic_state_head"):
        print("\n=== Phase 2.0 — DynamicStateHead accuracy ===")
        print("  (model has no dynamic_state_head — skipping)")
        return
    print("\n=== Phase 2.0 — DynamicStateHead accuracy on val states ===")
    import torch.nn.functional as F
    pos_errs, ghost_errs, power_errs = [], [], []
    food_bces, food_accs, food_count_preds, food_count_trues = [], [], [], []
    food_cell_pos_preds, food_cell_pos_trues = [], []

    for i in range(0, len(S_va), batch):
        s = torch.from_numpy(S_va[i:i + batch]).float().to(device)
        z = member.encoder(s)
        pred = member.dynamic_state_head(z)
        pos_pred, ghost_pred, food_logits, power_pred = (
            pred[:, 0:2], pred[:, 2:18], pred[:, 18:459], pred[:, 459:460]
        )
        pos_true, ghost_true, food_true_raw, power_true = (
            s[:, 0:2], s[:, 2:18], s[:, 18:459], s[:, 900:901]
        )
        food_true = (food_true_raw > 0.5).float()

        pos_errs.append(((pos_pred - pos_true) ** 2).mean(dim=-1).cpu().numpy())
        ghost_errs.append(((ghost_pred - ghost_true) ** 2).mean(dim=-1).cpu().numpy())
        power_errs.append(((power_pred - power_true) ** 2).mean(dim=-1).cpu().numpy())

        food_bces.append(F.binary_cross_entropy_with_logits(
            food_logits, food_true, reduction="none").mean(dim=-1).cpu().numpy())
        # accuracy
        food_pred_binary = (food_logits > 0).float()
        food_accs.append((food_pred_binary == food_true).float().mean(dim=-1).cpu().numpy())
        # implied count
        food_count_preds.append(food_pred_binary.sum(dim=-1).cpu().numpy())
        food_count_trues.append(food_true.sum(dim=-1).cpu().numpy())

    print(f"  pacman pos      MSE : {np.mean(np.concatenate(pos_errs)):.6f}")
    print(f"  ghost slots     MSE : {np.mean(np.concatenate(ghost_errs)):.6f}")
    print(f"  power timer     MSE : {np.mean(np.concatenate(power_errs)):.6f}")
    print(f"  food mask       BCE : {np.mean(np.concatenate(food_bces)):.4f}")
    print(f"  food mask cell-accuracy : {np.mean(np.concatenate(food_accs)):.4f}  "
          f"(naive all-zero would be ≈ {1 - 100/441:.3f})")
    p = np.concatenate(food_count_preds); t = np.concatenate(food_count_trues)
    print(f"  food count (sum>0.5)  pred mean/std={p.mean():.2f}/{p.std():.2f}  "
          f"true mean/std={t.mean():.2f}/{t.std():.2f}  count MSE={((p-t)**2).mean():.2f}")


def phase_2_1_linear_probe(member, S_tr, A_tr, R_tr, Sn_tr,
                            S_va, A_va, R_va, Sn_va, device):
    print("\n=== Phase 2.1 — Linear probe on frozen latent ===")
    print("  encoding train/val states...")
    Z_tr  = encode_states(member, S_tr,  device)
    Zn_tr = encode_states(member, Sn_tr, device)
    Z_va  = encode_states(member, S_va,  device)
    Zn_va = encode_states(member, Sn_va, device)

    # one-hot action embedding
    n_actions = int(max(A_tr.max(), A_va.max()) + 1)
    def onehot(a):
        oh = np.zeros((len(a), n_actions), dtype=np.float32)
        oh[np.arange(len(a)), a] = 1.0
        return oh
    A_tr_oh, A_va_oh = onehot(A_tr), onehot(A_va)

    # ground-truth "food delta" boolean from state vectors (food slice = [18:459])
    def food_delta(s, sn):
        food_t  = (s [:, 18:459] > 0.5)
        food_n  = (sn[:, 18:459] > 0.5)
        # +1 cells that disappeared (pacman ate them)
        eaten = (food_t & ~food_n).sum(axis=1).astype(np.float32)
        return eaten
    fd_tr = food_delta(S_tr, Sn_tr)
    fd_va = food_delta(S_va, Sn_va)

    inputs = {
        "z_t only"               : (Z_tr,                                          Z_va),
        "z_t + a_t"              : (np.concatenate([Z_tr, A_tr_oh], axis=1),       np.concatenate([Z_va, A_va_oh], axis=1)),
        "z_t + z_{t+1}"          : (np.concatenate([Z_tr, Zn_tr],   axis=1),       np.concatenate([Z_va, Zn_va], axis=1)),
        "z_t + a_t + z_{t+1}"    : (np.concatenate([Z_tr, A_tr_oh, Zn_tr], axis=1),np.concatenate([Z_va, A_va_oh, Zn_va], axis=1)),
        "ORACLE: food_delta only": (fd_tr.reshape(-1, 1),                          fd_va.reshape(-1, 1)),
    }

    print(f"  fit ridge regressors (alpha=1.0) on {len(R_tr)} train transitions:")
    print(f"  {'input':30s}  {'train_mse':>10s}  {'val_mse':>10s}")
    for name, (X_tr, X_va) in inputs.items():
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, R_tr)
        tr_mse = ((reg.predict(X_tr) - R_tr) ** 2).mean()
        va_mse = ((reg.predict(X_va) - R_va) ** 2).mean()
        print(f"  {name:30s}  {tr_mse:>10.4f}  {va_mse:>10.4f}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/pacman_classic_v4/best.pt")
    p.add_argument("--data-dir",   default="data/replay/pacman_classic")
    p.add_argument("--k-step",     type=int, default=10)
    p.add_argument("--n-trajs",    type=int, default=100)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = EnsembleWorldModel.load(args.checkpoint).to(device)
    ensemble.eval()
    member = ensemble.members[0]
    print(f"loaded {args.checkpoint}  (members={len(ensemble.members)})")

    train_buf = SequenceReplayBuffer(args.data_dir, split="train")
    val_buf   = SequenceReplayBuffer(args.data_dir, split="val")

    print("\nloading train transitions (subset)...")
    S_tr, A_tr, R_tr, D_tr, Sn_tr = load_all_transitions(train_buf, max_trajs=800)
    print(f"  train transitions: {len(R_tr)}")
    print("loading val transitions (full)...")
    S_va, A_va, R_va, D_va, Sn_va = load_all_transitions(val_buf)
    print(f"  val transitions  : {len(R_va)}")

    # need our model's K-step reward_mse for context — reuse eval pipeline
    from world_model import evaluate_k_step_rollout
    metrics = evaluate_k_step_rollout(ensemble, val_buf,
                                      K=args.k_step, N=args.n_trajs, seed=0)
    phase_1_1_baselines(R_tr, R_va, metrics["k_step_reward_mse"])
    phase_1_2_per_step(ensemble, val_buf, K=args.k_step, N=args.n_trajs)
    phase_1_3_reward_class(ensemble, val_buf, N=args.n_trajs)
    measure_food_count_head(member, S_va, device)
    measure_dynamic_state_head(member, S_va, device)
    phase_2_1_linear_probe(member,
                           S_tr, A_tr, R_tr, Sn_tr,
                           S_va, A_va, R_va, Sn_va,
                           device)


if __name__ == "__main__":
    main()
