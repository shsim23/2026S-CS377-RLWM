"""Intrinsic world-model evaluation — no policy needed (spec §10).

  * k-step open-loop rollout: warm the posterior on a real context window, then
    roll the PRIOR forward N steps (no observations, real actions), decoding at
    each horizon. Reports dynamic-state recon error, reward MSE (raw space via
    symexp), and continue accuracy vs horizon.
  * one-step metrics: decoder recon error, reward MSE, continue accuracy.
  * collapse metrics: per-group categorical entropy + count of near-deterministic
    (collapsed) groups. Healthy = no collapsed groups, entropy well above zero.

Cross-layout generalization (spec §10) = run the same metrics on a dataset
collected from the held-out TEST layout pool.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .nn import OneHotCategoricalST, symexp
from .rssm import extract_dynamic, DYN_BINARY_MASK
from .world_model import DreamerWorldModel

COLLAPSE_ENTROPY_NATS = 0.1   # group entropy below this ⇒ "collapsed"


def _dyn_errors(recon_state: torch.Tensor, target_state: torch.Tensor):
    """recon_state / target are actual dynamic-state values (B,T,460)."""
    tgt = extract_dynamic(target_state)
    mask = torch.as_tensor(DYN_BINARY_MASK, device=recon_state.device)
    cont_mse = ((recon_state[..., ~mask] - tgt[..., ~mask]) ** 2).mean().item()
    bin_pred = (recon_state[..., mask] > 0.5).float()
    bin_acc = (bin_pred == (tgt[..., mask] > 0.5).float()).float().mean().item()
    return cont_mse, bin_acc


@torch.no_grad()
def one_step_and_collapse(model: DreamerWorldModel, batch: dict, context: int = 0) -> dict:
    out = model.observe(batch["states"], batch["actions"], batch["is_first"])
    sl = slice(context, None)

    recon_state = model.decoder.reconstruct(out["recon"])
    cont_mse, bin_acc = _dyn_errors(recon_state[:, sl], batch["states"][:, sl])

    reward_pred = model.reward_from_logits(out["reward_logits"])
    reward_mse = ((reward_pred - batch["rewards"])[:, sl] ** 2).mean().item()

    cont_acc = (((out["cont_logits"] > 0).float() == batch["continues"])[:, sl]).float().mean().item()

    # Collapse: per-group entropy of the posterior.
    post = OneHotCategoricalST(out["post_logits"], model.cfg.unimix)
    ent_per_group = post.entropy_per_group()[:, sl]        # (B, T, G)
    mean_group_ent = ent_per_group.mean().item()
    # A group is "collapsed" if its entropy is near-zero on average.
    group_ent_mean = ent_per_group.reshape(-1, ent_per_group.shape[-1]).mean(0)
    n_collapsed = int((group_ent_mean < COLLAPSE_ENTROPY_NATS).sum().item())

    return {
        "one_step/recon_cont_mse": cont_mse,
        "one_step/recon_bin_acc": bin_acc,
        "one_step/reward_mse": reward_mse,
        "one_step/cont_acc": cont_acc,
        "collapse/mean_group_entropy": mean_group_ent,
        "collapse/n_collapsed_groups": n_collapsed,
        "collapse/max_entropy": float(np.log(model.cfg.classes)),
    }


@torch.no_grad()
def k_step_rollout(model: DreamerWorldModel, batch: dict, context: int, horizon: int) -> dict:
    """Warm on [0:context] with the posterior, then open-loop the prior for
    `horizon` steps using the real actions. Returns per-horizon arrays."""
    states, actions = batch["states"], batch["actions"]
    is_first = batch["is_first"]
    B, L, _ = states.shape
    horizon = min(horizon, L - context)

    e = model.embed_layout(states)                         # (B, L, e_dim)

    warm = model.observe(states[:, :context], actions[:, :context], is_first[:, :context])
    h = warm["h"][:, -1]
    z = warm["z"][:, -1]

    cont_mse_h, reward_mse_h, cont_acc_h = [], [], []
    for k in range(horizon):
        t = context + k
        h = model.seq(h, model._flat(z), actions[:, t], e[:, t])
        prior_logits = model.prior(h)
        z = OneHotCategoricalST(prior_logits, model.cfg.unimix).sample_st()
        z_flat = model._flat(z)

        recon_state = model.decoder.reconstruct(model.decoder(h, z_flat))
        cmse, _ = _dyn_errors(recon_state.unsqueeze(1), states[:, t:t + 1])
        cont_mse_h.append(cmse)

        reward_pred = model.reward_from_logits(model.reward_head(h, z_flat))
        reward_mse_h.append(((reward_pred - batch["rewards"][:, t]) ** 2).mean().item())

        cont_pred = (model.cont_head(h, z_flat) > 0).float()
        cont_acc_h.append((cont_pred == batch["continues"][:, t]).float().mean().item())

    return {
        "kstep/recon_cont_mse": np.array(cont_mse_h),
        "kstep/reward_mse": np.array(reward_mse_h),
        "kstep/cont_acc": np.array(cont_acc_h),
    }


@torch.no_grad()
def evaluate(model: DreamerWorldModel, replay, context: int, horizon: int,
             n_windows: int, device, seed: int = 0) -> dict:
    """Aggregate one-step + collapse + k-step metrics over `n_windows`."""
    model.eval()
    one_step_acc: dict[str, list] = {}
    kstep_acc: dict[str, list] = {}
    for batch in replay.iter_eval_windows(n_windows, device=device, seed=seed):
        batch = {k: v.unsqueeze(0).to(device) if v.dim() >= 1 else v.to(device)
                 for k, v in batch.items()}
        os_m = one_step_and_collapse(model, batch, context)
        for k, v in os_m.items():
            one_step_acc.setdefault(k, []).append(v)
        ks_m = k_step_rollout(model, batch, context, horizon)
        for k, v in ks_m.items():
            kstep_acc.setdefault(k, []).append(v)

    metrics = {k: float(np.mean(v)) for k, v in one_step_acc.items()}
    for k, v in kstep_acc.items():
        arr = np.stack(v, axis=0).mean(axis=0)             # mean over windows → per-horizon
        metrics[k + "_mean"] = float(arr.mean())
        metrics[k + "_final"] = float(arr[-1])
        metrics[k + "_curve"] = arr.tolist()
    return metrics
