"""World-model training objective (spec §6).

    L(φ) = E_q [ Σ_t  β_pred·L_pred + β_dyn·L_dyn + β_rep·L_rep ]
    β_pred = 1.0,  β_dyn = 0.5,  β_rep = 0.1

    L_pred = −ln p(x_dyn | z,h) − ln p(r | z,h) − ln p(c | z,h)
    L_dyn  = max(1, KL[ sg(q(z|h,x)) ‖     p(z|h)  ])
    L_rep  = max(1, KL[     q(z|h,x)  ‖ sg(p(z|h)) ])

Free bits = 1 nat clips each KL below 1 nat. No KL annealing, no weight decay,
no dropout, and — unlike the legacy v10c model — NO variance regularizer
(the categorical latent + free-bits KL is the collapse-prevention mechanism,
spec §4.1 / §6).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import symlog, two_hot_encode, categorical_kl
from .rssm import extract_dynamic, DYN_BINARY_MASK


def _recon_loss(recon: torch.Tensor, target_state: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Negative log-likelihood of the dynamic state.

    Continuous dims: symlog + Gaussian (squared error). Binary dims (ghost
    alive/valid flags + food cells): Bernoulli / BCE-with-logits. Both summed
    over feature dims, averaged over (B, T)."""
    target = extract_dynamic(target_state)                 # (B, T, 460)
    mask = torch.as_tensor(DYN_BINARY_MASK, device=recon.device)

    cont_pred = recon[..., ~mask]
    cont_tgt = symlog(target[..., ~mask])
    l_cont = ((cont_pred - cont_tgt) ** 2).sum(dim=-1)     # (B, T)

    bin_pred = recon[..., mask]
    bin_tgt = (target[..., mask] > 0.5).float()
    l_bin = F.binary_cross_entropy_with_logits(bin_pred, bin_tgt, reduction="none").sum(dim=-1)

    nll = l_cont + l_bin                                   # (B, T)
    return nll, {"recon_cont": l_cont.mean().item(), "recon_bin": l_bin.mean().item()}


def compute_loss(outputs: dict, batch: dict, reward_bins: torch.Tensor,
                 beta_pred: float = 1.0, beta_dyn: float = 0.5, beta_rep: float = 0.1,
                 free_nats: float = 1.0, context: int = 0) -> tuple[torch.Tensor, dict]:
    """Compute the world-model loss. `context` excludes the first `context`
    steps (h_0 warm-up from real history) from every term (spec §7)."""
    states = batch["states"]
    rewards = batch["rewards"]
    continues = batch["continues"]

    sl = slice(context, None)

    # ---- L_pred ----
    recon_nll, recon_metrics = _recon_loss(outputs["recon"], states)

    reward_target_symlog = symlog(rewards)
    two_hot = two_hot_encode(reward_target_symlog, reward_bins).detach()  # sg on target
    log_probs = F.log_softmax(outputs["reward_logits"], dim=-1)
    reward_nll = -(two_hot * log_probs).sum(dim=-1)                       # (B, T)

    cont_nll = F.binary_cross_entropy_with_logits(
        outputs["cont_logits"], continues, reduction="none")             # (B, T)

    L_pred = (recon_nll[:, sl] + reward_nll[:, sl] + cont_nll[:, sl]).mean()

    # ---- L_dyn / L_rep (KL balancing + free bits) ----
    post = outputs["post_logits"]
    prior = outputs["prior_logits"]
    kl_dyn = categorical_kl(post.detach(), prior)          # sg(post) ‖ prior  → trains prior
    kl_rep = categorical_kl(post, prior.detach())          # post ‖ sg(prior)  → trains posterior
    L_dyn = torch.clamp(kl_dyn[:, sl], min=free_nats).mean()
    L_rep = torch.clamp(kl_rep[:, sl], min=free_nats).mean()

    loss = beta_pred * L_pred + beta_dyn * L_dyn + beta_rep * L_rep

    # ---- diagnostics ----
    with torch.no_grad():
        reward_pred = _reward_scalar(outputs["reward_logits"], reward_bins)
        reward_mse = ((reward_pred - rewards)[:, sl] ** 2).mean().item()
        cont_acc = (((outputs["cont_logits"] > 0).float() == continues)[:, sl]).float().mean().item()

    metrics = {
        "loss": loss.item(),
        "L_pred": L_pred.item(),
        "L_dyn": L_dyn.item(),
        "L_rep": L_rep.item(),
        "reward_nll": reward_nll[:, sl].mean().item(),
        "cont_nll": cont_nll[:, sl].mean().item(),
        "kl_dyn_raw": kl_dyn[:, sl].mean().item(),
        "kl_rep_raw": kl_rep[:, sl].mean().item(),
        "reward_mse": reward_mse,
        "cont_acc": cont_acc,
        **recon_metrics,
    }
    return loss, metrics


def _reward_scalar(reward_logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    from .nn import two_hot_decode, symexp
    probs = F.softmax(reward_logits, dim=-1)
    return symexp(two_hot_decode(probs, bins))
