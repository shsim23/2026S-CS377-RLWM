"""Neural-net primitives for the DreamerV3-style RSSM (spec §4.2, §5).

Faithful to DreamerV3: RMSNorm + SiLU, symlog input squashing, straight-through
categorical latents with 1% unimix, and two-hot symlog regression for the reward
head. Kept deliberately small (state-based, MLP only — no CNN; spec §9).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the project's symlog/symexp so behaviour matches the rest of the repo.
from ..utils import symlog, symexp  # noqa: F401  (re-exported)


# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (DreamerV3 default norm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 1) -> nn.Sequential:
    """Linear → (RMSNorm → SiLU → Linear) × layers. `layers` = # hidden layers.

    layers=1 → in→hidden→out with one RMSNorm/SiLU block on the hidden.
    """
    mods: list[nn.Module] = [nn.Linear(in_dim, hidden), RMSNorm(hidden), nn.SiLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), RMSNorm(hidden), nn.SiLU()]
    mods += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*mods)


# --------------------------------------------------------------------------- #
class OneHotCategoricalST:
    """Vector of categorical distributions with 1% unimix and straight-through
    one-hot sampling (spec §4.1).

    `logits` shape: (..., groups, classes). All operations act on the last dim.
    """

    def __init__(self, logits: torch.Tensor, unimix: float = 0.01):
        # Unimix: 0.99·softmax(logits) + 0.01·uniform. Reparameterised back into
        # logits so `.logits` and `.probs` stay consistent for KL/entropy.
        probs = F.softmax(logits, dim=-1)
        if unimix > 0.0:
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1.0 - unimix) * probs + unimix * uniform
            logits = torch.log(probs + 1e-8)
        self.logits = logits
        self.probs = probs

    def sample_st(self) -> torch.Tensor:
        """Straight-through one-hot sample: one-hot on forward, softmax grad on
        backward. Returns (..., groups, classes)."""
        idx = torch.distributions.Categorical(probs=self.probs).sample()
        onehot = F.one_hot(idx, num_classes=self.probs.shape[-1]).type(self.probs.dtype)
        # ST estimator: detach the (sample − probs) so the backward pass sees probs.
        return onehot + (self.probs - self.probs.detach())

    def mode(self) -> torch.Tensor:
        idx = self.probs.argmax(dim=-1)
        return F.one_hot(idx, num_classes=self.probs.shape[-1]).type(self.probs.dtype)

    def entropy(self) -> torch.Tensor:
        # Per-sample entropy summed over groups (last two dims -> scalar per batch).
        ent = -(self.probs * torch.log(self.probs + 1e-8)).sum(dim=-1)
        return ent.sum(dim=-1)

    def entropy_per_group(self) -> torch.Tensor:
        """Entropy of each categorical group (..., groups) — for collapse metrics."""
        return -(self.probs * torch.log(self.probs + 1e-8)).sum(dim=-1)


def categorical_kl(post_logits: torch.Tensor, prior_logits: torch.Tensor,
                   unimix: float = 0.01) -> torch.Tensor:
    """KL[ post || prior ] summed over the categorical groups.

    Inputs are raw logits of shape (..., groups, classes). Returns (...,) — one
    KL value per sequence position, summed across the 32 groups.
    """
    p = OneHotCategoricalST(post_logits, unimix)
    q = OneHotCategoricalST(prior_logits, unimix)
    kl = (p.probs * (torch.log(p.probs + 1e-8) - torch.log(q.probs + 1e-8))).sum(dim=-1)
    return kl.sum(dim=-1)  # sum over groups


# --------------------------------------------------------------------------- #
def two_hot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Two-hot encode scalar targets `x` over the (sorted, equally spaced) `bins`.

    Returns a soft label of shape (*x.shape, K) that places mass on the two
    nearest bins so its expectation equals `x` (clamped into the support).
    """
    K = bins.shape[0]
    x = x.clamp(bins[0], bins[-1])
    # Index of the lower bin.
    idx = torch.bucketize(x, bins, right=True) - 1
    idx = idx.clamp(0, K - 2)
    lo = bins[idx]
    hi = bins[idx + 1]
    w_hi = (x - lo) / (hi - lo + 1e-8)
    w_lo = 1.0 - w_hi
    out = torch.zeros(*x.shape, K, device=x.device, dtype=x.dtype)
    out.scatter_(-1, idx.unsqueeze(-1), w_lo.unsqueeze(-1))
    out.scatter_(-1, (idx + 1).unsqueeze(-1), w_hi.unsqueeze(-1))
    return out


def two_hot_decode(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected bin value E_b[b] under the softmax `probs` over `bins`."""
    return (probs * bins).sum(dim=-1)
