"""RSSM components for the state-based DreamerV3 world model (spec §4–§5).

Six components (spec §4):
    Sequence model:        h_t  = f(h_{t-1}, z_{t-1}, a_{t-1}, e)
    Encoder (posterior):   z_t  ~ q(z_t | h_t, x_t)
    Dynamics pred (prior): ẑ_t  ~ p(ẑ_t | h_t)
    Reward predictor:      r̂_t  ~ p(r_t | h_t, z_t)   (two-hot symlog)
    Continue predictor:    ĉ_t  ~ p(c_t | h_t, z_t)   (Bernoulli)
    Decoder:               x̂_t  ~ p(x_dyn | h_t, z_t)

Layout conditioning (spec §2): a small MLP maps the static wall_mask → e, fed to
the sequence model so one model generalizes across maze layouts.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .nn import RMSNorm, mlp, OneHotCategoricalST, symlog

# --- State-vector slices (pacman_env.state.StateBuilder, 901-dim) ----------- #
PAC_SLICE = slice(0, 2)
GHOST_SLICE = slice(2, 18)
FOOD_SLICE = slice(18, 459)
WALL_SLICE = slice(459, 900)     # static layout → conditioning e (NOT reconstructed)
POWER_SLICE = slice(900, 901)
STATE_DIM = 901
WALL_DIM = 441
DYN_DIM = 460                    # 2 + 16 + 441 + 1 (everything except walls)


def extract_dynamic(state: torch.Tensor) -> torch.Tensor:
    """Full 901-d state → 460-d dynamic target (drop the static wall_mask)."""
    return torch.cat([
        state[..., PAC_SLICE],
        state[..., GHOST_SLICE],
        state[..., FOOD_SLICE],
        state[..., POWER_SLICE],
    ], dim=-1)


def _dyn_binary_mask() -> np.ndarray:
    """Boolean mask over the 460 dynamic dims: True = binary (BCE), False =
    continuous (symlog+MSE). Binary = ghost alive/valid flags + food cells."""
    m = np.zeros(DYN_DIM, dtype=bool)
    # ghost block occupies dyn indices [2:18] = 4 ghosts × [x, y, alive, valid]
    for i in range(4):
        base = 2 + i * 4
        m[base + 2] = True   # alive
        m[base + 3] = True   # valid
    m[18:459] = True          # food mask (441 binary cells)
    return m


DYN_BINARY_MASK = _dyn_binary_mask()      # (460,) bool


# --------------------------------------------------------------------------- #
class LayoutEmbedder(nn.Module):
    """wall_mask (441) → layout embedding e (e_dim). Computed once per episode
    and broadcast over time (spec §2)."""

    def __init__(self, e_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.net = mlp(WALL_DIM, hidden, e_dim, layers=1)

    def forward(self, wall_mask: torch.Tensor) -> torch.Tensor:
        return self.net(wall_mask)


class SequenceModel(nn.Module):
    """Layout-conditioned GRU core: h_t = f(h_{t-1}, z_{t-1}, a_{t-1}, e)."""

    def __init__(self, stoch_dim: int, action_dim: int, e_dim: int,
                 deter: int = 256, hidden: int = 256, action_emb: int = 16):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, action_emb)
        self.in_proj = nn.Sequential(
            nn.Linear(stoch_dim + action_emb + e_dim, hidden),
            RMSNorm(hidden), nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden, deter)
        self.deter = deter

    def forward(self, h_prev: torch.Tensor, z_prev_flat: torch.Tensor,
                a_prev: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        a_emb = self.action_embed(a_prev)
        x = torch.cat([z_prev_flat, a_emb, e], dim=-1)
        x = self.in_proj(x)
        return self.gru(x, h_prev)


class Encoder(nn.Module):
    """Posterior q(z | h, x): combines the recurrent state with symlog(obs)."""

    def __init__(self, deter: int, groups: int, classes: int, hidden: int = 256):
        super().__init__()
        self.groups = groups
        self.classes = classes
        self.net = mlp(deter + STATE_DIM, hidden, groups * classes, layers=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([h, symlog(x)], dim=-1)
        logits = self.net(feat)
        return logits.reshape(*logits.shape[:-1], self.groups, self.classes)


class DynamicsPredictor(nn.Module):
    """Prior p(ẑ | h). Used (without observations) at imagination time."""

    def __init__(self, deter: int, groups: int, classes: int, hidden: int = 256):
        super().__init__()
        self.groups = groups
        self.classes = classes
        self.net = mlp(deter, hidden, groups * classes, layers=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.net(h)
        return logits.reshape(*logits.shape[:-1], self.groups, self.classes)


class RewardHead(nn.Module):
    """Two-hot symlog reward head (spec §5.1). Output layer is zero-initialised
    so initial reward predictions start at 0 (symexp(0)=0)."""

    def __init__(self, deter: int, stoch_dim: int, num_bins: int = 255,
                 vmin: float = -20.0, vmax: float = 20.0, hidden: int = 256):
        super().__init__()
        self.net = mlp(deter + stoch_dim, hidden, num_bins, layers=1)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("bins", torch.linspace(vmin, vmax, num_bins))

    def forward(self, h: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        """Returns logits over the bins (apply softmax + two_hot_decode for the scalar)."""
        return self.net(torch.cat([h, z_flat], dim=-1))


class ContinueHead(nn.Module):
    """Bernoulli continue = 1 − done (spec §5.2). Returns a logit."""

    def __init__(self, deter: int, stoch_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp(deter + stoch_dim, hidden, 1, layers=1)

    def forward(self, h: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z_flat], dim=-1)).squeeze(-1)


class Decoder(nn.Module):
    """Reconstruct the 460-d dynamic state from {h, z} (spec §5.3).

    Outputs raw head values; the loss splits them into continuous (symlog+MSE)
    and binary (BCE-with-logits) components via DYN_BINARY_MASK. `reconstruct`
    converts raw outputs back to actual state values for eval/debug."""

    def __init__(self, deter: int, stoch_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp(deter + stoch_dim, hidden, DYN_DIM, layers=2)

    def forward(self, h: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z_flat], dim=-1))

    @staticmethod
    def reconstruct(raw: torch.Tensor) -> torch.Tensor:
        """Raw head output → actual dynamic-state values (symexp on continuous
        dims, sigmoid on binary dims)."""
        from .nn import symexp
        mask = torch.as_tensor(DYN_BINARY_MASK, device=raw.device)
        cont = symexp(raw)
        binv = torch.sigmoid(raw)
        return torch.where(mask, binv, cont)
