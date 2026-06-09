"""DreamerV3-style state-based RSSM world model (spec §4–§5, §12).

Scope (spec §1): world model ONLY — no actor/critic/value/returns/planning.
Predicts, from (state, action) sequences, the next latent state, reward, and
continue flag, and reconstructs the dynamic part of the state. Supports open-loop
imagination (prior rollout without observations) for the later policy phase.

Deviations from standard DreamerV3 (spec §9):
  * State-based MLP encoder/decoder (no CNN).
  * Decoder reconstructs the ~460-d DYNAMIC state only; the static wall_mask
    enters as the layout conditioning `e`, not as a reconstruction target.
  * Sequence model is conditioned on `e` for cross-layout generalization.
  * No actor/critic and associated machinery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import OneHotCategoricalST, two_hot_decode, symexp
from .rssm import (
    LayoutEmbedder, SequenceModel, Encoder, DynamicsPredictor,
    RewardHead, ContinueHead, Decoder, PositionHead, POS_DIMS,
    WALL_SLICE,
)


@dataclass
class WorldModelConfig:
    action_dim: int = 5
    groups: int = 32              # categorical latent groups (spec §4.1)
    classes: int = 32             # classes per group
    deter: int = 256              # GRU recurrent units h (spec §8)
    hidden: int = 256             # MLP hidden units
    e_dim: int = 32               # layout embedding dim
    action_emb: int = 16
    num_bins: int = 255           # two-hot reward buckets (spec §5.1)
    vmin: float = -20.0
    vmax: float = 20.0
    unimix: float = 0.01          # 1% unimix (spec §4.1)
    gru_blocks: int = 1           # >1 = block-diagonal GRU (DreamerV3 `blocks`)
    position_mode: str = "regress"  # "regress" (symlog-MSE scalar) | "twohot" (grid CE)
    pos_bins: int = 21              # grid bins per coordinate in twohot mode

    @property
    def stoch_dim(self) -> int:
        return self.groups * self.classes


class DreamerWorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig = WorldModelConfig()):
        super().__init__()
        self.cfg = cfg
        c = cfg
        self.layout_embedder = LayoutEmbedder(c.e_dim, c.hidden)
        self.seq = SequenceModel(c.stoch_dim, c.action_dim, c.e_dim,
                                 c.deter, c.hidden, c.action_emb, c.gru_blocks)
        self.encoder = Encoder(c.deter, c.groups, c.classes, c.hidden)
        self.prior = DynamicsPredictor(c.deter, c.groups, c.classes, c.hidden)
        self.reward_head = RewardHead(c.deter, c.stoch_dim, c.num_bins,
                                      c.vmin, c.vmax, c.hidden)
        self.cont_head = ContinueHead(c.deter, c.stoch_dim, c.hidden)
        self.decoder = Decoder(c.deter, c.stoch_dim, c.hidden)
        self.position_head = (
            PositionHead(c.deter, c.stoch_dim, n_coords=len(POS_DIMS),
                         n_bins=c.pos_bins, hidden=c.hidden)
            if c.position_mode == "twohot" else None
        )
        # expose for downstream / interface convenience
        self.latent_dim = c.stoch_dim
        self.gru_hidden = c.deter

    # ------------------------------------------------------------------ #
    def initial_state(self, batch: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or next(self.parameters()).device
        h = torch.zeros(batch, self.cfg.deter, device=device)
        z = torch.zeros(batch, self.cfg.groups, self.cfg.classes, device=device)
        return h, z

    def embed_layout(self, state_or_wall: torch.Tensor) -> torch.Tensor:
        """Accept either a full 901-d state (extracts wall_mask) or a 441-d
        wall_mask directly, and return the layout embedding e."""
        if state_or_wall.shape[-1] == 901:
            wall = state_or_wall[..., WALL_SLICE]
        else:
            wall = state_or_wall
        return self.layout_embedder(wall)

    def _flat(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(*z.shape[:-2], self.cfg.stoch_dim)

    def reward_from_logits(self, reward_logits: torch.Tensor) -> torch.Tensor:
        """Two-hot softmax → symlog scalar → symexp → raw reward."""
        probs = F.softmax(reward_logits, dim=-1)
        symlog_r = two_hot_decode(probs, self.reward_head.bins)
        return symexp(symlog_r)

    # ------------------------------------------------------------------ #
    def observe(self, states: torch.Tensor, actions: torch.Tensor,
                is_first: torch.Tensor, e: Optional[torch.Tensor] = None) -> dict:
        """Teacher-forced posterior rollout over a (B, L, ...) sequence.

        `actions[:, t]` is the action that led INTO `states[:, t]` (a_{t-1}).
        `is_first[:, t]` resets the recurrent carry at episode boundaries inside
        the window. `e` may be precomputed per-step (B, L, e_dim); if None it is
        computed from each step's wall_mask (handles windows spanning layouts).
        Returns per-step stacks used by the loss.
        """
        B, L, _ = states.shape
        if e is None:
            e = self.embed_layout(states)               # (B, L, e_dim)

        h, z = self.initial_state(B, states.device)
        hs, zs, post_l, prior_l = [], [], [], []
        for t in range(L):
            reset = is_first[:, t].unsqueeze(-1).float()  # 1 where new episode
            h = h * (1.0 - reset)
            z = z * (1.0 - reset.unsqueeze(-1))
            h = self.seq(h, self._flat(z), actions[:, t], e[:, t])
            prior_logits = self.prior(h)
            post_logits = self.encoder(h, states[:, t])
            z = OneHotCategoricalST(post_logits, self.cfg.unimix).sample_st()
            hs.append(h); zs.append(z)
            post_l.append(post_logits); prior_l.append(prior_logits)

        h_seq = torch.stack(hs, dim=1)                   # (B, L, deter)
        z_seq = torch.stack(zs, dim=1)                   # (B, L, G, C)
        post_logits = torch.stack(post_l, dim=1)
        prior_logits = torch.stack(prior_l, dim=1)
        z_flat = self._flat(z_seq)

        recon = self.decoder(h_seq, z_flat)              # (B, L, 460)
        reward_logits = self.reward_head(h_seq, z_flat)  # (B, L, K)
        cont_logits = self.cont_head(h_seq, z_flat)      # (B, L)
        position_logits = (self.position_head(h_seq, z_flat)   # (B, L, 10, bins)
                           if self.position_head is not None else None)

        return {
            "h": h_seq, "z": z_seq,
            "post_logits": post_logits, "prior_logits": prior_logits,
            "recon": recon, "reward_logits": reward_logits, "cont_logits": cont_logits,
            "position_logits": position_logits,
        }

    # ------------------------------------------------------------------ #
    # Interface contract for the later policy phase (spec §12).
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor, h: torch.Tensor, a_prev: torch.Tensor,
               e: torch.Tensor, z_prev: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Posterior step (uses the observation x). Advances the recurrent state
        with the previous action then samples z ~ q(z | h, x).

        z_prev defaults to zeros (use at the first step). `e` is the layout
        embedding (B, e_dim); pass `embed_layout(state)` if you only have walls.
        """
        if z_prev is None:
            z_prev = torch.zeros(x.shape[0], self.cfg.groups, self.cfg.classes,
                                 device=x.device)
        h = self.seq(h, self._flat(z_prev), a_prev, e)
        post_logits = self.encoder(h, x)
        z = OneHotCategoricalST(post_logits, self.cfg.unimix).sample_st()
        return h, z

    @torch.no_grad()
    def imagine_step(self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor,
                     e: torch.Tensor) -> dict:
        """Prior step (NO observation). Advances h with action a, samples the
        next latent from the prior, and predicts reward + continue."""
        h_next = self.seq(h, self._flat(z), a, e)
        prior_logits = self.prior(h_next)
        z_next = OneHotCategoricalST(prior_logits, self.cfg.unimix).sample_st()
        z_flat = self._flat(z_next)
        reward = self.reward_from_logits(self.reward_head(h_next, z_flat))
        cont = torch.sigmoid(self.cont_head(h_next, z_flat))
        return {"h": h_next, "z_next": z_next, "reward": reward, "cont": cont}

    def reconstruct_with_pos(self, recon_raw: torch.Tensor,
                             position_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decoder raw output → actual dynamic-state values, overwriting the 10
        entity-coordinate dims with the two-hot PositionHead's expected cell when
        in twohot mode (`position_logits` given). Regress mode: pass None."""
        state = Decoder.reconstruct(recon_raw)
        if position_logits is not None:
            coords = two_hot_decode(F.softmax(position_logits, dim=-1),
                                    self.position_head.bins)        # (..., 10)
            state = state.clone()
            state[..., POS_DIMS] = coords.to(state.dtype)
        return state

    def decode_state(self, h: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        """Full reconstructed dynamic state from {h, z_flat} (eval / rollout)."""
        raw = self.decoder(h, z_flat)
        pos = self.position_head(h, z_flat) if self.position_head is not None else None
        return self.reconstruct_with_pos(raw, pos)

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the ~460-d dynamic state from {h, z} (eval/debug)."""
        return self.decode_state(h, self._flat(z))
