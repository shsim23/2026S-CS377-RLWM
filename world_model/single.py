from __future__ import annotations
import torch
import torch.nn as nn

from .modules.encoder import StateEncoder
from .modules.action import ActionEmbedder
from .modules.dynamics import LatentDynamics
from .modules.heads import DoneHead, DynamicStateHead, FoodEatenHead
from .utils import symlog
from .constants import (
    STATE_DIM, ACTION_DIM, LATENT_DIM, GRU_HIDDEN, HIDDEN_DIM, ACTION_EMB_DIM
)

# Reward function constants (match configs/env/default.yaml).
# v8.1: reward is computed deterministically from the food-count delta predicted
# by DynamicStateHead. Removes the learned reward_head, which was stuck in a
# mean-predictor plateau (~0.22) across v0–v7.
_FOOD_SLICE = slice(18, 459)
_STEP_PENALTY = -0.01
_PELLET_VALUE = 1.0


def _food_eaten_reward_symlog(food_eaten_prob: torch.Tensor) -> torch.Tensor:
    """v10: reward derived from FoodEatenHead's binary classifier output.
    food_eaten_prob ∈ [0, 1] is the head's sigmoid probability."""
    r_raw = _STEP_PENALTY + _PELLET_VALUE * food_eaten_prob
    return symlog(r_raw)


class SingleWorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        latent_dim: int = LATENT_DIM,
        gru_hidden: int = GRU_HIDDEN,
        hidden_dim: int = HIDDEN_DIM,
        action_emb_dim: int = ACTION_EMB_DIM,
    ):
        super().__init__()
        self.encoder         = StateEncoder(state_dim, latent_dim, hidden_dim)
        self.action_embedder = ActionEmbedder(action_dim, action_emb_dim)
        self.dynamics        = LatentDynamics(latent_dim, action_emb_dim, gru_hidden, hidden_dim)
        self.done_head       = DoneHead(latent_dim + gru_hidden, hidden=128)
        # Reconstructs the dynamic (non-wall) parts of the state from latent z.
        self.dynamic_state_head = DynamicStateHead(latent_dim, hidden=256)
        # v10: dedicated binary classifier for the food-eaten event.
        # Source of the predicted reward.
        self.food_eaten_head = FoodEatenHead(latent_dim, hidden=128)

        self.latent_dim = latent_dim
        self.gru_hidden = gru_hidden

    def forward_sequence(self, states: torch.Tensor, actions: torch.Tensor, burnin: int = 5) -> dict:
        """
        Args:
            states:  (B, L, state_dim)
            actions: (B, L) int64
            burnin:  steps excluded from loss
        Returns dict: z_all, z_preds, r_preds, d_preds, burnin
        """
        B, L, _ = states.shape
        z_all = self.encoder(states)                              # (B, L, latent_dim)
        dyn_state_preds = self.dynamic_state_head(z_all)          # (B, L, 460)

        h = torch.zeros(B, self.gru_hidden, device=states.device)
        z_preds, d_preds = [], []

        for t in range(L - 1):
            a_emb = self.action_embedder(actions[:, t])
            z_next, h = self.dynamics(z_all[:, t], a_emb, h)
            d = self.done_head(z_next, h)
            z_preds.append(z_next)
            d_preds.append(d)

        z_preds = torch.stack(z_preds, dim=1)                     # (B, L-1, latent_dim)

        # dyn_state aux on both paths (kept from v8.2 for state reconstruction)
        dyn_state_z_preds = self.dynamic_state_head(z_preds)      # (B, L-1, 460)

        # v10: reward derived from FoodEatenHead instead of dyn_state sum.
        # Dual-path training:
        #   enc-enc: (z_all[t], z_all[t+1])     — clean teaching signal
        #   enc-dyn: (z_all[t], z_preds[t])     — matches inference
        food_eaten_logit_enc = self.food_eaten_head(z_all[:, :-1], z_all[:, 1:])
        food_eaten_logit_dyn = self.food_eaten_head(z_all[:, :-1], z_preds)
        # Reward uses the enc-dyn path (z_preds), matching K-step rollout.
        r_preds = _food_eaten_reward_symlog(torch.sigmoid(food_eaten_logit_dyn))

        return {
            "z_all":                  z_all,
            "z_preds":                z_preds,
            "r_preds":                r_preds,
            "d_preds":                torch.stack(d_preds, dim=1),
            "dyn_state_preds":        dyn_state_preds,
            "dyn_state_z_preds":      dyn_state_z_preds,
            "food_eaten_logit_enc":   food_eaten_logit_enc,   # (B, L-1)
            "food_eaten_logit_dyn":   food_eaten_logit_dyn,   # (B, L-1)
            "burnin":                 burnin,
        }

    def encode(self, s: torch.Tensor):
        z = self.encoder(s)
        h = torch.zeros(s.shape[0], self.gru_hidden, device=s.device)
        return z, h

    def imagine_step(self, z: torch.Tensor, h: torch.Tensor, a: torch.Tensor) -> dict:
        a_emb = self.action_embedder(a)
        z_next, h_next = self.dynamics(z, a_emb, h)
        food_eaten_prob = torch.sigmoid(self.food_eaten_head(z, z_next))
        r_symlog = _food_eaten_reward_symlog(food_eaten_prob)
        d_prob   = self.done_head(z_next, h_next)
        return {"z_next": z_next, "h_next": h_next, "reward_symlog": r_symlog, "done": d_prob}
