from __future__ import annotations
import torch
import torch.nn as nn

from .modules.encoder import StateEncoder
from .modules.action import ActionEmbedder
from .modules.dynamics import LatentDynamics
from .modules.heads import RewardHead, DoneHead
from .constants import (
    STATE_DIM, ACTION_DIM, LATENT_DIM, GRU_HIDDEN, HIDDEN_DIM, ACTION_EMB_DIM
)


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
        self.reward_head     = RewardHead(latent_dim + gru_hidden, hidden=128)
        self.done_head       = DoneHead(latent_dim + gru_hidden, hidden=128)

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
        z_all = self.encoder(states)   # (B, L, latent_dim)

        h = torch.zeros(B, self.gru_hidden, device=states.device)
        z_preds, r_preds, d_preds = [], [], []

        for t in range(L - 1):
            a_emb = self.action_embedder(actions[:, t])
            z_next, h = self.dynamics(z_all[:, t], a_emb, h)
            r = self.reward_head(z_next, h)
            d = self.done_head(z_next, h)
            z_preds.append(z_next)
            r_preds.append(r)
            d_preds.append(d)

        return {
            "z_all":   z_all,
            "z_preds": torch.stack(z_preds, dim=1),
            "r_preds": torch.stack(r_preds, dim=1),
            "d_preds": torch.stack(d_preds, dim=1),
            "burnin":  burnin,
        }

    def encode(self, s: torch.Tensor):
        z = self.encoder(s)
        h = torch.zeros(s.shape[0], self.gru_hidden, device=s.device)
        return z, h

    def imagine_step(self, z: torch.Tensor, h: torch.Tensor, a: torch.Tensor) -> dict:
        a_emb = self.action_embedder(a)
        z_next, h_next = self.dynamics(z, a_emb, h)
        r_symlog = self.reward_head(z_next, h_next)
        d_prob   = self.done_head(z_next, h_next)
        return {"z_next": z_next, "h_next": h_next, "reward_symlog": r_symlog, "done": d_prob}
