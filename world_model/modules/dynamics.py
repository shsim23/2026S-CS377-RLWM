import torch
import torch.nn as nn


class LatentDynamics(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        action_emb_dim: int = 32,
        gru_hidden: int = 256,
        hidden: int = 256,
    ):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_emb_dim, gru_hidden)
        self.predictor = nn.Sequential(
            nn.Linear(gru_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_t, a_emb_t, h_t):
        gru_input = torch.cat([z_t, a_emb_t], dim=-1)
        h_next = self.gru(gru_input, h_t)
        z_next = self.predictor(h_next)
        return z_next, h_next
