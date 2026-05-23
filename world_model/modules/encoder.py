import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int = 901, latent_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, s):
        if s.dim() == 3:
            B, L, D = s.shape
            return self.net(s.reshape(B * L, D)).reshape(B, L, -1)
        return self.net(s)
