import torch
import torch.nn as nn


class RewardHead(nn.Module):
    def __init__(self, input_dim: int = 384, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        return self.net(x).squeeze(-1)


class DoneHead(nn.Module):
    def __init__(self, input_dim: int = 384, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        return torch.sigmoid(self.net(x).squeeze(-1))
