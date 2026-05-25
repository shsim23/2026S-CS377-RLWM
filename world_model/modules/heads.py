import torch
import torch.nn as nn


class RewardHead(nn.Module):
    """Predicts (symlog) reward from (z_t, z_next, h_next).

    v6 architecture: receives BOTH the previous and current latent so that the
    head can express delta-style features like food_count_t − food_count_{t+1}.
    Diagnostic showed (z_next, h_next) alone could not extract the food-delta
    needed for reward, even when the encoder was forced to preserve count via
    the FoodCountHead auxiliary loss.
    """
    def __init__(self, input_dim: int = 512, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_t, z_next, h):
        x = torch.cat([z_t, z_next, h], dim=-1)
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


class DynamicStateHead(nn.Module):
    """Auxiliary: reconstructs the *dynamic* parts of the state from z alone.

    Output layout (460 dims, matches StateBuilder layout minus wall mask):
        [0:2]     pacman pos            (MSE)
        [2:18]    ghost slots (4×4)     (MSE)
        [18:459]  food mask (441)       (per-cell BCE, output is logits)
        [459:460] power timer           (MSE)

    v5/v6 used a 1-dim FoodCountHead, which collapsed to mean prediction
    because count varies by only ±0..1 per step. A multi-dim, mostly-binary
    reconstruction target makes mean-collapse a much worse local minimum.
    """
    OUTPUT_DIM = 460
    FOOD_SLICE = slice(18, 459)

    def __init__(self, latent_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),     nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, self.OUTPUT_DIM),
        )

    def forward(self, z):
        return self.net(z)


class FoodEatenHead(nn.Module):
    """Predicts P(food was eaten in transition t -> t+1) from (z_t, z_next).

    v10: direct discrimination head for the discrete reward-relevant event,
    bypassing the dyn_state sum-of-441-sigmoids that the v8.2 diagnostic
    showed loses the binary signal (39% FN, 8% sign-error).
    """
    def __init__(self, latent_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),         nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_t, z_next):
        x = torch.cat([z_t, z_next], dim=-1)
        return self.net(x).squeeze(-1)   # logit
