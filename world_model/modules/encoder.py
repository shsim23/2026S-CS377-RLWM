"""Encoders mapping state vectors → latent z.

Two encoders are provided:

* `StateEncoder` (legacy MLP) — kept so existing v10c-and-earlier checkpoints
  remain loadable.

* `CNNStateEncoder` (v11+, default) — recognises the 21×21 spatial structure
  of the Pac-Man state. The flat MLP encoder had to learn local update rules
  ("food[5,7] disappears iff pacman moves to (5,7)") from a 901-D flat vector,
  which the conv kernels capture for free. Design follows the conv-stack
  pattern used by DreamerV2 / PlaNet / MuZero — small 3×3 kernels, channel
  doubling per stride-2 downsample, GroupNorm + SiLU.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from pacman_env.constants import MAX_GRID_H, MAX_GRID_W


# --------------------------------------------------------------------------- #
class StateEncoder(nn.Module):
    """Legacy MLP encoder. Kept for backwards compatibility with v0–v10c
    checkpoints; new training defaults to CNNStateEncoder."""
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


# --------------------------------------------------------------------------- #
class _LayerNormChannel(nn.Module):
    """LayerNorm over the channel dim of a (B, C, H, W) tensor.

    DreamerV3-style conv normalization. GroupNorm on tiny spatial maps was
    noisy in v11 (3×3 final feature) — channel-only LayerNorm keeps the
    spatial dim's variance intact while still stabilising per-channel scale.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H, W, C) → LN(C) → (B, C, H, W)
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class CNNStateEncoder(nn.Module):
    """CNN encoder that reshapes the flat state vector into a 5-channel
    21×21 spatial grid before convolution.

    Channels
    --------
        ch0: walls       (binary, episode-constant)        ← state[459:900]
        ch1: food mask   (binary, dynamic)                 ← state[18:459]
        ch2: pacman      (one-hot 21×21)                   ← state[0:2]
        ch3: ghosts      (multi-hot 21×21, alive only)     ← state[2:18]
        ch4: power timer (broadcast scalar)                ← state[900]

    Conv stack — v12 revision (A+D):
        * Only ONE spatial downsample (21 → 11). v11 (21→11→6→3) lost the
          1-pixel-scale food-eaten signal in the receptive field averaging.
        * Channel-wise LayerNorm in place of GroupNorm — stable on the
          small 11×11 feature maps.

        Conv 3×3 s1 p1 (5→32)     LN-C  SiLU
        Conv 3×3 s2 p1 (32→64)    LN-C  SiLU   ← only downsample, 21→11
        Conv 3×3 s1 p1 (64→64)    LN-C  SiLU
        Conv 3×3 s1 p1 (64→64)    LN-C  SiLU
    then flatten (64·11·11 = 7744) → Linear(hidden) → LayerNorm SiLU → Linear(latent_dim).
    """

    GRID_H = MAX_GRID_H   # 21
    GRID_W = MAX_GRID_W   # 21

    def __init__(self, state_dim: int = 901, latent_dim: int = 128, hidden: int = 256):
        super().__init__()
        assert state_dim == 901, "CNNStateEncoder assumes the StateBuilder 901-D layout"
        self.cnn = nn.Sequential(
            nn.Conv2d(5,  32, kernel_size=3, stride=1, padding=1), _LayerNormChannel(32), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), _LayerNormChannel(64), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), _LayerNormChannel(64), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), _LayerNormChannel(64), nn.SiLU(),
        )
        # spatial: 21 → 11 (only one stride-2)
        self.fc = nn.Sequential(
            nn.Linear(64 * 11 * 11, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    # ------------------------------------------------------------------ #
    def _state_to_grid(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, 901) → grid: (B, 5, 21, 21).

        Coordinate denormalization mirrors `pacman_env.state._normalize`:
            x_grid = round((x_norm + 1) · (W-1) / 2)
        """
        B = state.shape[0]
        H, W = self.GRID_H, self.GRID_W
        dev = state.device
        idx = torch.arange(B, device=dev)

        # binary masks reshape directly
        walls = state[:, 459:900].reshape(B, H, W)
        food  = state[:, 18:459].reshape(B, H, W)

        # pacman one-hot
        px = ((state[:, 0] + 1.0) * (W - 1) / 2.0).round().long().clamp(0, W - 1)
        py = ((state[:, 1] + 1.0) * (H - 1) / 2.0).round().long().clamp(0, H - 1)
        pac = torch.zeros(B, H, W, device=dev, dtype=state.dtype)
        pac[idx, py, px] = 1.0

        # ghosts multi-hot (state[2:18] = 4 ghosts × [x, y, alive, valid])
        ghost = torch.zeros(B, H, W, device=dev, dtype=state.dtype)
        for i in range(4):
            base = 2 + i * 4
            gx_n  = state[:, base + 0]
            gy_n  = state[:, base + 1]
            alive = state[:, base + 2]
            valid = state[:, base + 3]
            mask = (alive > 0.5) & (valid > 0.5)
            if mask.any():
                gx = ((gx_n + 1.0) * (W - 1) / 2.0).round().long().clamp(0, W - 1)
                gy = ((gy_n + 1.0) * (H - 1) / 2.0).round().long().clamp(0, H - 1)
                ghost[idx[mask], gy[mask], gx[mask]] = 1.0

        # power timer broadcast (state[900])
        power_grid = state[:, 900:901].unsqueeze(-1).expand(B, H, W)

        return torch.stack([walls, food, pac, ghost, power_grid], dim=1)

    # ------------------------------------------------------------------ #
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Handles (B, D) or (B, L, D) inputs identically to StateEncoder."""
        if s.dim() == 3:
            B, L, D = s.shape
            flat = s.reshape(B * L, D)
            grid = self._state_to_grid(flat)
            h = self.cnn(grid).flatten(1)
            z = self.fc(h)
            return z.reshape(B, L, -1)
        grid = self._state_to_grid(s)
        h = self.cnn(grid).flatten(1)
        return self.fc(h)
