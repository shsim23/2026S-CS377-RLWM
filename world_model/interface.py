"""Typing-only contract for the policy team. No implementation here."""
from __future__ import annotations
from typing import Dict, Protocol, Tuple

import torch


class WorldModelProtocol(Protocol):
    latent_dim: int
    gru_hidden: int

    def encode(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def warmup_h(
        self, prefix_states: torch.Tensor, prefix_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def imagine_step(
        self, z: torch.Tensor, h: torch.Tensor, a: torch.Tensor
    ) -> Dict[str, torch.Tensor]: ...
