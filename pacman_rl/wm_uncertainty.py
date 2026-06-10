from __future__ import annotations

from dataclasses import dataclass

import torch

from world_model.dreamer.rssm import FOOD_SLICE, GHOST_SLICE, PAC_SLICE, POWER_SLICE


@dataclass
class SelfEnsembleStats:
    uncertainty: torch.Tensor
    uncertainty_norm: torch.Tensor
    confidence: torch.Tensor
    truncate: torch.Tensor


class RunningMean:
    def __init__(self, momentum: float = 0.99, initial: float = 1.0, device: torch.device | str = "cpu") -> None:
        self.momentum = float(momentum)
        self.value = torch.tensor(float(initial), dtype=torch.float32, device=device)
        self.initialized = False

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.detach().mean().clamp_min(1e-8)
        if not self.initialized:
            self.value = mean
            self.initialized = True
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * mean
        return x / (self.value + 1e-8)


def decoded_state_variance(decoded_samples: torch.Tensor) -> torch.Tensor:
    """Return one uncertainty scalar per env from K decoded state samples."""
    if decoded_samples.ndim != 3:
        raise ValueError("decoded_samples must have shape (K, B, D)")
    if decoded_samples.shape[0] <= 1:
        return torch.zeros(decoded_samples.shape[1], dtype=decoded_samples.dtype, device=decoded_samples.device)
    return decoded_samples.var(dim=0, unbiased=False).mean(dim=-1)


def component_weighted_decoded_state_variance(
    decoded_samples: torch.Tensor,
    component_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Aggregate decoded-state sample variance with per-component weights.

    The input is the full 901-d state. Wall-mask dimensions are intentionally
    ignored because they are static layout context, not a stochastic prediction.
    """
    if decoded_samples.ndim != 3:
        raise ValueError("decoded_samples must have shape (K, B, D)")
    if decoded_samples.shape[-1] <= POWER_SLICE.start:
        raise ValueError("decoded_samples must contain full 901-d decoded states")
    if decoded_samples.shape[0] <= 1:
        return torch.zeros(decoded_samples.shape[1], dtype=decoded_samples.dtype, device=decoded_samples.device)

    weights = component_weights or {}
    component_terms = [
        ("pacman_position", decoded_samples[..., PAC_SLICE]),
        ("ghost_positions", decoded_samples[..., GHOST_SLICE].reshape(decoded_samples.shape[0], decoded_samples.shape[1], -1, 4)[..., :2]),
        ("food_mask", decoded_samples[..., FOOD_SLICE]),
        ("power_timer", decoded_samples[..., POWER_SLICE]),
    ]

    total = torch.zeros(decoded_samples.shape[1], dtype=decoded_samples.dtype, device=decoded_samples.device)
    total_weight = 0.0
    for name, component in component_terms:
        weight = float(weights.get(name, 1.0))
        if weight < 0.0:
            raise ValueError(f"component weight must be non-negative: {name}")
        if weight == 0.0:
            continue
        component_var = component.var(dim=0, unbiased=False).reshape(decoded_samples.shape[1], -1).mean(dim=-1)
        total = total + weight * component_var
        total_weight += weight

    if total_weight == 0.0:
        return torch.zeros(decoded_samples.shape[1], dtype=decoded_samples.dtype, device=decoded_samples.device)
    return total / total_weight


def confidence_from_uncertainty(
    uncertainty_norm: torch.Tensor,
    alpha: float,
    scale: float,
) -> torch.Tensor:
    return (float(scale) * torch.sigmoid(-float(alpha) * (uncertainty_norm - 1.0))).detach()


def self_ensemble_stats(
    uncertainty: torch.Tensor,
    uncertainty_norm: torch.Tensor,
    alpha: float,
    confidence_weight_scale: float,
    threshold: float,
) -> SelfEnsembleStats:
    confidence = confidence_from_uncertainty(uncertainty_norm, alpha, confidence_weight_scale)
    truncate = uncertainty_norm > float(threshold)
    return SelfEnsembleStats(uncertainty, uncertainty_norm, confidence, truncate)
