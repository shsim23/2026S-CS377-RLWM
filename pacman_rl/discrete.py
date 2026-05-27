from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rsl_rl.modules.distribution import Distribution


class CategoricalDistribution(Distribution):
    """Categorical action distribution for rsl_rl's MLPModel.

    The rsl_rl PPO storage still sees one action dimension, while the MLP head
    emits one logit for each discrete Pacman action.
    """

    def __init__(self, output_dim: int, num_categories: int = 5) -> None:
        if output_dim != 1:
            raise ValueError(
                "CategoricalDistribution expects env.num_actions == 1 "
                f"(one integer action id), got {output_dim}."
            )
        super().__init__(output_dim)
        self.num_categories = int(num_categories)
        self._distribution: Categorical | None = None
        self._logits: torch.Tensor | None = None
        Categorical.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        self._logits = mlp_output
        self._distribution = Categorical(logits=mlp_output)

    def sample(self) -> torch.Tensor:
        assert self._distribution is not None
        return self._distribution.sample().unsqueeze(-1)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(mlp_output, dim=-1, keepdim=True)

    def as_deterministic_output_module(self) -> nn.Module:
        return _ArgmaxDeterministicOutput()

    @property
    def input_dim(self) -> int:
        return self.num_categories

    @property
    def mean(self) -> torch.Tensor:
        assert self._distribution is not None
        return self._distribution.probs

    @property
    def std(self) -> torch.Tensor:
        assert self._distribution is not None
        probs = self._distribution.probs
        return torch.sqrt(torch.clamp(probs * (1.0 - probs), min=0.0))

    @property
    def entropy(self) -> torch.Tensor:
        assert self._distribution is not None
        return self._distribution.entropy()

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        assert self._logits is not None
        return (self._logits,)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        assert self._distribution is not None
        actions = outputs.long().squeeze(-1)
        return self._distribution.log_prob(actions)

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        (old_logits,) = old_params
        (new_logits,) = new_params
        old_dist = Categorical(logits=old_logits)
        new_dist = Categorical(logits=new_logits)
        return torch.distributions.kl_divergence(old_dist, new_dist)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        if hasattr(mlp, "init_weights"):
            mlp.init_weights(1.0)


class _ArgmaxDeterministicOutput(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True)


class TorchCategoricalPolicy(nn.Module):
    """Small export wrapper useful if direct JIT/ONNX export is requested."""

    def __init__(self, obs_normalizer: nn.Module, mlp: nn.Module) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(obs_normalizer)
        self.mlp = copy.deepcopy(mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(self.obs_normalizer(x))
        return torch.argmax(logits, dim=-1, keepdim=True)
