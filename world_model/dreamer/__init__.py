"""DreamerV3-style state-based RSSM world model (WORLDMODEL_DREAMERV3_SPEC.md).

A fresh implementation, independent of the legacy v10c / JEPA model in the parent
`world_model` package (kept as an ablation baseline per spec §11). Categorical
latents (32×32, straight-through, 1% unimix), two-hot symlog reward head,
KL/free-bits objective, layout conditioning, LaProp + AGC.
"""
from .world_model import DreamerWorldModel, WorldModelConfig
from .loss import compute_loss
from .optim import LaProp, adaptive_grad_clip
from .replay import SequenceReplay
from .eval import evaluate

__all__ = [
    "DreamerWorldModel",
    "WorldModelConfig",
    "compute_loss",
    "LaProp",
    "adaptive_grad_clip",
    "SequenceReplay",
    "evaluate",
]
