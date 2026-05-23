import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model import EnsembleWorldModel
from world_model.constants import STATE_DIM, LATENT_DIM, GRU_HIDDEN


def test_sigma_nonzero_on_random_init():
    """Freshly initialized ensemble should disagree — σ > 0."""
    ens = EnsembleWorldModel(num_members=5)
    z = torch.randn(8, LATENT_DIM)
    h = torch.zeros(8, GRU_HIDDEN)
    a = torch.randint(0, 5, (8,))
    out = ens.imagine_step(z, h, a)
    assert (out["sigma"] > 0).all(), "sigma should be strictly positive on random init"


def test_sigma_scalar_per_element():
    ens = EnsembleWorldModel(num_members=5)
    z = torch.randn(8, LATENT_DIM)
    h = torch.zeros(8, GRU_HIDDEN)
    a = torch.randint(0, 5, (8,))
    out = ens.imagine_step(z, h, a)
    assert out["sigma"].shape == (8,)
