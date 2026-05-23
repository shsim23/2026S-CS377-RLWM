import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model import EnsembleWorldModel, SingleWorldModel
from world_model.constants import STATE_DIM, LATENT_DIM, GRU_HIDDEN


B, L = 4, 10


def test_single_forward_sequence_shapes():
    m = SingleWorldModel()
    states  = torch.randn(B, L, STATE_DIM)
    actions = torch.randint(0, 5, (B, L))
    out = m.forward_sequence(states, actions, burnin=2)
    assert out["z_all"].shape   == (B, L, LATENT_DIM)
    assert out["z_preds"].shape == (B, L - 1, LATENT_DIM)
    assert out["r_preds"].shape == (B, L - 1)
    assert out["d_preds"].shape == (B, L - 1)


def test_single_imagine_step_shapes():
    m = SingleWorldModel()
    z = torch.randn(B, LATENT_DIM)
    h = torch.zeros(B, GRU_HIDDEN)
    a = torch.randint(0, 5, (B,))
    out = m.imagine_step(z, h, a)
    assert out["z_next"].shape        == (B, LATENT_DIM)
    assert out["h_next"].shape        == (B, GRU_HIDDEN)
    assert out["reward_symlog"].shape == (B,)
    assert out["done"].shape          == (B,)


def test_ensemble_encode_shapes():
    ens = EnsembleWorldModel(num_members=3)
    s = torch.randn(B, STATE_DIM)
    z, h = ens.encode(s)
    assert z.shape == (B, LATENT_DIM)
    assert h.shape == (B, GRU_HIDDEN)


def test_ensemble_imagine_step_shapes():
    ens = EnsembleWorldModel(num_members=3)
    z = torch.randn(B, LATENT_DIM)
    h = torch.zeros(B, GRU_HIDDEN)
    a = torch.randint(0, 5, (B,))
    out = ens.imagine_step(z, h, a)
    assert out["z_next"].shape == (B, LATENT_DIM)
    assert out["h_next"].shape == (B, GRU_HIDDEN)
    assert out["reward"].shape == (B,)
    assert out["done"].shape   == (B,)
    assert out["sigma"].shape  == (B,)


def test_warmup_h_shapes():
    ens = EnsembleWorldModel(num_members=3)
    P = 5
    prefix_states  = torch.randn(B, P, STATE_DIM)
    prefix_actions = torch.randint(0, 5, (B, P - 1))
    z, h = ens.warmup_h(prefix_states, prefix_actions)
    assert z.shape == (B, LATENT_DIM)
    assert h.shape == (B, GRU_HIDDEN)
