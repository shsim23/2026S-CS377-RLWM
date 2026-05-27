import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("rsl_rl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_rl.discrete import CategoricalDistribution


def test_categorical_distribution_shapes_and_values():
    dist = CategoricalDistribution(output_dim=1, num_categories=5)
    logits = torch.randn(7, 5)
    dist.update(logits)

    actions = dist.sample()
    assert actions.shape == (7, 1)
    assert actions.dtype == torch.long
    assert int(actions.min()) >= 0
    assert int(actions.max()) < 5

    log_prob = dist.log_prob(actions)
    entropy = dist.entropy
    kl = dist.kl_divergence(dist.params, dist.params)

    assert log_prob.shape == (7,)
    assert entropy.shape == (7,)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)
