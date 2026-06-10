import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_rl.wm_uncertainty import (
    RunningMean,
    component_weighted_decoded_state_variance,
    confidence_from_uncertainty,
    decoded_state_variance,
    self_ensemble_stats,
)


def test_single_self_ensemble_inference_has_zero_uncertainty():
    samples = torch.ones(1, 2, 4)

    uncertainty = decoded_state_variance(samples)
    uncertainty_norm = RunningMean(device="cpu").normalize(uncertainty)
    stats = self_ensemble_stats(uncertainty, uncertainty_norm, alpha=0.5, confidence_weight_scale=2.0, threshold=2.0)

    assert uncertainty.tolist() == pytest.approx([0.0, 0.0])
    assert stats.confidence.tolist() == pytest.approx([2.0 * torch.sigmoid(torch.tensor(0.5)).item()] * 2)
    assert not stats.truncate.any()


def test_multiple_self_ensemble_samples_measure_decoded_state_variance():
    samples = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[2.0, 0.0], [1.0, 3.0]],
            [[4.0, 0.0], [1.0, 5.0]],
        ],
        dtype=torch.float32,
    )

    uncertainty = decoded_state_variance(samples)

    assert uncertainty[0] > 0.0
    assert uncertainty[1] > 0.0
    assert uncertainty.tolist() == pytest.approx([4.0 / 3.0, 4.0 / 3.0])


def test_component_weights_change_decoded_state_variance_aggregation():
    samples = torch.zeros(2, 1, 901)
    samples[1, 0, 0] = 2.0      # Pac-Man position variance: 1.0 over x, 0.0 over y => 0.5
    samples[1, 0, 18] = 4.0     # Food mask variance: 4.0 over one dim, averaged over 441 dims

    pacman_only = component_weighted_decoded_state_variance(
        samples,
        {"pacman_position": 1.0, "ghost_positions": 0.0, "food_mask": 0.0, "power_timer": 0.0},
    )
    food_only = component_weighted_decoded_state_variance(
        samples,
        {"pacman_position": 0.0, "ghost_positions": 0.0, "food_mask": 1.0, "power_timer": 0.0},
    )
    equal_weighted = component_weighted_decoded_state_variance(
        samples,
        {"pacman_position": 1.0, "ghost_positions": 0.0, "food_mask": 1.0, "power_timer": 0.0},
    )

    assert pacman_only.item() == pytest.approx(0.5)
    assert food_only.item() == pytest.approx(4.0 / 441.0)
    assert equal_weighted.item() == pytest.approx((0.5 + 4.0 / 441.0) / 2.0)


def test_zero_component_weights_produce_zero_uncertainty():
    samples = torch.randn(3, 2, 901)

    uncertainty = component_weighted_decoded_state_variance(
        samples,
        {"pacman_position": 0.0, "ghost_positions": 0.0, "food_mask": 0.0, "power_timer": 0.0},
    )

    assert uncertainty.tolist() == pytest.approx([0.0, 0.0])


def test_sigmoid_confidence_weighting_midpoint_and_monotonicity():
    uncertainty_norm = torch.tensor([0.0, 1.0, 3.0])

    confidence = confidence_from_uncertainty(uncertainty_norm, alpha=0.5, scale=2.0)

    assert confidence[1].item() == pytest.approx(1.0)
    assert confidence[0].item() > 1.0
    assert confidence[2].item() < 1.0
    assert confidence[0] > confidence[1] > confidence[2]
    assert torch.all(confidence > 0.0)
    assert torch.all(confidence < 2.0)


def test_self_ensemble_threshold_is_the_only_truncation_rule():
    uncertainty = torch.tensor([0.1, 5.0])
    uncertainty_norm = torch.tensor([0.5, 2.5])

    stats = self_ensemble_stats(uncertainty, uncertainty_norm, alpha=0.5, confidence_weight_scale=2.0, threshold=2.0)

    assert stats.truncate.tolist() == [False, True]
