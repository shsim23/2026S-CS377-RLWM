import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("rsl_rl")
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_rl.wm_ppo import ConfidenceRolloutStorage


def test_confidence_rollout_storage_yields_transition_confidence():
    obs = TensorDict({"policy": torch.zeros(2, 3)}, batch_size=[2])
    storage = ConfidenceRolloutStorage("rl", 2, 2, obs, [1], device="cpu")
    for step in range(2):
        t = RolloutStorage.Transition()
        t.observations = TensorDict({"policy": torch.full((2, 3), float(step))}, batch_size=[2])
        t.actions = torch.zeros(2, 1)
        t.rewards = torch.zeros(2)
        t.dones = torch.zeros(2, dtype=torch.bool)
        t.values = torch.zeros(2, 1)
        t.actions_log_prob = torch.zeros(2, 1)
        t.distribution_params = (torch.zeros(2, 1),)
        t.confidence = torch.tensor([0.25, 0.75]) + step
        storage.add_transition(t)
    storage.returns.zero_()
    storage.advantages.zero_()
    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    assert hasattr(batch, "confidence")
    assert sorted(batch.confidence.flatten().tolist()) == pytest.approx([0.25, 0.75, 1.25, 1.75])
