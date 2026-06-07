import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_env.constants import MAX_GRID_H, MAX_GRID_W, STATE_DIM
from world_model.dreamer.rssm import FOOD_SLICE, GHOST_SLICE, PAC_SLICE, POWER_SLICE, WALL_SLICE
from pacman_rl.wm_rewards import DecodedStateRewardComputer


def _norm(x: int, y: int) -> tuple[float, float]:
    return 2.0 * x / (MAX_GRID_W - 1) - 1.0, 2.0 * y / (MAX_GRID_H - 1) - 1.0


def _state(pac=(1, 1), ghost=None, food=None, power=0.0):
    s = torch.zeros(STATE_DIM, dtype=torch.float32)
    s[PAC_SLICE] = torch.tensor(_norm(*pac))
    walls = torch.zeros(MAX_GRID_H, MAX_GRID_W, dtype=torch.float32)
    walls[0, :] = 1.0
    walls[-1, :] = 1.0
    walls[:, 0] = 1.0
    walls[:, -1] = 1.0
    s[WALL_SLICE] = walls.flatten()
    if ghost is not None:
        ghosts = torch.zeros(4, 4, dtype=torch.float32)
        ghosts[0, :2] = torch.tensor(_norm(*ghost))
        ghosts[0, 2] = 1.0
        ghosts[0, 3] = 1.0
        s[GHOST_SLICE] = ghosts.flatten()
    if food:
        food_mask = torch.zeros(MAX_GRID_H, MAX_GRID_W, dtype=torch.float32)
        for x, y in food:
            food_mask[y, x] = 1.0
        s[FOOD_SLICE] = food_mask.flatten()
    s[POWER_SLICE] = power
    return s


def _computer(reward=None):
    return DecodedStateRewardComputer({
        "reward": reward or {
            "pellet": 10.0,
            "power_pellet": 10.0,
            "ghost_eaten": 20.0,
            "death": -100.0,
            "win": 200.0,
            "sparse_remaining_pellet_penalty": 0.0,
            "dense_remaining_pellet_ratio_penalty": 0.0,
        },
        "power_pellet": {"enabled": False},
    })


def _compute(prev, pred, ended=False, total=10, reward=None):
    return _computer(reward).compute(
        prev.unsqueeze(0),
        pred.unsqueeze(0),
        episode_ended=torch.tensor([ended]),
        total_pellets=torch.tensor([total], dtype=torch.float32),
    )


def test_pellet_removed_at_predicted_pacman_position_gets_pellet_reward():
    prev = _state(pac=(1, 1), food=[(2, 1), (5, 5)])
    pred = _state(pac=(2, 1), food=[(5, 5)])
    out = _compute(prev, pred)
    assert out.ate_pellet.item() == 1.0
    assert out.reward.item() == pytest.approx(10.0)


def test_food_removed_elsewhere_gets_no_pellet_reward():
    prev = _state(pac=(1, 1), food=[(5, 5), (6, 5)])
    pred = _state(pac=(2, 1), food=[(6, 5)])
    out = _compute(prev, pred)
    assert out.ate_pellet.item() == 0.0
    assert out.reward.item() == pytest.approx(0.0)


def test_all_food_gone_adds_win_reward():
    prev = _state(pac=(1, 1), food=[(2, 1)])
    pred = _state(pac=(2, 1), food=[])
    out = _compute(prev, pred)
    assert out.win.item() == 1.0
    assert out.reward.item() == pytest.approx(210.0)


def test_predicted_pacman_ghost_overlap_without_power_adds_death_reward():
    prev = _state(pac=(1, 1), ghost=(3, 1), food=[(5, 5)])
    pred = _state(pac=(2, 1), ghost=(2, 1), food=[(5, 5)])
    out = _compute(prev, pred)
    assert out.death.item() == 1.0
    assert out.reward.item() == pytest.approx(-100.0)


def test_dense_and_sparse_remaining_pellet_penalties_match_reward_computer():
    reward = {
        "pellet": 10.0,
        "power_pellet": 10.0,
        "ghost_eaten": 20.0,
        "death": -100.0,
        "win": 200.0,
        "sparse_remaining_pellet_penalty": -3.0,
        "dense_remaining_pellet_ratio_penalty": -2.0,
    }
    prev = _state(pac=(1, 1), food=[(4, 4), (5, 4), (6, 4), (7, 4)])
    pred = _state(pac=(1, 1), food=[(6, 4), (7, 4)])
    out = _compute(prev, pred, ended=True, total=4, reward=reward)
    assert out.remaining_pellets.item() == 2.0
    assert out.reward.item() == pytest.approx(-7.0)
