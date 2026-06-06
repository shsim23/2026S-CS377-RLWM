import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_env.constants import Action, MAX_GRID_H, MAX_GRID_W, STATE_DIM
from world_model.dreamer.rssm import FOOD_SLICE, GHOST_SLICE, PAC_SLICE, WALL_SLICE
from pacman_rl.wm_reliability import RuleBasedTransitionScorer


def _norm(x: int, y: int) -> tuple[float, float]:
    return 2.0 * x / (MAX_GRID_W - 1) - 1.0, 2.0 * y / (MAX_GRID_H - 1) - 1.0


def _state(pac=(1, 1), ghost=None, food=None):
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
    return s


def _score(curr, pred, action, done_prob=None):
    scorer = RuleBasedTransitionScorer()
    current = curr.unsqueeze(0)
    predicted = pred.unsqueeze(0)
    actions = torch.tensor([int(action)])
    if done_prob is not None:
        done_prob = torch.tensor([done_prob], dtype=torch.float32)
    return scorer.score(current, predicted, actions, done_prob=done_prob).item()


def test_legal_pacman_action_scores_zero():
    curr = _state(pac=(1, 1))
    pred = _state(pac=(2, 1))
    assert _score(curr, pred, Action.RIGHT) == pytest.approx(0.0)


def test_wall_hit_expects_pacman_to_stay_put():
    curr = _state(pac=(1, 1))
    pred = _state(pac=(1, 1))
    assert _score(curr, pred, Action.LEFT) == pytest.approx(0.0)


def test_pacman_speed_violation_increases_score():
    curr = _state(pac=(1, 1))
    pred = _state(pac=(3, 1))
    assert _score(curr, pred, Action.RIGHT) > 0.0


def test_legal_stochastic_ghost_move_is_not_penalized():
    curr = _state(pac=(1, 1), ghost=(5, 5))
    pred = _state(pac=(1, 1), ghost=(5, 6))
    assert _score(curr, pred, Action.NOOP) == pytest.approx(0.0)


def test_food_cannot_appear():
    curr = _state(pac=(1, 1), food=[])
    pred = _state(pac=(1, 1), food=[(4, 4)])
    assert _score(curr, pred, Action.NOOP) > 0.0


def test_removed_food_must_be_at_predicted_pacman_position():
    curr = _state(pac=(1, 1), food=[(5, 5)])
    pred = _state(pac=(2, 1), food=[])
    assert _score(curr, pred, Action.RIGHT) > 0.0


def test_collision_requires_high_done_probability_when_available():
    curr = _state(pac=(1, 1), ghost=(2, 1))
    pred = _state(pac=(2, 1), ghost=(2, 1))
    low_done = _score(curr, pred, Action.RIGHT, done_prob=0.0)
    high_done = _score(curr, pred, Action.RIGHT, done_prob=1.0)
    assert low_done > high_done
