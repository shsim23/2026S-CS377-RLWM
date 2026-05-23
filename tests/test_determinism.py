import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_env import PacmanEnv, Action


SIMPLE = """\
%%%%%
%P.G%
%...%
%%%%%
"""

ACTIONS = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.NOOP, Action.RIGHT,
           Action.DOWN, Action.NOOP, Action.LEFT, Action.UP, Action.RIGHT]


def make_env(layout_str=SIMPLE, **kwargs):
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(layout_str)
        name = f.name
    env = PacmanEnv(layout_path=name, **kwargs)
    os.unlink(name)
    return env


def rollout(env, seed, actions):
    obs, _ = env.reset(seed=seed)
    traj = [obs.copy()]
    for a in actions:
        obs, _, terminated, truncated, _ = env.step(int(a))
        traj.append(obs.copy())
        if terminated or truncated:
            break
    return traj


def test_same_seed_same_trajectory():
    env1 = make_env()
    env2 = make_env()
    t1 = rollout(env1, seed=7, actions=ACTIONS)
    t2 = rollout(env2, seed=7, actions=ACTIONS)
    assert len(t1) == len(t2)
    for o1, o2 in zip(t1, t2):
        np.testing.assert_array_equal(o1, o2)


def test_different_seeds_different_trajectory():
    # Use a layout where the ghost has multiple legal moves so epsilon-greedy
    # produces different paths under different seeds.
    bigger = """\
%%%%%%%%%
%P......%
%......G%
%.......%
%......G%
%.......%
%%%%%%%%%
"""
    env1 = make_env(bigger)
    env2 = make_env(bigger)
    long_actions = ACTIONS * 3
    t1 = rollout(env1, seed=10, actions=long_actions)
    t2 = rollout(env2, seed=99, actions=long_actions)
    differ = any(not np.array_equal(o1, o2) for o1, o2 in zip(t1, t2))
    assert differ, "Different seeds should produce different ghost behaviour"


def test_reset_reproduces_initial_state():
    env = make_env()
    obs1, _ = env.reset(seed=42)
    env.step(Action.RIGHT)
    env.step(Action.DOWN)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
