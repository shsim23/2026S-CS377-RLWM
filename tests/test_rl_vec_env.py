import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("rsl_rl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_env.constants import STATE_DIM
from pacman_rl.vec_env import RslPacmanVecEnv
from pacman_rl.video import _make_render_env


def test_vec_env_shapes():
    cfg = {
        "layout_file": str(Path(__file__).resolve().parents[1] / "layouts/train/corridor.txt"),
        "num_envs": 2,
        "max_steps": 5,
        "randomize_spawn": False,
    }
    env = RslPacmanVecEnv(cfg, device="cpu")
    try:
        obs = env.get_observations()
        assert obs["policy"].shape == (2, STATE_DIM)
        actions = torch.zeros(2, 1, dtype=torch.long)
        next_obs, rewards, dones, extras = env.step(actions)
        assert next_obs["policy"].shape == (2, STATE_DIM)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert extras["time_outs"].shape == (2,)
    finally:
        env.close()


def test_vec_env_threads_ghost_speed_ratio():
    cfg = {
        "layout_file": str(Path(__file__).resolve().parents[1] / "layouts/train/corridor.txt"),
        "num_envs": 1,
        "randomize_spawn": False,
        "ghost": {"speed_ratio": 0.5},
    }
    env = RslPacmanVecEnv(cfg, device="cpu")
    try:
        assert env.envs[0].ghost_speed_ratio == 0.5
    finally:
        env.close()


def test_video_env_threads_ghost_speed_ratio():
    cfg = {
        "layout_file": str(Path(__file__).resolve().parents[1] / "layouts/train/corridor.txt"),
        "randomize_spawn": False,
        "ghost": {"speed_ratio": 0.25},
    }
    env = _make_render_env(cfg)
    try:
        assert env.ghost_speed_ratio == 0.25
    finally:
        env.close()
