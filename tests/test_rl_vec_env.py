import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("rsl_rl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_env.constants import STATE_DIM
from pacman_rl.vec_env import RslPacmanVecEnv


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
