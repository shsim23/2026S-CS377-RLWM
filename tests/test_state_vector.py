import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_env import PacmanEnv
from pacman_env.constants import STATE_DIM, MAX_GHOSTS


SIMPLE = """\
%%%%%
%P.G%
%...%
%%%%%
"""


def make_env(layout_str=SIMPLE, **kwargs):
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(layout_str)
        name = f.name
    env = PacmanEnv(layout_path=name, **kwargs)
    os.unlink(name)
    return env


def test_obs_shape_and_dtype():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (STATE_DIM,)
    assert obs.dtype == np.float32


def test_obs_range():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.min() >= -1.0 - 1e-6
    assert obs.max() <= 1.0 + 1e-6


def test_ghost_slots_beyond_num_ghosts():
    env = make_env()  # num_ghosts=1
    obs, _ = env.reset(seed=0)
    # Slots 1-3 should be zero (valid=0)
    slot_start = 2  # after pacman_xy
    for i in range(1, MAX_GHOSTS):
        base = slot_start + i * 4
        assert obs[base + 3] == 0.0, f"Slot {i} valid flag should be 0"


def test_wall_mask_matches_layout():
    env = make_env()
    obs, _ = env.reset(seed=0)
    wall_start = 2 + MAX_GHOSTS * 4 + 441
    wall_flat = obs[wall_start: wall_start + 441]
    # Top-left cell is a wall (SIMPLE layout)
    assert wall_flat[0] == 1.0
    # Cell (1,1) is walkable — Pac-Man start
    from pacman_env.constants import MAX_GRID_W
    idx = 1 * MAX_GRID_W + 1  # row 1, col 1
    assert wall_flat[idx] == 0.0
