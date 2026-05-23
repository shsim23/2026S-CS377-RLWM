import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_env.layout import Layout, LayoutParser
from pacman_env.constants import MAX_GRID_H, MAX_GRID_W

VALID = """\
%%%%%
%P..%
%..G%
%%%%%
"""


def test_parse_valid():
    layout = LayoutParser.from_string(VALID, "test")
    assert layout.pacman_start == (1, 1)
    assert layout.ghost_starts == [(3, 2)]
    assert layout.height == 4
    assert layout.width == 5


def test_food_and_walls():
    layout = LayoutParser.from_string(VALID, "test")
    assert layout.walls[0, 0]
    assert not layout.walls[1, 1]
    assert layout.initial_food[1, 2]
    assert layout.initial_food[1, 3]
    assert not layout.initial_food[1, 1]  # P cell has no food


def test_padding():
    layout = LayoutParser.from_string(VALID, "test")
    padded = layout.to_padded_arrays()
    assert padded["walls"].shape == (MAX_GRID_H, MAX_GRID_W)
    # Padded area should be walls
    assert padded["walls"][4, 0]


def test_reject_no_pacman():
    bad = "%%%%%\n%...%\n%..G%\n%%%%%\n"
    with pytest.raises(ValueError, match="no Pac-Man"):
        LayoutParser.from_string(bad, "bad")


def test_reject_no_ghost():
    bad = "%%%%%\n%P..%\n%...%\n%%%%%\n"
    with pytest.raises(ValueError, match="no ghost"):
        LayoutParser.from_string(bad, "bad")


def test_reject_not_enclosed():
    bad = "XXXXX\n%P..%\n%..G%\n%%%%%\n"
    with pytest.raises(ValueError):
        LayoutParser.from_string(bad, "bad")


def test_reject_too_large():
    # Build a 22x22 layout — exceeds MAX_GRID_H/W
    row = "%" * 22
    inner = "%" + "." * 20 + "%"
    lines = [row] + [inner] * 19 + ["%" + "P" + "." * 10 + "G" + "." * 8 + "%"] + [row]
    text = "\n".join(lines)
    with pytest.raises(ValueError, match="exceeds max"):
        LayoutParser.from_string(text, "toobig")
