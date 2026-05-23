from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .constants import MAX_GRID_H, MAX_GRID_W, Tile


@dataclass(frozen=True)
class Layout:
    name: str
    height: int
    width: int
    walls: np.ndarray          # (H, W) bool
    initial_food: np.ndarray   # (H, W) bool
    initial_power: np.ndarray  # (H, W) bool
    pacman_start: Tuple[int, int]
    ghost_starts: List[Tuple[int, int]]

    def to_padded_arrays(self) -> Dict[str, np.ndarray]:
        """Pad masks to (MAX_GRID_H, MAX_GRID_W) with walls."""
        def pad(arr: np.ndarray, fill: bool) -> np.ndarray:
            out = np.full((MAX_GRID_H, MAX_GRID_W), fill, dtype=bool)
            out[: self.height, : self.width] = arr
            return out

        return {
            "walls":         pad(self.walls, True),
            "initial_food":  pad(self.initial_food, False),
            "initial_power": pad(self.initial_power, False),
        }


class LayoutParser:
    @staticmethod
    def from_file(path: str) -> Layout:
        with open(path, "r") as f:
            text = f.read()
        name = path.split("/")[-1].replace(".txt", "")
        return LayoutParser.from_string(text, name)

    @staticmethod
    def from_string(text: str, name: str = "anonymous") -> Layout:
        lines = text.splitlines()
        # Remove empty trailing lines
        while lines and not lines[-1].strip():
            lines.pop()

        height = len(lines)
        width = max(len(row) for row in lines)

        walls   = np.zeros((height, width), dtype=bool)
        food    = np.zeros((height, width), dtype=bool)
        power   = np.zeros((height, width), dtype=bool)
        pacman_start = None
        ghost_starts = []

        for y, row in enumerate(lines):
            for x, ch in enumerate(row):
                if ch == Tile.WALL:
                    walls[y, x] = True
                elif ch == Tile.FOOD:
                    food[y, x] = True
                elif ch == Tile.POWER_PELLET:
                    power[y, x] = True
                elif ch == Tile.PACMAN_START:
                    pacman_start = (x, y)
                elif ch == Tile.GHOST_START:
                    ghost_starts.append((x, y))

        layout = Layout(
            name=name,
            height=height,
            width=width,
            walls=walls,
            initial_food=food,
            initial_power=power,
            pacman_start=pacman_start,
            ghost_starts=ghost_starts,
        )
        LayoutParser.validate(layout)
        return layout

    @staticmethod
    def validate(layout: Layout) -> None:
        if layout.height > MAX_GRID_H or layout.width > MAX_GRID_W:
            raise ValueError(
                f"Layout {layout.name} is {layout.height}x{layout.width}, "
                f"exceeds max {MAX_GRID_H}x{MAX_GRID_W}"
            )
        if layout.pacman_start is None:
            raise ValueError(f"Layout {layout.name} has no Pac-Man start ('P')")
        if not layout.ghost_starts:
            raise ValueError(f"Layout {layout.name} has no ghost start ('G')")

        # Check fully enclosed (border must all be walls)
        H, W = layout.height, layout.width
        if not (
            layout.walls[0, :].all()
            and layout.walls[H - 1, :].all()
            and layout.walls[:, 0].all()
            and layout.walls[:, W - 1].all()
        ):
            raise ValueError(f"Layout {layout.name} is not fully enclosed by walls")
