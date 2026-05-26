"""Pac-Man maze generator (Phase 1).

Public API:
    generate_maze(...) -> dict   # see generator.py for full signature
    ascii_render(maze) -> str    # ASCII visualization
"""
from .generator import generate_maze
from .visualizer import ascii_render, render_image, summary_line

__all__ = ["generate_maze", "ascii_render", "render_image", "summary_line"]
