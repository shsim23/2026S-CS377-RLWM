from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RewardConfig:
    pellet:       float = 1.0
    power_pellet: float = 5.0
    ghost_eaten:  float = 10.0
    death:        float = -10.0
    win:          float = 50.0
    sparse_remaining_pellet_penalty: float = 0.0
    dense_remaining_pellet_ratio_penalty: float = 0.0


@dataclass
class StepEvent:
    ate_pellet: bool = False
    ate_power:  bool = False
    ate_ghosts: int  = 0
    died:       bool = False
    won:        bool = False
    remaining_pellets: int = 0
    total_pellets: int = 0
    episode_ended: bool = False


class RewardComputer:
    def __init__(self, config: RewardConfig):
        self.cfg = config

    def compute(self, event: StepEvent) -> float:
        r = 0.0
        if event.ate_pellet:
            r += self.cfg.pellet
        if event.ate_power:
            r += self.cfg.power_pellet
        r += event.ate_ghosts * self.cfg.ghost_eaten
        if event.total_pellets > 0:
            remaining_ratio = event.remaining_pellets / event.total_pellets
            r += self.cfg.dense_remaining_pellet_ratio_penalty * remaining_ratio
        if event.episode_ended:
            r += self.cfg.sparse_remaining_pellet_penalty * event.remaining_pellets
        if event.died:
            r += self.cfg.death
        if event.won:
            r += self.cfg.win
        return r
