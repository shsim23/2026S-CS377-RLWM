from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    pellet:       float = 1.0
    power_pellet: float = 5.0
    ghost_eaten:  float = 10.0
    death:        float = -10.0
    win:          float = 50.0
    step_penalty: float = -0.01


@dataclass
class StepEvent:
    ate_pellet: bool = False
    ate_power:  bool = False
    ate_ghosts: int  = 0
    died:       bool = False
    won:        bool = False


class RewardComputer:
    def __init__(self, config: RewardConfig):
        self.cfg = config

    def compute(self, event: StepEvent) -> float:
        r = self.cfg.step_penalty
        if event.ate_pellet:
            r += self.cfg.pellet
        if event.ate_power:
            r += self.cfg.power_pellet
        r += event.ate_ghosts * self.cfg.ghost_eaten
        if event.died:
            r += self.cfg.death
        if event.won:
            r += self.cfg.win
        return r
