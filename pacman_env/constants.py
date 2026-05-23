from enum import IntEnum


class Tile:
    WALL         = '%'
    FOOD         = '.'
    POWER_PELLET = 'o'
    PACMAN_START = 'P'
    GHOST_START  = 'G'
    EMPTY        = ' '


class Action(IntEnum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
    NOOP  = 4


ACTION_DELTAS = {
    Action.UP:    ( 0, -1),
    Action.DOWN:  ( 0,  1),
    Action.LEFT:  (-1,  0),
    Action.RIGHT: ( 1,  0),
    Action.NOOP:  ( 0,  0),
}

MAX_GRID_H = 21
MAX_GRID_W = 21
MAX_GHOSTS = 4
MAX_FOOD_POSITIONS = MAX_GRID_H * MAX_GRID_W   # 441

# 2 + 4*4 + 441 + 441 + 1 = 901
STATE_DIM = 2 + MAX_GHOSTS * 4 + MAX_FOOD_POSITIONS * 2 + 1
