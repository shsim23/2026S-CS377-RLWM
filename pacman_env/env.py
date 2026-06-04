from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from .constants import Action, ACTION_DELTAS, STATE_DIM
from .layout import Layout, LayoutParser
from .state import GameState, StateBuilder
from .ghost import GhostController
from .reward import RewardComputer, RewardConfig, StepEvent
from .renderer import Renderer


class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}

    def __init__(
        self,
        layout_path: str,
        num_ghosts: int = 1,
        ghost_epsilon: float = 0.2,
        ghost_policy: str = "chase_stochastic",
        ghost_speed_ratio: float = 1.0,
        power_pellet_enabled: bool = False,
        frightened_duration: int = 30,
        max_steps: int = 500,
        reward_config: Optional[RewardConfig] = None,
        render_mode: Optional[str] = None,
        randomize_spawn: bool = False,
        min_spawn_dist: int = 2,
    ):
        super().__init__()
        self.layout: Layout = LayoutParser.from_file(layout_path)
        # When randomizing spawn we are no longer bound by the number of 'G'
        # markers in the layout — any walkable cell can host a ghost.
        if randomize_spawn:
            self.num_ghosts = num_ghosts
        else:
            self.num_ghosts = min(num_ghosts, len(self.layout.ghost_starts))
        if ghost_speed_ratio < 0.0:
            raise ValueError("ghost_speed_ratio must be >= 0.0")
        self.ghost_epsilon = ghost_epsilon
        self.ghost_policy = ghost_policy
        self.ghost_speed_ratio = float(ghost_speed_ratio)
        self.power_pellet_enabled = power_pellet_enabled
        self.frightened_duration = frightened_duration
        self.max_steps = max_steps
        self.reward_cfg = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.randomize_spawn = randomize_spawn
        self.min_spawn_dist = min_spawn_dist
        self._walkable_cells: List[Tuple[int, int]] = [
            (x, y)
            for y in range(self.layout.height)
            for x in range(self.layout.width)
            if not self.layout.walls[y, x]
        ]

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)

        self._reward_computer = RewardComputer(self.reward_cfg)
        self._state_builder = StateBuilder(self.layout, self.num_ghosts)
        self._renderer = Renderer(self.layout, render_mode)

        # Runtime state (initialized in reset)
        self.game_state: Optional[GameState] = None
        self.np_random: np.random.Generator = np.random.default_rng()
        self._ghost_controller: Optional[GhostController] = None
        self._ghost_move_credit: float = 0.0
        self._total_pellets: int = 0
        self._cumulative_reward: float = 0.0
        self._last_event: Optional[StepEvent] = None

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        if self.ghost_policy == "chase_stochastic":
            self._ghost_controller = GhostController(self.ghost_epsilon, self.np_random)
        elif self.ghost_policy == "chase":
            self._ghost_controller = GhostController(0.0, self.np_random)
        else:  # random
            self._ghost_controller = GhostController(1.0, self.np_random)

        self.game_state = self._init_game_state()
        self._ghost_move_credit = 0.0
        self._total_pellets = int(self.game_state.food_mask.sum())
        self._cumulative_reward = 0.0
        self._last_event = None

        obs = self._state_builder.build(self.game_state)
        return obs, self._build_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.game_state is not None, "Call reset() before step()"

        gs = self.game_state
        prev_gs = gs.copy()

        # 1. Move Pac-Man
        self._move_pacman(action)

        # 2. Intermediate collision check (Pac-Man walked onto a ghost)
        event = StepEvent()
        self._check_collision(event)

        # 3. Pac-Man consumes items
        if not event.died:
            self._consume_items(event)

        # 4. Ghosts move according to their speed ratio
        if not event.died:
            self._move_ghosts_on_schedule(event)

        # 6. Win condition
        if not event.died and gs.food_mask.sum() == 0:
            event.won = True

        # 7. Termination state used by sparse terminal rewards
        next_step_count = gs.step_count + 1
        terminated = event.died or event.won
        truncated = next_step_count >= self.max_steps
        event.remaining_pellets = int(gs.food_mask.sum())
        event.total_pellets = self._total_pellets
        event.episode_ended = terminated or truncated

        # 8. Power timer tick
        if gs.power_mode_timer > 0:
            gs.power_mode_timer -= 1

        # 9. Reward
        reward = self._reward_computer.compute(event)
        self._cumulative_reward += reward

        # 10. Bookkeeping
        gs.step_count = next_step_count
        gs.done = terminated or truncated

        self._last_event = event
        obs = self._state_builder.build(gs)
        info = self._build_info(event)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.game_state is None:
            return None
        return self._renderer.render(self.game_state)

    def close(self):
        self._renderer.close()

    # ------------------------------------------------------------------ #
    def _init_game_state(self) -> GameState:
        food = self.layout.initial_food.copy()
        if not self.power_pellet_enabled:
            # Treat power pellets as regular pellets (eat them but no effect)
            food = food | self.layout.initial_power

        if self.randomize_spawn:
            pacman_pos, ghost_positions = self._sample_random_spawn()
            # Match the original-layout convention that the Pac-Man cell has no
            # pellet, so the very first step doesn't auto-consume a pellet.
            px, py = pacman_pos
            food[py, px] = False
        else:
            pacman_pos = self.layout.pacman_start
            ghost_positions = [self.layout.ghost_starts[i] for i in range(self.num_ghosts)]

        return GameState(
            pacman_pos=pacman_pos,
            ghost_positions=ghost_positions,
            ghost_alive=[True] * self.num_ghosts,
            food_mask=food,
            power_mode_timer=0,
            step_count=0,
            done=False,
        )

    def _sample_random_spawn(self) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
        """Sample Pac-Man + ghost positions uniformly over walkable cells.

        Pac-Man is uniform over all walkable cells. Each ghost is uniform over
        walkable cells not already occupied AND at Manhattan distance >=
        min_spawn_dist from Pac-Man. If that constraint cannot be satisfied
        (very small/cramped layouts), the distance constraint is relaxed but
        cells remain distinct.
        """
        walkable = self._walkable_cells
        if len(walkable) < 1 + self.num_ghosts:
            raise ValueError(
                f"Layout '{self.layout.name}' has only {len(walkable)} walkable cells; "
                f"need at least {1 + self.num_ghosts}."
            )

        idx = int(self.np_random.integers(len(walkable)))
        pacman_pos = walkable[idx]

        used = {pacman_pos}
        ghost_positions: List[Tuple[int, int]] = []
        for _ in range(self.num_ghosts):
            candidates = [
                c for c in walkable
                if c not in used
                and (abs(c[0] - pacman_pos[0]) + abs(c[1] - pacman_pos[1])) >= self.min_spawn_dist
            ]
            if not candidates:  # Fallback: any unused walkable cell
                candidates = [c for c in walkable if c not in used]
            idx = int(self.np_random.integers(len(candidates)))
            ghost_positions.append(candidates[idx])
            used.add(candidates[idx])

        return pacman_pos, ghost_positions

    def _move_pacman(self, action: int) -> None:
        gs = self.game_state
        dx, dy = ACTION_DELTAS[Action(action)]
        nx, ny = gs.pacman_pos[0] + dx, gs.pacman_pos[1] + dy
        if 0 <= ny < self.layout.height and 0 <= nx < self.layout.width:
            if not self.layout.walls[ny, nx]:
                gs.pacman_pos = (nx, ny)

    def _move_ghosts(self, event: StepEvent) -> None:
        gs = self.game_state
        walls = self.layout.walls
        new_positions = list(gs.ghost_positions)
        for i in range(self.num_ghosts):
            if not gs.ghost_alive[i]:
                continue
            new_positions[i] = self._ghost_controller.step(
                gs.ghost_positions[i], gs.pacman_pos, walls
            )
        gs.ghost_positions = new_positions

    def _move_ghosts_on_schedule(self, event: StepEvent) -> None:
        self._ghost_move_credit += self.ghost_speed_ratio
        while self._ghost_move_credit + 1e-12 >= 1.0 and not event.died:
            self._ghost_move_credit -= 1.0
            self._move_ghosts(event)
            self._check_collision(event)

    def _consume_items(self, event: StepEvent) -> None:
        gs = self.game_state
        px, py = gs.pacman_pos
        if gs.food_mask[py, px]:
            gs.food_mask[py, px] = False
            event.ate_pellet = True
        elif self.power_pellet_enabled and self.layout.initial_power[py, px]:
            # Power pellet consumed
            event.ate_power = True
            gs.power_mode_timer = self.frightened_duration

    def _check_collision(self, event: StepEvent) -> None:
        gs = self.game_state
        px, py = gs.pacman_pos
        for i in range(self.num_ghosts):
            if not gs.ghost_alive[i]:
                continue
            gx, gy = gs.ghost_positions[i]
            if (gx, gy) == (px, py):
                if gs.power_mode_timer > 0:
                    gs.ghost_alive[i] = False
                    event.ate_ghosts += 1
                else:
                    event.died = True

    def _build_info(self, event: Optional[StepEvent] = None) -> dict:
        gs = self.game_state
        if gs is None:
            return {}
        ev = event or StepEvent()
        return {
            "step": gs.step_count,
            "score": self._cumulative_reward,
            "pellets_remaining": int(gs.food_mask.sum()),
            "event": {
                "ate_pellet": ev.ate_pellet,
                "ate_power": ev.ate_power,
                "ate_ghosts": ev.ate_ghosts,
                "died": ev.died,
                "won": ev.won,
                "remaining_pellets": ev.remaining_pellets,
                "total_pellets": ev.total_pellets,
                "episode_ended": ev.episode_ended,
            },
            "layout_id": self.layout.name,
        }
