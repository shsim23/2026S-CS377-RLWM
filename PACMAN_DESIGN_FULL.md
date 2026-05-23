# Pac-Man + World Model Design Document

> **Project**: Variance-Aware Policy Learning in State-Based World Models (CS377 Team 4)
> **Document scope**: Complete design — Environment (Part 1) + World Model (Part 2)
> **Version**: v2.0 (All design decisions frozen, ready for implementation)
> **Implementation target**: Two team members work on these parts; a third teammate handles policy (PPO + VPA) separately.

## Quick Navigation

- **Part 1: Environment Design** — Sections 0–14 — Pac-Man environment with stochastic ghosts
- **Part 2: World Model Design** — Sections 15–26 — JEPA-inspired RSSM-light with ensemble
- **Implementation Roadmap** — Section 27 — Week-by-week build order

## Document Reading Guide for Claude Code

If you are implementing this:
1. Read **Part 1 (sections 0–14)** first to understand the environment.
2. Then read **Part 2 (sections 15–26)** for the world model.
3. Then follow **Section 27 (Implementation Order)** to build incrementally.
4. The **Frozen Decisions** boxes (sections 14 and 26) are the authoritative spec. If anything below conflicts with those boxes, the boxes win.
5. All hyperparameters in `configs/` files are the single source of truth at runtime.

---

## 0. Project Context (recap)

This project investigates whether a reinforcement learning policy can be trained **entirely inside a learned world model** and then deployed in unseen environments via zero-shot transfer. Three design decisions define the project:

1. **State-space world model**: Inputs and outputs are 1D state vectors (coordinates and binary masks), not pixels. This avoids pixel-decoder overhead and improves long-horizon prediction accuracy.
2. **Variance-Penalized Advantage (VPA)**: PPO's advantage is penalized by the world model's epistemic uncertainty σ to prevent the policy from exploiting world model inaccuracies (model exploitation).
3. **Zero-shot transfer evaluation**: A policy trained inside the world model is evaluated on unseen Pac-Man layouts without further training.

The environment must support all three. It must be (a) lightweight enough to collect millions of transitions, (b) compatible with a fixed-dimensional state vector so the same world model works across layouts, and (c) contain enough stochasticity that the world model's σ is non-trivial (otherwise VPA reduces to vanilla PPO).

---

## 1. Design Philosophy

### 1.1 What we include vs. exclude

The full original Pac-Man (1980) has elements that interact badly with state-space world models: pseudo-random PRNG, frightened-mode ghost behavior, ghost personalities with cross-ghost dependencies (Inky depends on Blinky's position), warp tunnels (discontinuous transitions), and complex respawn sub-routines. We therefore use a **tiered design**:

| Tier | Components | Status |
|---|---|---|
| **Tier 1 (MVP)** | Grid movement, walls, pellets, single ghost (ε-greedy chase), one life, layout text format | ✅ Implemented first |
| **Tier 2 (extension)** | Multiple ghosts (2–4), power pellets + frightened mode | ⚙️ Config flag; enabled after MVP validated |
| **Tier 3 (excluded)** | Ghost personalities (Blinky/Pinky/Inky/Clyde), ghost house respawn, fruits, warp tunnels, scatter/chase mode switching | ❌ Out of scope |

The justification for excluding Tier 3 elements (especially ghost personalities) is that they introduce cross-agent dependencies and multimodal future state distributions that destabilize JEPA-style 1D world models without contributing to our main contribution (VPA). Probabilistic Dreaming (Maes et al., 2026) explicitly identifies multimodal predator behavior as a failure case for unimodal Gaussian latent models.

### 1.2 Why stochastic ghosts (ε-greedy) are essential

Our primary contribution, VPA, only demonstrates its value when the world model has non-trivial epistemic uncertainty σ. In a fully deterministic environment, the world model achieves near-perfect prediction → σ ≈ 0 → VPA degenerates to vanilla PPO and our experimental claim cannot be empirically supported.

We therefore use an ε-greedy chase policy for ghosts:

```
At each ghost step:
  with probability ε:  take a uniformly random legal action
  otherwise:           take the BFS-shortest-path action toward Pac-Man
```

Default ε = 0.2. This creates **multimodal future state distributions** near Pac-Man (where ghost behavior matters most), which the world model struggles to predict precisely, yielding high σ exactly in the regions where VPA's penalty should activate.

**Justification with respect to the original game**: The original Pac-Man (1980) was technically deterministic but is widely experienced as stochastic because (a) ghost behavior depends on a complex web of inter-ghost and Pac-Man positions, (b) the slightest deviation in Pac-Man's path triggers chain reactions in ghost behavior, and (c) frightened-mode movement is explicitly PRNG-based. The canonical stochastic variant **Ms. Pac-Man (1981)** introduces explicit randomness into ghost movement and is a well-established RL benchmark (e.g., DeepMind's Hybrid Reward Architecture). Our ε-greedy design follows the spirit of Ms. Pac-Man and aligns with this lineage.

---

## 2. Environment Specification

### 2.1 Configuration (YAML)

All environment behavior is controlled via a YAML config:

```yaml
# configs/env/mvp_tier1.yaml
env:
  layout_file: "layouts/train/medium_classic.txt"

  ghost:
    num_ghosts: 1                # 1 (MVP), expandable to 4
    policy: "chase_stochastic"   # "chase" | "chase_stochastic" | "random"
    epsilon: 0.2                 # random action probability for chase_stochastic
    personality: "homogeneous"   # "homogeneous" | "diverse" (placeholder, future work)

  power_pellet:
    enabled: false               # MVP: disabled; Tier 2: true
    frightened_duration: 30      # steps that frightened mode lasts

  reward:
    pellet: 1.0
    power_pellet: 5.0
    ghost_eaten: 10.0
    death: -10.0
    win: 50.0
    step_penalty: -0.01

  episode:
    max_steps: 500

  render_mode: null              # null | "human" | "rgb_array" | "ansi"
```

### 2.2 Layout format

We use a text-based grid format derived from the Berkeley AI Pac-Man project. Each character represents one cell:

| Character | Meaning |
|---|---|
| `%` | Wall |
| `.` | Pellet (food) |
| `o` | Power pellet |
| `P` | Pac-Man start position (exactly one) |
| `G` | Ghost start position (one or more) |
| (space) | Empty cell |

Example (`layouts/train/medium_classic.txt`):

```
%%%%%%%%%%%%%%%
%.............%
%.%%%.%%%%%.%.%
%.%...........%
%.%.%%%.%%%.%.%
%.....%.%.....%
%%%%%.%.%.%%%%%
%.....%G%.....%
%.%%%.%.%.%%%.%
%.............%
%.%.%%%%%%%.%.%
%.............%
%.%%%.%%%.%%%.%
%......P......%
%%%%%%%%%%%%%%%
```

**Constraints**:
- Maximum size: 21 × 21 (smaller layouts are padded with walls).
- Must be fully enclosed by walls.
- Exactly one `P`.
- At least `num_ghosts` instances of `G`.
- Total pellet count ≤ `MAX_FOOD_POSITIONS = 441`.

### 2.3 Layout directory split

```
layouts/
├── train/                     # Used for world model + policy training
│   ├── small_open.txt         # 11×11, sparse walls (easy)
│   ├── medium_classic.txt     # 15×15, standard maze
│   └── corridor.txt           # 15×15, narrow corridors (hard)
└── eval/                      # Held out, used only for zero-shot evaluation
    ├── unseen_topology.txt    # Different wall pattern
    └── unseen_size.txt        # Different size
```

The world model is trained on `train/` layouts and evaluated on `eval/` layouts.

---

## 3. State Vector Specification

### 3.1 Dimensions

The state vector has a **fixed dimension across all layouts**, ensuring that the same world model can be applied to unseen layouts (zero-shot transfer).

```python
# pacman_env/constants.py

MAX_GRID_H = 21
MAX_GRID_W = 21
MAX_GHOSTS = 4                                  # config.num_ghosts ≤ MAX_GHOSTS
MAX_FOOD_POSITIONS = MAX_GRID_H * MAX_GRID_W    # 441

# State vector composition
#   pacman_xy:    2   (normalized [-1, 1])
#   ghosts:       MAX_GHOSTS × 4 = 16
#                 per ghost: x, y, alive_flag, valid_slot_flag
#   food_mask:    441 (binary; 1 = pellet remaining)
#   wall_mask:    441 (binary; 1 = wall; layout conditioning)
#   power_timer:  1   (normalized [0, 1]; 0 if power pellet disabled or not active)
# Total: 2 + 16 + 441 + 441 + 1 = 901
STATE_DIM = 901
```

### 3.2 Coordinate normalization

All coordinates are mapped from raw grid indices to `[-1, 1]`:

```python
def normalize_coord(x: int, y: int) -> Tuple[float, float]:
    x_norm = 2.0 * x / (MAX_GRID_W - 1) - 1.0
    y_norm = 2.0 * y / (MAX_GRID_H - 1) - 1.0
    return x_norm, y_norm
```

**Rationale**: Coordinates and binary masks all live in [-1, 1] / [0, 1] ranges, giving the neural network a balanced input distribution. This is critical because our 901-dim input mixes continuous coordinates with binary masks; without normalization, gradient flow into mask features would be drowned out by coordinate features.

### 3.3 Ghost slot encoding

To support a variable number of ghosts (1 in MVP, up to 4 later) with fixed dimension, we use **slot-based encoding**:

```
For ghost slot i ∈ {0, 1, 2, 3}:
  if i < num_ghosts:
    x, y    = normalized coordinates of ghost i
    alive   = 1.0 if ghost i is alive else 0.0
    valid   = 1.0   (slot is in use)
  else:
    x, y    = 0.0
    alive   = 0.0
    valid   = 0.0   (slot is padding)
```

The `valid` flag tells the world model "ignore this slot". This is essential for Tier 2 transitions when `num_ghosts` is increased.

### 3.4 Wall mask as layout conditioning

The `wall_mask` (441 dim, flattened from 21×21) tells the world model where the walls are. **This is the key trick for cross-layout generalization**: the same world model can be applied to a new layout simply by changing the `wall_mask` portion of the input. The world model learns dynamics conditioned on the wall structure.

### 3.5 Why a flat 1D vector?

We chose a flat concatenated 1D vector (not `Dict` observations, not 2D channel grid) because:
- It matches our project's explicit design intent (state-space, not pixel-space).
- Most JEPA-style models accept 1D input naturally via MLP encoders.
- It removes ambiguity in serialization and replay buffer storage.

Internally, we use a `dataclass` (`GameState`) for human-readable code, then flatten to `np.ndarray` at the boundary with the world model.

---

## 4. Game Dynamics

### 4.1 Action space

```python
class Action(IntEnum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
    NOOP  = 4

action_space = gym.spaces.Discrete(5)
```

If Pac-Man attempts to move into a wall, the action is silently converted to NOOP (the agent stays in place). The world model learns this from the `wall_mask` portion of the state.

### 4.2 Step timing

Pac-Man and ghosts both move exactly 1 cell per step (synchronous). This simplifies world model learning by keeping transitions discrete and uniform.

**Catching dynamics**: Although both move at the same speed, ghosts can catch Pac-Man because:
- Pac-Man must navigate corners and dead-ends, creating inefficiency.
- The ε-greedy randomness occasionally encircles Pac-Man.
- Pac-Man is constrained by the goal of eating pellets, sometimes forcing suboptimal paths.

If catching turns out to be too rare during sanity checks (random policy survives > 95% of episodes), we tune **layout difficulty** (narrower corridors, more dead-ends) before changing the timing model. A `ghost_step_period` config field is reserved for future tuning but not exposed in MVP.

### 4.3 Step resolution order

Each `env.step(action)` resolves events in this fixed order:

1. **Pac-Man moves** (or NOOP if wall).
2. **Check intermediate collision**: if Pac-Man landed on a ghost, mark death event.
3. **Pac-Man consumes**: if Pac-Man's new cell has a pellet/power pellet, mark eating event and update `food_mask`.
4. **Ghosts move**: each ghost uses the `GhostController` policy.
5. **Check post-move collision**: if any ghost landed on Pac-Man (or vice versa via swap), mark death event.
6. **Check win**: if `food_mask.sum() == 0`, mark win event.
7. **Compute reward** from accumulated events.
8. **Compute termination flags** and return.

The two collision checks (steps 2 and 5) handle the edge case where Pac-Man and a ghost swap positions in the same step.

### 4.4 Termination

```python
terminated = (event.died) or (event.won)
truncated  = (game_state.step_count >= max_steps)
```

- `terminated=True`: episode ended due to game logic (ghost collision or all pellets eaten).
- `truncated=True`: episode hit the time limit (default 500 steps).
- Both follow Gymnasium 0.29+ conventions.

### 4.5 Lives

The agent has **one life only**. Any ghost collision ends the episode immediately. No respawn.

### 4.6 Reward structure

```python
@dataclass
class RewardConfig:
    pellet:        float = 1.0
    power_pellet:  float = 5.0   # Tier 2
    ghost_eaten:   float = 10.0  # Tier 2
    death:         float = -10.0
    win:           float = 50.0
    step_penalty:  float = -0.01
```

**Design rationale**:
- Magnitudes are kept small and within roughly the same order so that PPO's advantage and the world model's σ are on comparable scales. This is critical for VPA: if rewards range over ±1000 while σ ranges over [0, 1], the `λ·σ` penalty is overwhelmed and we cannot demonstrate VPA's effect.
- No reward shaping based on distance to ghosts is used. The clean narrative is that VPA's σ-penalty (not handcrafted reward) teaches the policy to avoid risky regions. Adding distance shaping would confound the ablation.
- During world model training, we apply a `symlog` transform (Hafner et al., 2023) to the reward target for additional stability across orders of magnitude.

---

## 5. Ghost Controller

### 5.1 ε-greedy chase

```python
class GhostController:
    def __init__(self, epsilon: float, np_random: np.random.Generator):
        self.epsilon = epsilon
        self.rng = np_random

    def step(self, ghost_pos, pacman_pos, walls) -> Tuple[int, int]:
        if self.rng.random() < self.epsilon:
            return self._random_legal_step(ghost_pos, walls)
        return self._bfs_chase_step(ghost_pos, pacman_pos, walls)
```

- `_bfs_chase_step`: Runs BFS from `ghost_pos` to `pacman_pos` over walkable cells, returns the first cell along the shortest path. Deterministic given walls and positions.
- `_random_legal_step`: Returns one of the 4-connected neighbors that is not a wall, sampled uniformly. If all neighbors are walls, returns the current position.

### 5.2 Homogeneous policy

In MVP, **all ghosts share the same controller** (homogeneous). When `num_ghosts > 1`, every ghost uses the same ε and the same BFS chase. The `personality: "diverse"` config option is present as a placeholder but not implemented.

### 5.3 Why this design supports VPA

The ε-greedy mixture creates **bimodal next-state distributions** for ghost positions:
- With probability `1 - ε`: ghost takes BFS-optimal step.
- With probability `ε`: ghost takes a random step.

A unimodal Gaussian world model cannot represent this distribution accurately and will exhibit high epistemic uncertainty σ near the ghost. This is exactly where VPA's penalty should activate, providing a clean experimental signal.

---

## 6. Gymnasium API

### 6.1 PacmanEnv interface

```python
class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}

    def __init__(
        self,
        layout_path: str,
        num_ghosts: int = 1,
        ghost_epsilon: float = 0.2,
        ghost_policy: str = "chase_stochastic",
        power_pellet_enabled: bool = False,
        frightened_duration: int = 30,
        max_steps: int = 500,
        reward_config: Optional[RewardConfig] = None,
        render_mode: Optional[str] = None,
    ):
        ...
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]: ...
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]: ...
    def render(self) -> Optional[np.ndarray]: ...
    def close(self) -> None: ...
```

### 6.2 Info dict contents

The `info` dict returned by `step()` contains diagnostic information not part of the state vector:

```python
info = {
    "step": int,                          # current step count
    "score": float,                       # cumulative reward this episode
    "pellets_remaining": int,             # for win condition tracking
    "event": {                            # what happened this step
        "ate_pellet": bool,
        "ate_power": bool,
        "ate_ghosts": int,
        "died": bool,
        "won": bool,
    },
    "layout_id": str,                     # which layout this env was constructed from
}
```

WandB logging consumes these fields directly.

### 6.3 Seeding

Standard Gymnasium pattern:
```python
env.reset(seed=42)   # seeds np.random.Generator inside the env
```

The same `np_random` is passed to `GhostController`, so ε-greedy randomness is fully reproducible given the seed.

### 6.4 Separation of step() and render()

`env.step()` returns the 901-dim state vector regardless of `render_mode`. `env.render()` is a separate operation. This separation provides:

1. **Training speed**: Pygame rendering is ~25× slower than pure logic. Setting `render_mode=None` during data collection gives a major throughput boost.
2. **Headless compatibility**: Server, Colab, and Docker environments lack a display. Pygame would crash; with `render_mode=None`, the env runs anywhere.
3. **Representation hygiene**: The world model sees only the 1D state vector. The sprite render is purely for human consumption. Keeping these separate enforces our state-space design intent at the code level.
4. **Testability**: Unit tests do not need to instantiate Pygame.

---

## 7. Renderer

The renderer is fully decoupled from the environment logic. Three modes:

| Mode | Output | Use case |
|---|---|---|
| `None` | nothing | Training (default) |
| `"ansi"` | ASCII string with walls, pellets, P, G symbols | Quick debugging in terminal |
| `"rgb_array"` | `np.ndarray (H, W, 3)` from Pygame | Recording rollouts, WandB video logging |
| `"human"` | live Pygame window | Demos and human play |

```python
class Renderer:
    def __init__(self, layout: Layout, mode: Optional[str] = None):
        self.layout = layout
        self.mode = mode
        if mode == "human":
            self._init_pygame_window()
        # rgb_array, ansi: lazy init

    def render(self, game_state: GameState) -> Optional[Union[str, np.ndarray]]:
        if self.mode is None:
            return None
        if self.mode == "ansi":
            return self._render_ansi(game_state)
        return self._render_pygame(game_state)
```

Sprite assets live in `pacman_env/sprites/`. The classic-Pac-Man look is recreated by overlaying sprites on the wall_mask + food_mask, so demos look like the original game while internally using the clean grid representation.

---

## 8. Directory Layout

```
pacman-wm/
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── env/
│   │   ├── default.yaml
│   │   ├── mvp_tier1.yaml
│   │   └── full_tier2.yaml
│   ├── world_model/                 # TBD (Part 2)
│   └── policy/                      # TBD (other teammate)
│
├── layouts/
│   ├── train/
│   │   ├── small_open.txt
│   │   ├── medium_classic.txt
│   │   └── corridor.txt
│   └── eval/
│       ├── unseen_topology.txt
│       └── unseen_size.txt
│
├── pacman_env/                      # ★ This document covers everything in here
│   ├── __init__.py
│   ├── env.py                       # PacmanEnv (Gymnasium API)
│   ├── layout.py                    # LayoutParser, Layout
│   ├── state.py                     # StateBuilder, GameState
│   ├── ghost.py                     # GhostController
│   ├── reward.py                    # RewardComputer, RewardConfig, StepEvent
│   ├── constants.py                 # MAX_GRID_*, STATE_DIM, Action, Tile
│   ├── renderer.py                  # Renderer (Pygame + ANSI)
│   └── sprites/
│       ├── pacman.png
│       ├── ghost_red.png
│       ├── wall.png
│       ├── pellet.png
│       └── power_pellet.png
│
├── world_model/                     # TBD (Part 2 of this design doc)
├── policy/                          # TBD (other teammate)
│
├── scripts/
│   ├── play_human.py                # Keyboard control for sanity check
│   ├── play_random.py               # Random policy rollout
│   ├── collect_data.py              # Bulk data collection for world model
│   ├── train_world_model.py         # TBD
│   └── eval_zero_shot.py            # TBD
│
├── data/
│   └── replay/
│       ├── small_open/
│       ├── medium_classic/
│       └── corridor/
│
└── tests/
    ├── test_layout_parser.py
    ├── test_env_step.py
    ├── test_state_vector.py
    └── test_determinism.py
```

---

## 9. Module-by-Module API

### 9.1 `pacman_env/constants.py`

```python
from enum import IntEnum

class Tile:
    WALL          = '%'
    FOOD          = '.'
    POWER_PELLET  = 'o'
    PACMAN_START  = 'P'
    GHOST_START   = 'G'
    EMPTY         = ' '

class Action(IntEnum):
    UP = 0; DOWN = 1; LEFT = 2; RIGHT = 3; NOOP = 4

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
STATE_DIM = 2 + MAX_GHOSTS * 4 + MAX_FOOD_POSITIONS * 2 + 1   # 901
```

### 9.2 `pacman_env/layout.py`

```python
@dataclass(frozen=True)
class Layout:
    name: str
    height: int
    width: int
    walls: np.ndarray              # (H, W) bool
    initial_food: np.ndarray       # (H, W) bool
    initial_power: np.ndarray      # (H, W) bool
    pacman_start: Tuple[int, int]
    ghost_starts: List[Tuple[int, int]]

    def to_padded_arrays(self) -> Dict[str, np.ndarray]:
        """Pad masks to (MAX_GRID_H, MAX_GRID_W) with walls."""

class LayoutParser:
    @staticmethod
    def from_file(path: str) -> Layout: ...
    @staticmethod
    def from_string(text: str, name: str = "anonymous") -> Layout: ...
    @staticmethod
    def validate(layout: Layout) -> None:
        """Raise ValueError if invalid (size, enclosure, P count, G count)."""
```

### 9.3 `pacman_env/state.py`

```python
@dataclass
class GameState:
    pacman_pos:       Tuple[int, int]
    ghost_positions:  List[Tuple[int, int]]
    ghost_alive:      List[bool]
    food_mask:        np.ndarray           # (H, W) bool
    power_mode_timer: int                  # 0 means inactive
    step_count:       int
    done:             bool

    def copy(self) -> "GameState": ...

class StateBuilder:
    def __init__(self, layout: Layout, num_ghosts: int):
        self.layout = layout
        self.num_ghosts = num_ghosts
        self._wall_mask_padded = self._compute_wall_mask()

    def build(self, game_state: GameState) -> np.ndarray:
        """Returns shape (STATE_DIM,), dtype float32, values in [-1, 1]."""
```

### 9.4 `pacman_env/ghost.py`

```python
class GhostController:
    def __init__(self, epsilon: float, np_random: np.random.Generator): ...

    def step(
        self,
        ghost_pos: Tuple[int, int],
        pacman_pos: Tuple[int, int],
        walls: np.ndarray,
    ) -> Tuple[int, int]:
        """Returns next position for one ghost."""
```

### 9.5 `pacman_env/reward.py`

```python
@dataclass
class RewardConfig:
    pellet: float = 1.0
    power_pellet: float = 5.0
    ghost_eaten: float = 10.0
    death: float = -10.0
    win: float = 50.0
    step_penalty: float = -0.01

@dataclass
class StepEvent:
    ate_pellet: bool = False
    ate_power: bool = False
    ate_ghosts: int = 0
    died: bool = False
    won: bool = False

class RewardComputer:
    def __init__(self, config: RewardConfig): ...
    def compute(self, prev: GameState, new: GameState, event: StepEvent) -> float: ...
```

### 9.6 `pacman_env/env.py`

```python
class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}

    def __init__(self, layout_path, num_ghosts=1, ghost_epsilon=0.2,
                 ghost_policy="chase_stochastic", power_pellet_enabled=False,
                 frightened_duration=30, max_steps=500,
                 reward_config=None, render_mode=None): ...

    def reset(self, seed=None, options=None): ...
    def step(self, action): ...
    def render(self): ...
    def close(self): ...

    # Private helpers
    def _init_game_state(self) -> GameState: ...
    def _move_pacman(self, action: int) -> None: ...
    def _move_ghosts(self) -> None: ...
    def _resolve_events(self, prev: GameState) -> StepEvent: ...
    def _build_info(self, event: Optional[StepEvent] = None) -> dict: ...
```

### 9.7 `pacman_env/renderer.py`

```python
class Renderer:
    def __init__(self, layout: Layout, mode: Optional[str] = None): ...
    def render(self, game_state: GameState) -> Optional[Union[str, np.ndarray]]: ...
    def close(self) -> None: ...
```

---

## 10. Data Collection and Storage

### 10.1 Replay format

One episode is stored as one `.npz` file:

```python
np.savez_compressed(
    f"data/replay/{layout_id}/episode_{idx:06d}.npz",
    states  = np.ndarray,    # (T, STATE_DIM)   float32
    actions = np.ndarray,    # (T,)             int64
    rewards = np.ndarray,    # (T,)             float32
    dones   = np.ndarray,    # (T,)             bool (terminated OR truncated)
    layout_id = layout_id,   # str
    seed      = seed,        # int
)
```

This format supports both sequence-based world model training (transformer/RNN) and per-transition training (MLP dynamics).

### 10.2 Collection script

```bash
python scripts/collect_data.py \
    --layout layouts/train/medium_classic.txt \
    --num-episodes 5000 \
    --policy random \
    --output-dir data/replay/medium_classic/ \
    --seed 0
```

Initial collection uses a random policy. Later collections can use partially trained policies for on-policy world model improvement.

---

## 11. Logging

All training scripts log to WandB. The environment itself does not call WandB; it only returns metrics via the `info` dict. Logging happens in scripts:

```python
# In scripts/collect_data.py (illustrative)
import wandb
wandb.init(project="cs377-team4", config=cfg)

for episode_idx in range(num_episodes):
    obs, info = env.reset(seed=base_seed + episode_idx)
    episode_reward = 0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
    wandb.log({
        "episode/reward": episode_reward,
        "episode/length": info["step"],
        "episode/died": info["event"]["died"],
        "episode/won": info["event"]["won"],
        "episode/pellets_remaining": info["pellets_remaining"],
    })
```

---

## 12. Testing Plan

### 12.1 `tests/test_layout_parser.py`
- Parses canonical Berkeley layouts without error.
- Rejects layouts missing P, missing G, or not enclosed.
- Rejects layouts exceeding 21×21.
- Pads smaller layouts correctly.

### 12.2 `tests/test_env_step.py`
- Pac-Man cannot walk into walls (action becomes NOOP).
- Pellet consumption updates `food_mask` and triggers reward.
- Ghost collision triggers death and terminates.
- Win condition fires when all pellets are eaten.
- Pac-Man/ghost swap (cells exchanged in one step) correctly detected as collision.

### 12.3 `tests/test_state_vector.py`
- Shape is `(STATE_DIM,)` and dtype is `float32`.
- All values are within `[-1, 1]`.
- Ghost slots beyond `num_ghosts` have `valid=0`.
- `wall_mask` matches the parsed layout.

### 12.4 `tests/test_determinism.py`
- Two environments with the same seed produce identical trajectories given identical action sequences.
- Resetting with the same seed reproduces the initial state exactly.

---

## 13. Sanity Check Protocol (before world model training)

Before any world model training, validate the environment:

1. **Human play** (`scripts/play_human.py`): Play with the keyboard. Confirm the game feels like Pac-Man and ghost behavior looks reasonable.
2. **Random policy stats** (`scripts/play_random.py` × 1000 episodes):
   - Death rate should be 30–50% (ghosts threaten but don't trivialize).
   - Average episode length should be 100–300 steps (most pellets get eaten).
   - Win rate should be near 0% (a random policy should not solve the game).
3. **Determinism check**: Run two seeded episodes, confirm identical trajectories.
4. **State vector inspection**: Print one state vector, verify by hand that pellets, walls, and positions are correctly encoded.

If the random-policy death rate is < 5%, the layout is too easy → narrow corridors or increase ε. If > 80%, too hard → widen corridors or decrease ε.

---

## 14. Frozen Decisions Summary

The following decisions are frozen. Any change requires explicit re-discussion.

```yaml
architecture:
  framework: Gymnasium + PyTorch
  state_dim: 901              # float32, normalized [-1, 1]
  action_space: Discrete(5)   # UP / DOWN / LEFT / RIGHT / NOOP

layout:
  format: Berkeley text (%, ., o, P, G)
  max_size: 21 x 21
  padding: walls
  split: layouts/train/, layouts/eval/

game_rules:
  ghost_count_mvp: 1          # config-expandable to 4
  ghost_policy: ε-greedy chase (ε = 0.2 default)
  ghost_personality: homogeneous (placeholder for "diverse")
  power_pellet_mvp: disabled  # config-expandable
  tunnel: none
  ghost_house: none
  fruit: none
  life: 1
  speed: pacman = ghost = 1 cell/step

termination:
  ghost_collision: terminated = True
  all_pellets_eaten: terminated = True (win)
  max_steps: truncated = True (default 500)

reward:
  pellet: +1.0
  power_pellet: +5.0          # tier 2
  ghost_eaten: +10.0          # tier 2
  death: -10.0
  win: +50.0
  step_penalty: -0.01
  # world-model-side: symlog transform applied during training

rendering:
  modes: [None, "ansi", "rgb_array", "human"]
  default: None
  toolkit: Pygame

data:
  format: NPZ per episode
  fields: states, actions, rewards, dones, layout_id, seed
  path: data/replay/{layout_id}/episode_{idx:06d}.npz

logging:
  framework: WandB
  source: env.info dict + scripts

testing:
  - tests/test_layout_parser.py
  - tests/test_env_step.py
  - tests/test_state_vector.py
  - tests/test_determinism.py
```

---

*End of Part 1. Part 2 (World Model design) begins below.*

---
---

# Part 2: World Model Design

## 15. World Model Design Philosophy

### 15.1 Position in the model-based RL landscape

Our world model is a **JEPA-inspired RSSM-light** architecture. This term encodes three commitments:

1. **JEPA-inspired**: We predict in latent space (not pixel space, not raw state space). The training objective on the dynamics side is essentially `MSE(predicted_latent, target_latent_from_encoder)`. We do not reconstruct the input state for the main loss. This aligns with the project proposal's stated commitment to JEPA.

2. **RSSM-light**: We adopt three practical components from Dreamer's Recurrent State-Space Model: (a) a recurrent latent `h_t` carrying short-term history, (b) explicit reward and termination heads, and (c) symlog reward targets. We **drop** Dreamer's stochastic categorical latent and its KL-balanced training, because in our setup the auxiliary heads (reward, done) and the ensemble already provide enough regularization and uncertainty signal.

3. **Ensemble for uncertainty**: 5 independent world models are trained on bootstrapped data subsets. The disagreement among ensemble members is the epistemic uncertainty σ that powers VPA in the policy team's PPO.

### 15.2 Why hybrid, not pure JEPA or pure RSSM

A pure JEPA model has no reward/done heads; this would require the policy team to bolt on a separate reward predictor, and our state-space inputs are low-dimensional enough that representation collapse is mild but possible without an anchor — reward/done prediction provides that anchor for free.

A pure RSSM (DreamerV3) includes a stochastic categorical latent and a state decoder, which would add: (a) a second uncertainty source (aleatoric, mixed with our ensemble's epistemic) that complicates VPA's interpretation, and (b) a pixel-style decoder that contradicts our state-space design intent.

The hybrid takes JEPA's latent-prediction discipline (no state reconstruction loss) and RSSM's auxiliary heads (clean signal for downstream policy), giving us a clean, debuggable model that fits our compute budget (≈ 2.2 M parameters total for the 5-member ensemble).

### 15.3 Relationship to recent work

This setup closely resembles **RWM-U / MOPO-PPO** (2026): ensemble-based epistemic uncertainty estimation combined with uncertainty-penalized policy optimization. The novelty in our project is applying this paradigm with a JEPA-style latent-prediction backbone (rather than direct state regression) and using the resulting σ inside PPO's advantage rather than as a reward penalty.

---

## 16. Architecture Overview

### 16.1 Information flow

```
                        ┌──────────────────────────────────────────┐
                        │  EnsembleWorldModel  (5 independent       │
                        │                       SingleWorldModels)  │
                        │                                            │
   state s_t  ──────► [Encoder] ──► z_t ──┐                          │
   (B, 901)                                │                          │
                                            ▼                          │
   action a_t ──► [ActionEmbedder] ─► a_emb ──┐                       │
   (B,)                                       ▼                        │
                                  [Latent Dynamics (GRU)]              │
                                  h_t ──► h_{t+1}                      │
                                            │                          │
                                            ▼                          │
                                  [Predictor (MLP)]                    │
                                            │                          │
                                            ▼                          │
                                  z_{t+1} (predicted latent)           │
                                            │                          │
                       ┌────────────────────┼─────────────────┐       │
                       ▼                    ▼                 ▼        │
                  [Reward Head]      [Done Head]    (z_{t+1}, h_{t+1}) │
                       │                    │                 │        │
                       ▼                    ▼                 ▼        │
                    r̂_t                  d̂_t (prob)     to next step   │
                        └──────────────────────────────────────────────┘
                                            │
                              Aggregate across 5 members:
                                            │
                                            ▼
                          {z_next_mean, h_next_mean, r̂_mean, d̂_mean, σ}
                                                                    ▲
                                            σ = std of z_next across ensemble
                                                                    │
                                                          (Used by policy team's VPA)
```

### 16.2 Single member component summary

| Module | Function | Input | Output |
|---|---|---|---|
| `StateEncoder` | s → z | (B, 901) | (B, 128) |
| `ActionEmbedder` | discrete action → embedding | (B,) int | (B, 32) |
| `LatentDynamics` (GRU + MLP) | (z, a, h) → (z_next, h_next) | (B, 128), (B, 32), (B, 256) | (B, 128), (B, 256) |
| `RewardHead` | (z, h) → r̂ (symlog scale) | (B, 128+256) | (B,) |
| `DoneHead` | (z, h) → P(done) | (B, 128+256) | (B,) |

Parameter counts per single model:

```
StateEncoder:    901 → 256 → 256 → 128       ≈ 100K params
ActionEmbedder:  Embedding(5, 32)            ≈ 0.2K params
LatentDynamics:  GRUCell(160, 256) + MLP     ≈ 200K params
RewardHead:      MLP(384 → 128 → 128 → 1)    ≈ 70K params
DoneHead:        MLP(384 → 128 → 128 → 1)    ≈ 70K params
─────────────────────────────────────────────────────
Per single model:                            ≈ 440K params
Ensemble of 5:                               ≈ 2.2M params
```

VRAM estimate: ≈ 1–2 GB on V100 with batch_size=64, seq_length=50.

---

## 17. Module Specifications (`world_model/modules/`)

### 17.1 `world_model/constants.py`

```python
# Architecture dimensions (all frozen)
STATE_DIM      = 901      # matches env.STATE_DIM from Part 1
ACTION_DIM     = 5        # Discrete(5): UP/DOWN/LEFT/RIGHT/NOOP
ACTION_EMB_DIM = 32

LATENT_DIM     = 128
GRU_HIDDEN     = 256
HIDDEN_DIM     = 256

POLICY_INPUT_DIM = LATENT_DIM + GRU_HIDDEN   # 384, what policy sees

# Ensemble
NUM_ENSEMBLE_MEMBERS = 5
```

### 17.2 `world_model/modules/encoder.py`

```python
import torch.nn as nn

class StateEncoder(nn.Module):
    """
    Encode raw state vector (B, 901) into latent z (B, latent_dim).
    3-layer MLP with LayerNorm + SiLU.
    """
    def __init__(self, state_dim: int = 901, latent_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, s):
        # s: (B, 901)  or (B, L, 901) — handle both with reshape
        if s.dim() == 3:
            B, L, D = s.shape
            return self.net(s.reshape(B * L, D)).reshape(B, L, -1)
        return self.net(s)
```

### 17.3 `world_model/modules/action.py`

```python
import torch.nn as nn

class ActionEmbedder(nn.Module):
    """Discrete action (5 options) → continuous embedding (32-dim)."""
    def __init__(self, num_actions: int = 5, emb_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, emb_dim)

    def forward(self, a):
        # a: (B,) or (B, L) int64
        return self.embedding(a)
```

### 17.4 `world_model/modules/dynamics.py`

```python
import torch
import torch.nn as nn

class LatentDynamics(nn.Module):
    """
    (z_t, a_emb_t, h_t)  →  (z_{t+1}, h_{t+1})

    Single-step API; for sequence training, called in a loop (see SingleWorldModel.forward_sequence).
    """
    def __init__(
        self,
        latent_dim: int = 128,
        action_emb_dim: int = 32,
        gru_hidden: int = 256,
        hidden: int = 256,
    ):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_emb_dim, gru_hidden)
        self.predictor = nn.Sequential(
            nn.Linear(gru_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_t, a_emb_t, h_t):
        gru_input = torch.cat([z_t, a_emb_t], dim=-1)
        h_next = self.gru(gru_input, h_t)
        z_next = self.predictor(h_next)
        return z_next, h_next
```

### 17.5 `world_model/modules/heads.py`

```python
import torch
import torch.nn as nn

class RewardHead(nn.Module):
    """(z, h) → predicted reward in symlog scale. Apply symexp at inference for raw scale."""
    def __init__(self, input_dim: int = 384, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        return self.net(x).squeeze(-1)   # (B,) in symlog space


class DoneHead(nn.Module):
    """(z, h) → P(done) ∈ [0, 1]."""
    def __init__(self, input_dim: int = 384, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        return torch.sigmoid(self.net(x).squeeze(-1))   # (B,)
```

### 17.6 `world_model/utils.py`

```python
import torch
import torch.nn.functional as F

# ─── Symlog transform (DreamerV3) ───
def symlog(x):
    """Compresses large-magnitude rewards. Used as the regression target."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    """Inverse of symlog. Used to convert predicted reward back to raw scale."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ─── Variance regularization (collapse safety) ───
def variance_regularization(z: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """
    Encourage each latent dimension to have non-trivial spread.
    Simplified VICReg variance term.

    Args:
        z: (..., latent_dim) — embeddings across some batch dim
    Returns:
        scalar loss; higher when latent dims are collapsing.
    """
    z_flat = z.reshape(-1, z.shape[-1])    # (N, latent_dim)
    std = torch.sqrt(z_flat.var(dim=0) + 1e-4)
    return F.relu(target_std - std).mean()


# ─── Weight initialization (consistent across ensemble) ───
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity="linear")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.GRUCell):
        for name, param in m.named_parameters():
            if "weight" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
    elif isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
```

---

## 18. Single Model & Ensemble (`world_model/single.py`, `ensemble.py`)

### 18.1 `world_model/single.py`

```python
import torch
import torch.nn as nn
from .modules.encoder import StateEncoder
from .modules.action import ActionEmbedder
from .modules.dynamics import LatentDynamics
from .modules.heads import RewardHead, DoneHead
from .constants import *

class SingleWorldModel(nn.Module):
    """One member of the ensemble. Full encoder + dynamics + heads."""

    def __init__(
        self,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        gru_hidden=GRU_HIDDEN,
        hidden_dim=HIDDEN_DIM,
        action_emb_dim=ACTION_EMB_DIM,
    ):
        super().__init__()
        self.encoder         = StateEncoder(state_dim, latent_dim, hidden_dim)
        self.action_embedder = ActionEmbedder(action_dim, action_emb_dim)
        self.dynamics        = LatentDynamics(latent_dim, action_emb_dim, gru_hidden, hidden_dim)
        self.reward_head     = RewardHead(latent_dim + gru_hidden, hidden=128)
        self.done_head       = DoneHead(latent_dim + gru_hidden, hidden=128)

        self.latent_dim = latent_dim
        self.gru_hidden = gru_hidden

    # ─── Training-time forward (teacher forcing through a sequence) ───
    def forward_sequence(self, states, actions, burnin: int = 5):
        """
        Args:
            states:  (B, L, state_dim)
            actions: (B, L) int64
            burnin:  number of initial steps to exclude from loss
        Returns dict of tensors needed for loss computation:
            z_all:    (B, L, latent_dim) — encoder output for every step
            z_preds:  (B, L-1, latent_dim) — dynamics-predicted next latent
            r_preds:  (B, L-1)  — reward head output (symlog scale)
            d_preds:  (B, L-1)  — done head output (sigmoid prob)
        """
        B, L, _ = states.shape

        # 1. Encode all states in parallel
        z_all = self.encoder(states)   # (B, L, latent_dim)

        # 2. Roll the GRU through the sequence with teacher forcing
        h = torch.zeros(B, self.gru_hidden, device=states.device)
        z_preds, r_preds, d_preds = [], [], []

        for t in range(L - 1):
            a_emb = self.action_embedder(actions[:, t])   # (B, 32)
            z_next, h = self.dynamics(z_all[:, t], a_emb, h)
            r = self.reward_head(z_next, h)
            d = self.done_head(z_next, h)

            z_preds.append(z_next)
            r_preds.append(r)
            d_preds.append(d)

        return {
            "z_all":   z_all,
            "z_preds": torch.stack(z_preds, dim=1),   # (B, L-1, latent_dim)
            "r_preds": torch.stack(r_preds, dim=1),   # (B, L-1)
            "d_preds": torch.stack(d_preds, dim=1),   # (B, L-1)
            "burnin":  burnin,
        }

    # ─── Single-step API for inference ───
    def encode(self, s):
        """state → (z, h_init=zeros)."""
        z = self.encoder(s)
        h = torch.zeros(s.shape[0], self.gru_hidden, device=s.device)
        return z, h

    def imagine_step(self, z, h, a):
        """One imagined step. Returns dict (z_next, h_next, reward[symlog], done[prob])."""
        a_emb = self.action_embedder(a)
        z_next, h_next = self.dynamics(z, a_emb, h)
        r_symlog = self.reward_head(z_next, h_next)
        d_prob = self.done_head(z_next, h_next)
        return {
            "z_next": z_next, "h_next": h_next,
            "reward_symlog": r_symlog, "done": d_prob,
        }
```

### 18.2 `world_model/ensemble.py`

```python
import torch
import torch.nn as nn
from typing import Dict
from .single import SingleWorldModel
from .utils import symexp, weight_init
from .constants import NUM_ENSEMBLE_MEMBERS, LATENT_DIM, GRU_HIDDEN

class EnsembleWorldModel(nn.Module):
    """
    Wraps K independent SingleWorldModels. Public API for policy team.

    Public methods:
        encode(s)                          — for warming up before imagination
        warmup_h(states, actions)          — Option B: real-history burn-in for h
        imagine_step(z, h, a)              — single-step rollout returning σ
    """

    def __init__(self, num_members: int = NUM_ENSEMBLE_MEMBERS, **kwargs):
        super().__init__()
        self.K = num_members
        self.members = nn.ModuleList([
            SingleWorldModel(**kwargs) for _ in range(num_members)
        ])

        # Critical: initialize each member with a DIFFERENT random seed
        for k, m in enumerate(self.members):
            torch.manual_seed(42 + k * 1000)
            m.apply(weight_init)
        torch.manual_seed(42)   # restore

        self.latent_dim = self.members[0].latent_dim
        self.gru_hidden = self.members[0].gru_hidden

    # ─── Public API: encode + warmup ───
    def encode(self, s: torch.Tensor):
        """Returns ensemble-mean z and zero-initialized h."""
        zs = torch.stack([m.encoder(s) for m in self.members], dim=1)   # (B, K, latent)
        z_mean = zs.mean(dim=1)
        h_init = torch.zeros(s.shape[0], self.gru_hidden, device=s.device)
        return z_mean, h_init

    def warmup_h(self, prefix_states: torch.Tensor, prefix_actions: torch.Tensor):
        """
        Option B initialization: run GRU over a real-history prefix to warm up h.

        Args:
            prefix_states:  (B, P, state_dim)  — last P real states
            prefix_actions: (B, P-1)           — actions taken in the prefix
        Returns:
            z_final, h_final  — ready for imagination
        """
        B, P, _ = prefix_states.shape
        # Average over ensemble for warmup. Each member contributes equally.
        z_final_list, h_final_list = [], []
        for m in self.members:
            z_all = m.encoder(prefix_states)   # (B, P, latent)
            h = torch.zeros(B, self.gru_hidden, device=prefix_states.device)
            for t in range(P - 1):
                a_emb = m.action_embedder(prefix_actions[:, t])
                _, h = m.dynamics(z_all[:, t], a_emb, h)
            z_final_list.append(z_all[:, -1])
            h_final_list.append(h)
        z_final = torch.stack(z_final_list, dim=1).mean(dim=1)
        h_final = torch.stack(h_final_list, dim=1).mean(dim=1)
        return z_final, h_final

    # ─── Public API: imagine one step with σ ───
    @torch.no_grad()
    def imagine_step(
        self, z: torch.Tensor, h: torch.Tensor, a: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        One imagined step using the full ensemble.

        Returns:
            z_next:  (B, latent_dim)  — mean across ensemble
            h_next:  (B, gru_hidden)  — mean across ensemble
            reward:  (B,)             — mean reward in RAW scale (symlog inverted)
            done:    (B,)             — mean done probability
            sigma:   (B,)             — scalar epistemic uncertainty
                                        (sqrt of mean across-member variance of z_next)
        """
        z_nexts, h_nexts, r_symlogs, d_probs = [], [], [], []
        for m in self.members:
            out = m.imagine_step(z, h, a)
            z_nexts.append(out["z_next"])
            h_nexts.append(out["h_next"])
            r_symlogs.append(out["reward_symlog"])
            d_probs.append(out["done"])

        z_stack = torch.stack(z_nexts, dim=1)       # (B, K, latent)
        h_stack = torch.stack(h_nexts, dim=1)       # (B, K, gru_hidden)
        r_stack = torch.stack(r_symlogs, dim=1)     # (B, K)
        d_stack = torch.stack(d_probs, dim=1)       # (B, K)

        # σ: epistemic uncertainty — across-member std of next latent, aggregated to scalar
        z_var_per_dim = z_stack.var(dim=1)          # (B, latent)
        sigma = torch.sqrt(z_var_per_dim.mean(dim=-1) + 1e-8)   # (B,)

        return {
            "z_next": z_stack.mean(dim=1),
            "h_next": h_stack.mean(dim=1),
            "reward": symexp(r_stack.mean(dim=1)),  # raw scale for policy
            "done":   d_stack.mean(dim=1),
            "sigma":  sigma,
        }

    # ─── Save / load ───
    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "K": self.K}, path)

    @classmethod
    def load(cls, path: str, **kwargs):
        ckpt = torch.load(path, map_location="cpu")
        model = cls(num_members=ckpt["K"], **kwargs)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
```

---

## 19. Loss Function

### 19.1 Four-term loss

```
L_total = L_latent + β_r · L_reward + β_d · L_done + β_v · L_var
```

Suggested coefficients (in `configs/world_model/jepa_default.yaml`):
```
β_r = 1.0
β_d = 1.0
β_v = 0.01
```

### 19.2 Loss computation (`world_model/loss.py`)

```python
import torch
import torch.nn.functional as F
from .utils import symlog, variance_regularization

def compute_world_model_loss(
    outputs: dict,
    rewards_raw: torch.Tensor,     # (B, L) — env's raw rewards
    dones: torch.Tensor,           # (B, L) — bool
    beta_reward: float = 1.0,
    beta_done: float = 1.0,
    beta_var: float = 0.01,
):
    """
    Compute the 4-term training loss for one ensemble member.

    Args:
        outputs: dict returned by SingleWorldModel.forward_sequence(...).
          - z_all:   (B, L, latent_dim)
          - z_preds: (B, L-1, latent_dim)
          - r_preds: (B, L-1)
          - d_preds: (B, L-1)
          - burnin:  int
        rewards_raw: (B, L) — raw rewards from the env, will be symlog-transformed.
        dones:       (B, L) — bool dones from the env.
    Returns:
        loss, dict of individual loss components for logging.
    """
    burnin = outputs["burnin"]
    z_all   = outputs["z_all"]
    z_preds = outputs["z_preds"]
    r_preds = outputs["r_preds"]
    d_preds = outputs["d_preds"]

    # Targets (next-step aligned)
    z_target = z_all[:, 1:].detach()                            # stop-grad: critical for JEPA
    r_target = symlog(rewards_raw[:, :-1])
    d_target = dones[:, :-1].float()

    # Slice off burnin
    z_pred_s   = z_preds[:, burnin:]
    z_target_s = z_target[:, burnin:]
    r_pred_s   = r_preds[:, burnin:]
    r_target_s = r_target[:, burnin:]
    d_pred_s   = d_preds[:, burnin:]
    d_target_s = d_target[:, burnin:]

    # ─── Individual losses ───
    L_latent = ((z_pred_s - z_target_s) ** 2).mean()
    L_reward = ((r_pred_s - r_target_s) ** 2).mean()
    L_done   = F.binary_cross_entropy(d_pred_s.clamp(1e-7, 1 - 1e-7), d_target_s)
    L_var    = variance_regularization(z_all)

    loss = L_latent + beta_reward * L_reward + beta_done * L_done + beta_var * L_var

    return loss, {
        "L_total":  loss.item(),
        "L_latent": L_latent.item(),
        "L_reward": L_reward.item(),
        "L_done":   L_done.item(),
        "L_var":    L_var.item(),
    }
```

### 19.3 Key implementation notes

- **Stop gradient on `z_target`**: this is non-negotiable for JEPA. Without `.detach()`, encoder and predictor can co-collapse to the trivial constant solution.
- **Burn-in (5 steps)**: GRU hidden state starts at zeros and needs a few steps to become informative. Loss computed over t ∈ [burnin, L−1].
- **Done loss clamping**: `clamp(1e-7, 1−1e-7)` prevents `log(0)` if `d_pred` outputs an extreme.
- **`L_var` uses `z_all` (not `z_preds`)**: we regularize the encoder, not the predictor.

---

## 20. Policy Interface Contract

This section is the **authoritative contract** between the world-model team and the policy team. Any change here requires coordination.

### 20.1 Imports the policy team uses

```python
from world_model.ensemble import EnsembleWorldModel
```

### 20.2 The four operations available to policy

```python
class EnsembleWorldModel:
    @property
    def latent_dim(self) -> int:   ...    # 128
    @property
    def gru_hidden(self) -> int:   ...    # 256

    # 1) Cold-start encoding (h initialized to zeros)
    def encode(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        """
        s: (B, 901)
        Returns:
            z: (B, 128)
            h: (B, 256)  — zeros
        """

    # 2) Option B initialization: warm up h from a real-history prefix
    def warmup_h(
        self, prefix_states: Tensor, prefix_actions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        prefix_states:  (B, P, 901)   — last P real states
        prefix_actions: (B, P-1)      — actions in the prefix
        Returns:
            z: (B, 128)               — encoder output at the LAST state
            h: (B, 256)               — GRU hidden after rolling through the prefix
        """

    # 3) One imagined step with epistemic uncertainty
    def imagine_step(
        self, z: Tensor, h: Tensor, a: Tensor,
    ) -> Dict[str, Tensor]:
        """
        z: (B, 128)
        h: (B, 256)
        a: (B,) int64

        Returns dict with:
            z_next: (B, 128)
            h_next: (B, 256)
            reward: (B,)   — RAW scale (symlog already inverted)
            done:   (B,)   — probability in [0, 1]
            sigma:  (B,)   — non-negative scalar epistemic uncertainty
        """

    # 4) Load from checkpoint (returns a frozen, eval-mode model)
    @classmethod
    def load(cls, path: str, **kwargs) -> "EnsembleWorldModel":
        """Returns model in eval() mode with requires_grad=False on all params."""
```

### 20.3 Standard imagination rollout pattern (for the policy team)

```python
def imagine_rollout(wm, policy, real_prefix, horizon=15):
    """One imagined trajectory starting from a real-history prefix."""
    # 1) Initialize from real history (Option B)
    z, h = wm.warmup_h(real_prefix["states"], real_prefix["actions"])

    rollout = []
    for t in range(horizon):
        policy_input = torch.cat([z, h], dim=-1)   # (B, 384)
        a = policy.sample_action(policy_input)
        out = wm.imagine_step(z, h, a)

        rollout.append({
            "state": policy_input,
            "action": a,
            "reward": out["reward"],
            "done":   out["done"],
            "sigma":  out["sigma"],
        })

        # Stop if all parallel rollouts predict done
        if (out["done"] > 0.5).all():
            break

        z, h = out["z_next"], out["h_next"]

    return rollout
```

### 20.4 VPA computation (policy team's responsibility, shown for context)

```python
# After collecting imagined rollouts
A_PPO = compute_gae(rollouts, gamma=0.99, lam=0.95)
sigmas = torch.stack([step["sigma"] for step in rollouts])
A_VPA  = A_PPO - lambda_vpa * sigmas
# Use A_VPA in PPO's clipped surrogate loss
```

`lambda_vpa` is the central ablation hyperparameter. Recommended sweep: `[0.0, 0.1, 0.5, 1.0, 2.0]`. `lambda_vpa=0.0` is the vanilla PPO baseline; nonzero values are VPA.

### 20.5 Policy team must NOT do

- ❌ Modify the world model's parameters during policy training (it's frozen).
- ❌ Call `imagine_step` outside of `torch.no_grad()` (it's already wrapped with `@torch.no_grad()`).
- ❌ Assume `sigma` has any specific scale; it must be normalized via `lambda_vpa`.
- ❌ Bypass `warmup_h` and start imagination from zero `h` for long rollouts (would create train/inference mismatch; we agreed on Option B).

---

## 21. Training Pipeline (3 Phases)

### 21.1 Pipeline overview

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Data collection                                                │
│   scripts/collect_data.py                                               │
│   Output: data/replay/{layout_id}/episode_*.npz                         │
│   Policy: 90% random + 10% greedy-nearest-pellet                        │
│   Target: 200K total transitions across 3 train layouts                 │
└────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: World model training (this team)                               │
│   scripts/train_world_model.py                                          │
│   Loop: bootstrap batch → forward → compute_loss → backward → step      │
│   Eval every 500 steps: K-step rollout error on held-out val            │
│   Stop: thresholds met OR early stopping patience (10 evals)            │
│   Output: checkpoints/best.pt                                           │
└────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Policy training (policy team)                                  │
│   scripts/train_policy_vpa.py  (not our responsibility, but documented) │
│   Uses checkpoints/best.pt as frozen WM                                 │
│   PPO + VPA with λ ∈ {0.0, 0.1, 0.5, 1.0, 2.0}                          │
│   Eval on real env (train layouts + held-out eval layouts)              │
└────────────────────────────────────────────────────────────────────────┘
```

This is **Mode A (two-stage decoupled)**. If time permits, Mode B (iterative re-collection with learned policy) is a worthwhile ablation but not required for MVP.

### 21.2 Phase 1: Data collection (`scripts/collect_data.py`)

```python
"""
Usage:
  python scripts/collect_data.py \
    --layout layouts/train/medium_classic.txt \
    --num-transitions 70000 \
    --policy mixed \
    --output-dir data/replay/medium_classic \
    --seed 0
"""

def mixed_policy(obs, env, p_greedy=0.1):
    """90% random, 10% greedy toward nearest pellet."""
    if env.np_random.random() < p_greedy:
        return greedy_nearest_pellet(env.game_state)
    return env.action_space.sample()


def greedy_nearest_pellet(game_state):
    """BFS from pacman to nearest pellet, return first action of the path."""
    # Implementation: BFS over walkable cells, expand neighbors, track parents,
    # stop at the first pellet found, reconstruct path, return first step direction.
    ...
```

Collection across train layouts (target: 70K transitions per layout):

| Layout | Transitions | Approx. episodes |
|---|---|---|
| `small_open.txt` | 70,000 | ~250 |
| `medium_classic.txt` | 70,000 | ~200 |
| `corridor.txt` | 70,000 | ~280 |
| **Total** | **210,000** | **~730** |

Train/val split: hold out the last 10% of episodes per layout as `val/`.

### 21.3 Phase 2: World model training (`scripts/train_world_model.py`)

Hyperparameter file `configs/world_model/jepa_default.yaml`:

```yaml
world_model:
  # Architecture (frozen, must match constants.py)
  state_dim: 901
  latent_dim: 128
  gru_hidden: 256
  hidden_dim: 256
  action_emb_dim: 32
  num_ensemble_members: 5

  # Training
  batch_size: 64
  seq_length: 50
  burnin: 5
  learning_rate: 3.0e-4
  weight_decay: 0.0
  optimizer: adam
  grad_clip: 1.0

  # Loss weights
  beta_reward: 1.0
  beta_done: 1.0
  beta_var: 0.01

  # Schedule
  max_train_steps: 50000
  eval_every: 500
  patience: 10

  # K-step rollout eval
  k_step: 10
  n_eval_trajectories: 100

  # Stopping thresholds (all must be met)
  threshold_latent_mse: 0.05
  threshold_reward_mse: 0.10
  threshold_done_err: 0.10
```

Main training loop:

```python
def train(cfg):
    # 1) Setup
    ensemble = EnsembleWorldModel(num_members=cfg.num_ensemble_members)
    optimizers = [Adam(m.parameters(), lr=cfg.learning_rate) for m in ensemble.members]

    train_buffer = SequenceReplayBuffer(base_dir="data/replay", split="train")
    val_buffer   = SequenceReplayBuffer(base_dir="data/replay", split="val")

    wandb.init(project="cs377-team4", config=cfg)

    best_val_score = float("inf")
    patience_counter = 0

    # 2) Train loop
    for step in range(cfg.max_train_steps):
        loss_logs = train_one_step(ensemble, optimizers, train_buffer, cfg, step)

        if step % 100 == 0:
            wandb.log({**loss_logs, "step": step})

        if step % cfg.eval_every == 0 and step > 0:
            val_metrics = evaluate_k_step_rollout(
                ensemble, val_buffer, K=cfg.k_step, N=cfg.n_eval_trajectories,
            )
            wandb.log({**val_metrics, "step": step})

            score = val_metrics["k_step_latent_mse"]
            if score < best_val_score:
                best_val_score = score
                patience_counter = 0
                ensemble.save("checkpoints/best.pt")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"Early stopping at step {step}")
                    break

            if (val_metrics["k_step_latent_mse"] < cfg.threshold_latent_mse and
                val_metrics["k_step_reward_mse"] < cfg.threshold_reward_mse and
                val_metrics["k_step_done_err"]   < cfg.threshold_done_err):
                print(f"All thresholds met at step {step}")
                ensemble.save("checkpoints/best.pt")
                break

    ensemble.save("checkpoints/final.pt")
```

### 21.4 Per-step training (bootstrap sampling)

```python
def train_one_step(ensemble, optimizers, train_buffer, cfg, global_step):
    """One gradient step for every ensemble member, using bootstrap sampling."""
    total = {}
    for k, member in enumerate(ensemble.members):
        # Bootstrap: each member sees a different batch this step
        bootstrap_seed = global_step * cfg.num_ensemble_members + k
        batch = train_buffer.sample_sequence(
            batch_size=cfg.batch_size,
            seq_length=cfg.seq_length,
            bootstrap_seed=bootstrap_seed,
        )

        outputs = member.forward_sequence(
            states=batch["states"],
            actions=batch["actions"],
            burnin=cfg.burnin,
        )
        loss, log_dict = compute_world_model_loss(
            outputs=outputs,
            rewards_raw=batch["rewards"],
            dones=batch["dones"],
            beta_reward=cfg.beta_reward,
            beta_done=cfg.beta_done,
            beta_var=cfg.beta_var,
        )

        optimizers[k].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(member.parameters(), cfg.grad_clip)
        optimizers[k].step()

        for key, val in log_dict.items():
            total[f"member_{k}/{key}"] = val

    return total
```

---

## 22. Replay Buffer (`world_model/replay_buffer.py`)

### 22.1 SequenceReplayBuffer

```python
import numpy as np
from pathlib import Path
from typing import Optional

class SequenceReplayBuffer:
    """
    Episode-per-file NPZ replay buffer with sequence sampling.

    Directory structure expected:
        base_dir/
        ├── layout_a/
        │   ├── train/
        │   │   ├── episode_000000.npz
        │   │   └── ...
        │   └── val/
        │       └── ...
        ├── layout_b/
        │   └── ...

    Each .npz file contains:
        states  (T, 901) float32
        actions (T,)     int64
        rewards (T,)     float32
        dones   (T,)     bool
    """

    def __init__(self, base_dir: str, split: str = "train"):
        self.base_dir = Path(base_dir)
        self.split = split
        self.episode_files = sorted(self.base_dir.rglob(f"{split}/*.npz"))
        if not self.episode_files:
            raise FileNotFoundError(f"No episodes found under {base_dir}/*/{split}/")
        print(f"[ReplayBuffer:{split}] Found {len(self.episode_files)} episodes.")

    def sample_sequence(
        self,
        batch_size: int,
        seq_length: int,
        bootstrap_seed: Optional[int] = None,
    ) -> dict:
        """Sample B episodes, take a random L-length window from each."""
        rng = np.random.RandomState(bootstrap_seed) if bootstrap_seed is not None else np.random
        chosen_files = rng.choice(self.episode_files, size=batch_size, replace=True)

        states_b, actions_b, rewards_b, dones_b = [], [], [], []
        for path in chosen_files:
            ep = np.load(path)
            T = len(ep["states"])
            if T < seq_length:
                # Pad with last state repeated; mark padded steps as done.
                pad = seq_length - T
                s = np.concatenate([ep["states"], np.tile(ep["states"][-1:], (pad, 1))], axis=0)
                a = np.concatenate([ep["actions"], np.full(pad, 4, dtype=np.int64)], axis=0)  # NOOP
                r = np.concatenate([ep["rewards"], np.zeros(pad, dtype=np.float32)], axis=0)
                d = np.concatenate([ep["dones"], np.ones(pad, dtype=bool)], axis=0)
            else:
                start = rng.randint(0, T - seq_length + 1)
                s = ep["states"][start:start + seq_length]
                a = ep["actions"][start:start + seq_length]
                r = ep["rewards"][start:start + seq_length]
                d = ep["dones"][start:start + seq_length]
            states_b.append(s); actions_b.append(a); rewards_b.append(r); dones_b.append(d)

        import torch
        return {
            "states":  torch.from_numpy(np.stack(states_b)).float(),
            "actions": torch.from_numpy(np.stack(actions_b)).long(),
            "rewards": torch.from_numpy(np.stack(rewards_b)).float(),
            "dones":   torch.from_numpy(np.stack(dones_b)),
        }

    def sample_trajectories(self, N: int) -> list:
        """Full episodes for K-step rollout evaluation."""
        chosen = np.random.choice(self.episode_files, size=min(N, len(self.episode_files)), replace=False)
        trajs = []
        for path in chosen:
            ep = np.load(path)
            trajs.append({k: ep[k] for k in ep.keys()})
        return trajs
```

### 22.2 Multi-layout uniform mixing

`base_dir.rglob("train/*.npz")` collects episodes from every layout's `train/` directory. Bootstrap sampling with `replace=True` then mixes them uniformly. No per-layout balancing is needed because we collect roughly equal transitions per layout (70K each).

---

## 23. K-Step Rollout Evaluation (`world_model/eval.py`)

### 23.1 The metric that decides "is this world model usable?"

```python
import torch
import numpy as np

@torch.no_grad()
def evaluate_k_step_rollout(
    ensemble: "EnsembleWorldModel",
    val_buffer: "SequenceReplayBuffer",
    K: int = 10,
    N: int = 100,
):
    """
    Autoregressive K-step prediction error on the validation set.
    This mimics actual imagination usage during policy training.
    """
    latent_errs, reward_errs, done_errs = [], [], []
    sigma_values = []

    for traj in val_buffer.sample_trajectories(N):
        states  = torch.from_numpy(traj["states"]).float()
        actions = torch.from_numpy(traj["actions"]).long()
        rewards = traj["rewards"]
        dones   = traj["dones"]
        T = len(states)
        if T < K + 1:
            continue

        # Start from t=0 (no prefix warmup for this baseline metric)
        z, h = ensemble.encode(states[0:1])

        for t in range(K):
            a = actions[t:t+1]
            out = ensemble.imagine_step(z, h, a)

            # Compare imagined z_next with encoder(true s_{t+1})
            z_true, _ = ensemble.encode(states[t+1:t+2])
            latent_errs.append(((out["z_next"] - z_true) ** 2).mean().item())
            reward_errs.append((out["reward"].item() - float(rewards[t])) ** 2)
            done_errs.append(abs(out["done"].item() - float(dones[t])))
            sigma_values.append(out["sigma"].item())

            if out["done"].item() > 0.5:
                break  # imagined episode ended; stop the rollout

            # Autoregressive: use imagined z_next, h_next
            z, h = out["z_next"], out["h_next"]

    return {
        "k_step_latent_mse": float(np.mean(latent_errs)) if latent_errs else float("inf"),
        "k_step_reward_mse": float(np.mean(reward_errs)) if reward_errs else float("inf"),
        "k_step_done_err":   float(np.mean(done_errs))   if done_errs else float("inf"),
        "sigma_mean":        float(np.mean(sigma_values)) if sigma_values else 0.0,
    }
```

### 23.2 Logging cadence

- Every 500 train steps: compute K-step rollout error on val.
- Log to WandB: `eval/k_step_latent_mse`, `eval/k_step_reward_mse`, `eval/k_step_done_err`, `eval/sigma_mean`.
- Also periodically log ensemble diversity stats and `dead_dims` (latent dims with std < 0.01).

### 23.3 Sanity checks before declaring done

When `k_step_latent_mse < 0.05`, also confirm:
- `sigma_mean > 0.001` — ensemble isn't collapsed (would make VPA useless).
- `dead_dims < 10` (out of 128) — latent space is being used.
- Loss curve has plateaued for 5+ evals.

If any of these fail, **don't stop early**; the world model might be technically accurate but useless for VPA.

---

## 24. Reference Code Mapping

We do not fork any single repo. We compose components from multiple references into our own directory structure.

| Component | Reference | What we borrow |
|---|---|---|
| Symlog / symexp | `DrunkJin/dreamer-from-scratch` `utils.py` | 5-line implementation, copied directly |
| GRU-based dynamics structure | `NM512/dreamerv3-torch` `models.py` (RSSM block) | GRU input/output flow, hidden state handling |
| Reward/done head pattern | `DrunkJin/dreamer-from-scratch` `networks.py` | MLP topology with LayerNorm+SiLU |
| Sequence training with burn-in | `DrunkJin/dreamer-from-scratch` `rssm.py` | Teacher forcing through L steps |
| Ensemble dynamics structure | `facebookresearch/mbrl-lib` `models/gaussian_mlp.py` | Bootstrap sampling, per-member optimizer |
| Variance regularization | `lucas-maes/le-wm` (greatly simplified) | Conceptual basis only — we use a much simpler 1-line variant |
| Episode-per-file NPZ replay | `DrunkJin/dreamer-from-scratch` `replay_buffer.py` | Trajectory-aware sampling |

**Licenses**: All listed repos are MIT-licensed. Final report should attribute these in a `References` section. We do not redistribute their code verbatim — we re-implement following our spec, using these as inspiration.

---

## 25. Updated Directory Layout (Full Project)

```
pacman-wm/
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── env/
│   │   ├── default.yaml
│   │   ├── mvp_tier1.yaml
│   │   └── full_tier2.yaml
│   ├── world_model/
│   │   └── jepa_default.yaml
│   └── policy/                      # placeholder for policy team
│       └── ppo_vpa.yaml
│
├── layouts/
│   ├── train/
│   │   ├── small_open.txt
│   │   ├── medium_classic.txt
│   │   └── corridor.txt
│   └── eval/
│       ├── unseen_topology.txt
│       └── unseen_size.txt
│
├── pacman_env/                      # ★ Part 1 (environment)
│   ├── __init__.py
│   ├── env.py
│   ├── layout.py
│   ├── state.py
│   ├── ghost.py
│   ├── reward.py
│   ├── constants.py
│   ├── renderer.py
│   └── sprites/
│       ├── pacman.png
│       ├── ghost_red.png
│       ├── wall.png
│       ├── pellet.png
│       └── power_pellet.png
│
├── world_model/                     # ★ Part 2 (this team)
│   ├── __init__.py
│   ├── constants.py                 # LATENT_DIM, GRU_HIDDEN, etc.
│   ├── utils.py                     # symlog, symexp, variance_regularization, weight_init
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── encoder.py               # StateEncoder
│   │   ├── action.py                # ActionEmbedder
│   │   ├── dynamics.py              # LatentDynamics (GRU + MLP)
│   │   └── heads.py                 # RewardHead, DoneHead
│   ├── single.py                    # SingleWorldModel
│   ├── ensemble.py                  # EnsembleWorldModel (★ public API)
│   ├── loss.py                      # compute_world_model_loss
│   ├── replay_buffer.py             # SequenceReplayBuffer
│   ├── eval.py                      # evaluate_k_step_rollout
│   └── interface.py                 # WorldModelProtocol (typing-only contract)
│
├── policy/                          # placeholder for policy team
│   └── (TBD by policy teammate)
│
├── scripts/
│   ├── play_human.py                # human sanity check
│   ├── play_random.py               # random-policy stats
│   ├── collect_data.py              # Phase 1
│   ├── train_world_model.py         # Phase 2
│   └── eval_world_model.py          # standalone WM eval (post-training)
│
├── data/
│   └── replay/
│       ├── small_open/
│       │   ├── train/
│       │   └── val/
│       ├── medium_classic/
│       │   ├── train/
│       │   └── val/
│       └── corridor/
│           ├── train/
│           └── val/
│
├── checkpoints/
│   ├── best.pt                      # output of Phase 2
│   ├── final.pt
│   └── metadata.json
│
└── tests/
    ├── test_layout_parser.py
    ├── test_env_step.py
    ├── test_state_vector.py
    ├── test_determinism.py
    ├── test_world_model_shapes.py   # ★ new: verify all tensor shapes
    ├── test_ensemble_diversity.py   # ★ new: assert σ > 0 on random data
    └── test_symlog_roundtrip.py     # ★ new: symexp(symlog(x)) ≈ x
```

---

## 26. Frozen Decisions Summary (Part 2)

```yaml
# WORLD MODEL — FROZEN DECISIONS (any change requires explicit re-discussion)

architecture:
  paradigm: JEPA-inspired RSSM-light
  decoder: none (state-space, no reconstruction loss)
  stochastic_latent: false (deterministic only, no categorical)
  recurrent_state: GRU (single layer)

dimensions:
  state_dim: 901
  action_dim: 5 (Discrete)
  action_emb_dim: 32
  latent_dim: 128
  gru_hidden: 256
  hidden_dim: 256
  policy_input_dim: 384  # latent + gru_hidden

ensemble:
  num_members: 5
  diversity_method: random_init + bootstrap_sampling
  member_independence: full (no shared modules, no shared optimizer)
  sigma_aggregation: sqrt(mean across-member variance of z_next)
  sigma_shape: scalar (B,)

loss:
  L_latent: MSE(z_pred, z_target.detach())
  L_reward: MSE in symlog space
  L_done: BCE
  L_var: VICReg variance term (no covariance)
  weights:
    beta_reward: 1.0
    beta_done: 1.0
    beta_var: 0.01

training:
  mode: two_stage_decoupled  # Mode A
  sequence_length: 50
  burnin: 5
  batch_size: 64
  optimizer: Adam(lr=3e-4)
  grad_clip: 1.0
  teacher_forcing: true  # during training; rollout uses autoregression

data_collection:
  total_transitions: ~210,000  # 70K per train layout × 3
  policy: mixed (90% random + 10% greedy_nearest_pellet)
  train_val_split: 90/10 per layout

evaluation:
  metric: k_step_rollout_error (K=10, N=100 trajectories)
  thresholds:
    latent_mse: 0.05
    reward_mse: 0.10
    done_err: 0.10
  early_stopping_patience: 10 evals
  eval_every: 500 train steps

policy_interface:
  contract_module: world_model/interface.py
  required_methods: [encode, warmup_h, imagine_step, load]
  initial_h_strategy: option_B (warmup from real history prefix)
  returned_reward_scale: raw (symexp applied)
  sigma_form: scalar per batch element (B,)
  frozen_during_policy_training: true (requires_grad=False)

logging:
  framework: WandB
  per_step_metrics: [L_total, L_latent, L_reward, L_done, L_var]
  per_eval_metrics: [k_step_latent_mse, k_step_reward_mse, k_step_done_err, sigma_mean, dead_dims]

excluded_tricks:  # explicitly NOT used
  - KL_balancing (DreamerV3) — no stochastic latent
  - free_bits — no KL term
  - state_decoder — JEPA design intent
  - EMA target encoder — variance regularization replaces it
  - VICReg covariance term — too aggressive for our small latent
```

---

## 27. Implementation Order & Timeline

### 27.1 Build order (incremental, each step independently testable)

```
─── Week 1–2 (Part 1: Environment) ──────────────────────────────────
[A1]  pacman_env/constants.py        ← types, dimensions, action enum
[A2]  pacman_env/layout.py           ← Layout, LayoutParser
[A3]  pacman_env/ghost.py            ← GhostController (ε-greedy chase)
[A4]  pacman_env/reward.py           ← RewardConfig, StepEvent, RewardComputer
[A5]  pacman_env/state.py            ← GameState, StateBuilder
[A6]  pacman_env/renderer.py         ← Renderer (Pygame + ANSI)
[A7]  pacman_env/env.py              ← PacmanEnv (Gymnasium)
[A8]  tests/test_layout_parser.py
[A9]  tests/test_env_step.py
[A10] tests/test_state_vector.py
[A11] tests/test_determinism.py
[A12] scripts/play_human.py          ← keyboard sanity check
[A13] scripts/play_random.py         ← random-policy statistics

→ Milestone M1: Random policy stats look sane (death rate 30–50%).

[A14] scripts/collect_data.py        ← Phase 1 data collection
[A15] Layouts (layouts/train/*, layouts/eval/*)

→ Milestone M2: 210K transitions collected, split into train/val.

─── Week 3 (Part 2: World Model implementation) ─────────────────────
[B1]  world_model/constants.py
[B2]  world_model/utils.py           ← symlog, symexp, weight_init, var_reg
[B3]  world_model/modules/encoder.py
[B4]  world_model/modules/action.py
[B5]  world_model/modules/dynamics.py
[B6]  world_model/modules/heads.py
[B7]  world_model/single.py          ← SingleWorldModel
[B8]  world_model/ensemble.py        ← EnsembleWorldModel
[B9]  world_model/loss.py            ← compute_world_model_loss
[B10] world_model/replay_buffer.py   ← SequenceReplayBuffer
[B11] world_model/eval.py            ← evaluate_k_step_rollout
[B12] world_model/interface.py       ← WorldModelProtocol
[B13] tests/test_world_model_shapes.py
[B14] tests/test_ensemble_diversity.py
[B15] tests/test_symlog_roundtrip.py

→ Milestone M3: All shape tests pass; ensemble shows nonzero σ on random init.

─── Week 4 (Phase 2 training + validation) ──────────────────────────
[B16] scripts/train_world_model.py   ← full training script
[B17] configs/world_model/jepa_default.yaml
[B18] Run on collected data, monitor WandB
[B19] scripts/eval_world_model.py    ← standalone WM evaluation

→ Milestone M4: K-step rollout thresholds met. checkpoints/best.pt saved.

─── Week 5–6 (Policy integration + experiments) ─────────────────────
[C] Policy teammate uses checkpoints/best.pt
[C] λ_VPA sweep: {0.0, 0.1, 0.5, 1.0, 2.0}
[C] Zero-shot evaluation on layouts/eval/*

→ Milestone M5: Final results, ablations, plots for the report.
```

### 27.2 Implementation tips for Claude Code

When implementing each file:

1. **Always start by reading `constants.py`** — every other module depends on the dimension constants. Don't hardcode 901 or 128 anywhere; import from constants.

2. **Test shape compatibility immediately**. After implementing each module, create a small test (in `tests/`) that runs a dummy forward pass:
   ```python
   def test_encoder_shape():
       enc = StateEncoder()
       s = torch.randn(4, 901)
       z = enc(s)
       assert z.shape == (4, 128)
   ```

3. **Verify sequence handling separately**. The `forward_sequence` method handles (B, L, ...) tensors. Test this with B=2, L=10 dummy data before touching the real replay buffer.

4. **Bootstrap correctness check**: in `train_one_step`, log the actual indices each member sees. They should differ. If two members see the same batch, the bootstrap seed isn't propagating correctly.

5. **Watch for the GRU hidden state during evaluation**. K-step rollout uses autoregression — easy to accidentally leak ground-truth z. The eval code in section 23 has the correct pattern: only `actions[t]` is from ground truth, `z` and `h` come from previous predictions.

6. **Symlog applies to reward only**. Never to state, never to latent. The reward head outputs in symlog space and `ensemble.imagine_step` applies symexp before returning.

7. **`detach()` on z_target is non-negotiable**. If you remove it accidentally, the latent loss can drive both encoder and predictor to zero (collapse).

8. **Use the `interface.py` Protocol class as a typing-only contract**. It has no implementation, but importing it in the policy team's code lets them verify their assumptions about our API.

### 27.3 Definition of Done (per milestone)

| Milestone | Definition of Done |
|---|---|
| M1 (env runs) | `play_random.py` runs 1000 episodes without error; death rate in [30%, 50%]; determinism test passes |
| M2 (data collected) | `data/replay/*/train/` has ~190K total transitions; `val/` has ~21K; all `.npz` files load cleanly |
| M3 (modules implemented) | All shape tests pass; `EnsembleWorldModel.imagine_step` returns sigma > 0 on random init data |
| M4 (WM trained) | `checkpoints/best.pt` exists; `k_step_latent_mse < 0.05`, `reward_mse < 0.10`, `done_err < 0.10` all logged in WandB |
| M5 (results) | Policy with VPA outperforms vanilla PPO on at least one held-out eval layout; λ sweep documented |

### 27.4 Hand-off documents

When milestone M4 is reached, the world-model team produces:

1. `checkpoints/best.pt` — the frozen ensemble.
2. `checkpoints/metadata.json` — training config, final validation metrics, total train steps, layout list.
3. A short markdown note (`HANDOFF.md`) describing:
   - How to load the model (1 line: `EnsembleWorldModel.load("checkpoints/best.pt")`).
   - The exact API (refer to section 20 of this document).
   - Known limitations: typical σ range observed during validation, layouts that work best.

---

*End of design document. Implementation begins at section 27.1, step [A1].*
