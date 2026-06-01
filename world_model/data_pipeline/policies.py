"""Policy pool for offline data collection (spec §3.3).

A world model is only accurate inside the state distribution it was trained on.
To avoid narrow-policy collapse and to expose rare reward events, the dataset is
collected with a *mixed* policy pool spanning novice→expert behaviour:

  * `RandomPolicy`     — pure exploration (covers deaths, cornered states, ...).
  * `GreedyBFSPolicy`  — competent pellet-seeking (covers efficient trajectories
                         and the food-eat reward bucket).
  * `EpsilonGreedy`    — wraps a base policy, injecting `epsilon` random actions
                         to span the competence spectrum without retraining.

PPO checkpoints are intentionally excluded for now: the available agent was
trained on a single layout and does not transfer to freshly generated mazes.
The greedy-BFS heuristic generalizes to any layout, and the ε knob recreates the
"novice→expert" spread DreamerV3 would obtain online.

A policy is a callable `policy(env) -> int` (action index), matching the
existing convention in `scripts/collect_data.py`.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, List, Tuple

import numpy as np

from pacman_env import PacmanEnv
from pacman_env.constants import Action


# --------------------------------------------------------------------------- #
def greedy_nearest_pellet(env: PacmanEnv) -> int:
    """BFS from Pac-Man to the nearest pellet; return the first action index.

    Lifted from `scripts/collect_data.py` so the pipeline does not depend on a
    script module. Falls back to a random action when no pellet is reachable.
    """
    gs = env.game_state
    walls = env.layout.walls
    px, py = gs.pacman_pos
    food = gs.food_mask
    H, W = food.shape

    if not food.any():
        return int(env.action_space.sample())

    visited = {(px, py)}
    queue: deque = deque()
    for dx, dy, a in [
        (0, -1, Action.UP), (0, 1, Action.DOWN),
        (-1, 0, Action.LEFT), (1, 0, Action.RIGHT),
    ]:
        nx, ny = px + dx, py + dy
        if 0 <= ny < H and 0 <= nx < W and not walls[ny, nx]:
            if food[ny, nx]:
                return int(a)
            visited.add((nx, ny))
            queue.append((nx, ny, int(a)))

    while queue:
        cx, cy, first = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if walls[ny, nx] or (nx, ny) in visited:
                continue
            if food[ny, nx]:
                return first
            visited.add((nx, ny))
            queue.append((nx, ny, first))

    return int(env.action_space.sample())


# --------------------------------------------------------------------------- #
class RandomPolicy:
    name = "random"

    def __call__(self, env: PacmanEnv, obs=None) -> int:
        return int(env.action_space.sample())


class GreedyBFSPolicy:
    name = "greedy"

    def __call__(self, env: PacmanEnv, obs=None) -> int:
        return greedy_nearest_pellet(env)


class EpsilonGreedy:
    """With probability `epsilon` act randomly, otherwise defer to `base`.

    Uses the env's own RNG (`env.np_random`) so collection is reproducible from
    the per-episode seed.
    """

    def __init__(self, base: Callable[[PacmanEnv], int], epsilon: float):
        self.base = base
        self.epsilon = float(epsilon)
        base_name = getattr(base, "name", base.__class__.__name__)
        self.name = f"{base_name}_eps{self.epsilon:g}"

    def __call__(self, env: PacmanEnv, obs=None) -> int:
        if env.np_random.random() < self.epsilon:
            return int(env.action_space.sample())
        return self.base(env, obs)


# --------------------------------------------------------------------------- #
def make_policy_pool(
    epsilons: Tuple[float, ...] = (0.05, 0.2, 0.5),
    weights: Tuple[float, ...] | None = None,
) -> Tuple[List[object], np.ndarray]:
    """Build the default mixed policy pool and its sampling weights.

    Returns `(policies, weights)` where `weights` sums to 1. The default mix is
    ~25% pure-random and ~75% greedy at several ε levels (spec §3.3). Each
    greedy-ε variant shares the 75% mass equally.
    """
    greedy = GreedyBFSPolicy()
    policies: List[object] = [RandomPolicy()]
    policies += [EpsilonGreedy(greedy, e) for e in epsilons]

    if weights is None:
        random_w = 0.25
        greedy_total = 1.0 - random_w
        per_greedy = greedy_total / len(epsilons)
        weights = np.array([random_w] + [per_greedy] * len(epsilons), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        assert len(weights) == len(policies), "weights must match policy count"

    weights = weights / weights.sum()
    return policies, weights


# --------------------------------------------------------------------------- #
# RL-agent policies (spec §3.3, updated): per-layout PPO agents collect the data.
#
# A world model trained on data from a *competent but imperfect* agent sees the
# states a real policy actually visits (reaching goals, dying, cornered states),
# which the hand-crafted greedy pool under-covers. To keep the dataset off the
# narrow optimal manifold, each layout mixes two checkpoints of its own agent:
#   * an OPTIMAL (fully trained) checkpoint, sampled 70% of episodes, and
#   * a SUB-OPTIMAL checkpoint (~half-trained), sampled 30%.
# Actions are deterministic (argmax); diversity comes from the checkpoint mix,
# randomized spawns, stochastic ghosts, and the 1–4 ghost-count sweep.
#
# The agent is the rsl_rl `MLPModel` actor (a plain MLP, obs_normalization off),
# so we reconstruct it in pure torch from the saved `actor_state_dict` — the data
# pipeline therefore has NO rsl_rl dependency (that lives only in training).
# --------------------------------------------------------------------------- #
import torch
import torch.nn as nn

from pacman_env import STATE_DIM

_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


def build_actor_mlp(hidden_dims, activation: str, in_dim: int = STATE_DIM,
                    n_actions: int = 5) -> nn.Module:
    """Reconstruct the rsl_rl `MLPModel` actor as a bare `nn.Sequential` named
    `mlp`, so a saved `actor_state_dict` (keys `mlp.0.weight`, ...) loads directly."""
    act = _ACTIVATIONS[activation.lower()]
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, int(h)), act()]
        d = int(h)
    layers += [nn.Linear(d, n_actions)]

    net = nn.Module()
    net.mlp = nn.Sequential(*layers)
    net.forward = net.mlp.forward          # type: ignore[assignment]
    return net


class RLCheckpointPolicy:
    """Deterministic (argmax) actor loaded from a slim PPO checkpoint.

    Accepts either a slim checkpoint written by `scripts/wm_train_rl_agents.py`
    (carries `actor_cfg`) or a raw rsl_rl checkpoint (`actor_state_dict` + an
    explicitly supplied `actor_cfg`). Pure torch — no rsl_rl import.
    """

    def __init__(self, checkpoint, actor_cfg: dict | None = None,
                 device: str = "cpu", stochastic: bool = False, name: str | None = None):
        ck = torch.load(str(checkpoint), map_location=device, weights_only=False)
        sd = ck["actor_state_dict"] if "actor_state_dict" in ck else ck.get("actor", ck)
        cfg = actor_cfg or ck.get("actor_cfg")
        if cfg is None:
            raise ValueError(f"{checkpoint}: actor_cfg missing; pass actor_cfg explicitly.")
        if cfg.get("obs_normalization", False):
            raise NotImplementedError("obs_normalization=true checkpoints are unsupported.")
        self.net = build_actor_mlp(cfg["hidden_dims"], cfg["activation"],
                                   int(cfg.get("in_dim", STATE_DIM)),
                                   int(cfg.get("n_actions", 5))).to(device)
        self.net.load_state_dict(sd)
        self.net.eval()
        self.device = device
        self.stochastic = bool(stochastic)
        from pathlib import Path as _P
        self.name = name or f"rl:{_P(str(checkpoint)).parent.name}/{_P(str(checkpoint)).stem}"
        self.iter = int(ck.get("iter", -1))

    @torch.no_grad()
    def __call__(self, env: PacmanEnv, obs=None) -> int:
        if obs is None:                                    # fall back to current env state
            obs = env._state_builder.build(env.game_state)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.net(x)[0]
        if self.stochastic:
            return int(torch.distributions.Categorical(logits=logits).sample().item())
        return int(torch.argmax(logits).item())


class LayoutAgentPool:
    """The optimal/sub-optimal checkpoint pair for ONE layout, with 70/30 sampling.

    Expects `<agents_root>/<layout_id>/{optimal.pt, suboptimal.pt}` as written by
    the training driver. Call `choose(rng)` per episode to draw a policy.
    """

    def __init__(self, agents_root, layout_id: str, device: str = "cpu",
                 optimal_weight: float = 0.7, stochastic: bool = False):
        from pathlib import Path as _P
        d = _P(agents_root) / layout_id
        opt, sub = d / "optimal.pt", d / "suboptimal.pt"
        if not opt.exists() or not sub.exists():
            raise FileNotFoundError(
                f"Missing agent checkpoints for '{layout_id}' under {d} "
                f"(need optimal.pt and suboptimal.pt). Run scripts/wm_train_rl_agents.py first.")
        self.layout_id = layout_id
        self.optimal = RLCheckpointPolicy(opt, device=device, stochastic=stochastic)
        self.suboptimal = RLCheckpointPolicy(sub, device=device, stochastic=stochastic)
        self.optimal_weight = float(optimal_weight)

    def choose(self, rng: np.random.Generator):
        """Return (policy, tag) — tag is 'optimal' or 'suboptimal' for stats."""
        if rng.random() < self.optimal_weight:
            return self.optimal, "optimal"
        return self.suboptimal, "suboptimal"
