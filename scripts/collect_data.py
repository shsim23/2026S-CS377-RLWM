"""Phase 1: collect transition data and save as per-episode NPZ files."""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_env import PacmanEnv, Action
from pacman_env.state import GameState


# ------------------------------------------------------------------ #
def greedy_nearest_pellet(env: PacmanEnv) -> int:
    """BFS from Pac-Man to nearest pellet; return first action index."""
    gs = env.game_state
    walls = env.layout.walls
    px, py = gs.pacman_pos
    food = gs.food_mask
    H, W = food.shape

    if not food.any():
        return int(env.action_space.sample())

    visited = {(px, py)}
    # queue: (x, y, first_action)
    queue = deque()
    for dx, dy, a in [(0,-1,Action.UP),(0,1,Action.DOWN),(-1,0,Action.LEFT),(1,0,Action.RIGHT)]:
        nx, ny = px + dx, py + dy
        if 0 <= ny < H and 0 <= nx < W and not walls[ny, nx]:
            if food[ny, nx]:
                return int(a)
            visited.add((nx, ny))
            queue.append((nx, ny, int(a)))

    while queue:
        cx, cy, first = queue.popleft()
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
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


def mixed_policy(env: PacmanEnv, p_greedy: float = 0.1) -> int:
    if env.np_random.random() < p_greedy:
        return greedy_nearest_pellet(env)
    return int(env.action_space.sample())


def make_policy(name: str, p_greedy: float):
    if name == "random":
        return lambda e: int(e.action_space.sample())
    if name == "mixed":
        return lambda e: mixed_policy(e, p_greedy=p_greedy)
    raise ValueError(f"Unknown policy: {name}")


# ------------------------------------------------------------------ #
def collect_episode(env: PacmanEnv, policy, seed: int):
    obs, info = env.reset(seed=seed)
    states, actions, rewards, dones = [obs], [], [], []

    while True:
        a = policy(env)
        obs, reward, terminated, truncated, info = env.step(a)
        actions.append(a)
        rewards.append(reward)
        done = terminated or truncated
        dones.append(done)
        if not done:
            states.append(obs)
        else:
            # Append final obs
            states.append(obs)
            break

    # Trim to equal length (T transitions → T states, T actions, T rewards, T dones)
    T = len(actions)
    return {
        "states":    np.array(states[:T], dtype=np.float32),
        "actions":   np.array(actions,     dtype=np.int64),
        "rewards":   np.array(rewards,     dtype=np.float32),
        "dones":     np.array(dones,       dtype=bool),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)
    parser.add_argument("--num-transitions", type=int, default=70000)
    parser.add_argument("--policy", default="mixed", choices=["random", "mixed"])
    parser.add_argument("--p-greedy", type=float, default=0.1,
                        help="Probability of greedy-toward-pellet action in mixed policy.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--randomize-spawn", action="store_true",
                        help="Randomize Pac-Man and ghost start positions on every reset "
                             "(uniform over walkable cells; min Manhattan distance enforced).")
    parser.add_argument("--min-spawn-dist", type=int, default=2,
                        help="Min Manhattan distance between Pac-Man and each ghost at spawn.")
    parser.add_argument("--num-ghosts", type=int, default=1)
    parser.add_argument("--ghost-epsilon", type=float, default=0.2)
    args = parser.parse_args()

    out = Path(args.output_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    env = PacmanEnv(
        layout_path=args.layout,
        num_ghosts=args.num_ghosts,
        ghost_epsilon=args.ghost_epsilon,
        randomize_spawn=args.randomize_spawn,
        min_spawn_dist=args.min_spawn_dist,
    )
    policy = make_policy(args.policy, p_greedy=args.p_greedy)

    episodes = []
    total = 0
    ep_idx = 0
    while total < args.num_transitions:
        ep = collect_episode(env, policy, seed=args.seed + ep_idx)
        episodes.append(ep)
        total += len(ep["actions"])
        ep_idx += 1
        if ep_idx % 50 == 0:
            print(f"  Collected {total}/{args.num_transitions} transitions ({ep_idx} episodes)")

    # Split train/val
    val_n = max(1, int(len(episodes) * args.val_fraction))
    train_eps = episodes[:-val_n]
    val_eps   = episodes[-val_n:]

    for i, ep in enumerate(train_eps):
        np.savez_compressed(out / "train" / f"episode_{i:06d}.npz", **ep)
    for i, ep in enumerate(val_eps):
        np.savez_compressed(out / "val" / f"episode_{i:06d}.npz", **ep)

    print(f"Saved {len(train_eps)} train + {len(val_eps)} val episodes to {out}")


if __name__ == "__main__":
    main()
