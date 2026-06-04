"""Collect the fixed, mixed-policy, multi-layout offline transition dataset.

Spec §3. Samples (layout, policy) pairs — layouts from the TRAIN pool only —
rolls out the simulator, and writes a single concatenated step-stream that the
DreamerV3 replay buffer (`world_model/dreamer/replay.py`) samples length-L
subsequences from, across episode boundaries, using the per-step `is_first`
flag to reset the recurrent state (spec §3.4 / §7).

Alignment (DreamerV3 convention): at stored index t we keep
    states[t]      = s_t           (observation)
    actions[t]     = a_{t-1}        (action that led into s_t; NOOP at first)
    rewards[t]     = r_{t-1}        (reward received entering s_t; 0 at first)
    continues[t]   = 1 - done_{t-1} (0 at a terminal state, else 1)
    is_first[t]    = 1 at the first step of each episode
The terminal state s_T IS stored, so each episode contributes (T+1) steps.

Outputs under data/replay/<dataset>/:
    states.npy, actions.npy, rewards.npy, continues.npy, is_first.npy,
    layout_ids.npy, manifest.json

Usage
-----
    python scripts/wm_collect_dataset.py --dataset main --n-transitions 500000
    python scripts/wm_collect_dataset.py --dataset smoke --n-transitions 5000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pacman_env import PacmanEnv
from pacman_env.constants import Action
from world_model.data_pipeline.policies import make_policy_pool, LayoutAgentPool


EVENT_KEYS = ["ate_pellet", "ate_power", "ate_ghosts", "died", "won"]


def collect_episode(env: PacmanEnv, policy, seed: int):
    """Roll out one episode; return aligned arrays + event tally."""
    obs, info = env.reset(seed=seed)
    states = [obs]
    actions = [int(Action.NOOP)]
    rewards = [0.0]
    continues = [1.0]
    is_first = [True]
    events = {k: 0 for k in EVENT_KEYS}

    while True:
        a = int(policy(env, states[-1]))     # states[-1] = current observation s_t
        obs, reward, terminated, truncated, info = env.step(a)
        done = bool(terminated or truncated)

        states.append(obs)
        actions.append(a)
        rewards.append(float(reward))
        continues.append(0.0 if done else 1.0)
        is_first.append(False)

        ev = info.get("event", {})
        events["ate_pellet"] += int(bool(ev.get("ate_pellet", False)))
        events["ate_power"] += int(bool(ev.get("ate_power", False)))
        events["ate_ghosts"] += int(ev.get("ate_ghosts", 0))
        events["died"] += int(bool(ev.get("died", False)))
        events["won"] += int(bool(ev.get("won", False)))

        if done:
            break

    ep = {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "continues": np.asarray(continues, dtype=np.float32),
        "is_first": np.asarray(is_first, dtype=bool),
    }
    return ep, events


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Dataset name under --out-root.")
    p.add_argument("--layout-pool", default="layouts/wm_pool",
                   help="Directory with manifest.json from wm_generate_layouts.py.")
    p.add_argument("--pool-split", default="train", choices=["train", "test"],
                   help="Which layout pool to sample from. 'train' for the training "
                        "dataset; 'test' ONLY to build a held-out cross-layout EVAL "
                        "dataset (never mix a test dataset into world-model training).")
    p.add_argument("--only-layouts", type=int, nargs="+", default=None,
                   help="Restrict collection to these pool indices (e.g. `0` for the "
                        "first split layout / train_000). Use to build a SINGLE-MAP "
                        "dataset. The stored layout_ids are the pool indices, so "
                        "SequenceReplay(--layout-id N) selects the same map.")
    p.add_argument("--out-root", default="data/replay")
    p.add_argument("--n-transitions", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-ghosts", type=int, default=1)
    p.add_argument("--ghost-choices", type=int, nargs="+", default=None,
                   help="If set, sample the ghost count per episode uniformly from "
                        "these values (e.g. 1 2 3 4) for ghost-count coverage. "
                        "Overrides --num-ghosts.")
    p.add_argument("--ghost-epsilon", type=float, default=0.2)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--min-spawn-dist", type=int, default=3)
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.05, 0.2, 0.5],
                   help="ε levels for the greedy policies (novice→expert spread).")
    p.add_argument("--random-fraction", type=float, default=0.25,
                   help="Sampling mass for the pure-random policy.")
    # --- policy source (spec §3.3) ---
    p.add_argument("--policy-source", choices=["heuristic", "rl"], default="heuristic",
                   help="'heuristic': random+greedy-BFS pool (default). 'rl': per-layout "
                        "PPO agents (optimal/sub-optimal checkpoint mix).")
    p.add_argument("--rl-agents-root", default="checkpoints/rl_agents",
                   help="Root with <layout_id>/{optimal,suboptimal}.pt (wm_train_rl_agents.py).")
    p.add_argument("--rl-optimal-weight", type=float, default=0.7,
                   help="Per-episode probability of using the OPTIMAL checkpoint (else sub-optimal).")
    p.add_argument("--rl-stochastic", action="store_true",
                   help="Sample actions from the policy instead of argmax (off by default).")
    p.add_argument("--rl-allow-partial", action="store_true",
                   help="Collect from ONLY the layouts that already have a trained agent "
                        "(skip the rest with a warning) instead of requiring all of them.")
    p.add_argument("--device", default="cpu", help="Torch device for RL inference.")
    args = p.parse_args()

    pool_dir = (ROOT / args.layout_pool) if not Path(args.layout_pool).is_absolute() else Path(args.layout_pool)
    manifest_path = pool_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"Layout manifest not found: {manifest_path}. Run wm_generate_layouts.py first.")
    pool = json.loads(manifest_path.read_text())
    train_layouts = pool[args.pool_split]
    if not train_layouts:
        sys.exit(f"Layout pool '{args.pool_split}' is empty.")

    # Optional single-/few-map restriction (pool indices).
    only_layouts = None
    if args.only_layouts is not None:
        only_layouts = sorted(set(args.only_layouts))
        bad = [i for i in only_layouts if not (0 <= i < len(train_layouts))]
        if bad:
            sys.exit(f"--only-layouts {bad} out of range (pool '{args.pool_split}' "
                     f"has {len(train_layouts)} layouts).")
        print(f"[only-layouts] Restricting collection to pool indices {only_layouts} "
              f"({[train_layouts[i].get('layout_id', i) for i in only_layouts]}).")
    if args.pool_split == "test":
        print("[WARNING] Collecting from the TEST layout pool. Use this dataset ONLY "
              "for cross-layout evaluation, never for world-model training.")

    out_dir = (ROOT / args.out_root / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    ghost_choices = args.ghost_choices if args.ghost_choices else [args.num_ghosts]

    def layout_id_of(idx: int) -> str:
        return train_layouts[idx].get("layout_id", f"{args.pool_split}_{idx:03d}")

    # --- policy source ---
    use_rl = args.policy_source == "rl"
    if use_rl:
        from pathlib import Path as _Path

        def _has_agent(layout_idx: int) -> bool:
            d = _Path(args.rl_agents_root) / layout_id_of(layout_idx)
            return (d / "optimal.pt").exists() and (d / "suboptimal.pt").exists()

        # Which layouts to sample from.
        all_idx = list(range(len(train_layouts))) if only_layouts is None else list(only_layouts)
        if args.rl_allow_partial:
            rl_layout_indices = [i for i in all_idx if _has_agent(i)]
            skipped = [layout_id_of(i) for i in all_idx if i not in rl_layout_indices]
            if not rl_layout_indices:
                sys.exit(f"No trained agents found under {args.rl_agents_root}.")
            if skipped:
                print(f"[partial] Using {len(rl_layout_indices)}/{len(all_idx)} layouts with "
                      f"agents; skipping {len(skipped)} without: {skipped}")
        else:
            rl_layout_indices = all_idx
            missing = [layout_id_of(i) for i in rl_layout_indices if not _has_agent(i)]
            if missing:
                sys.exit(f"Missing trained agents for {missing} under {args.rl_agents_root} "
                         f"(use --rl-allow-partial to skip).")

        # Lazily build the per-layout optimal/sub-optimal agent pool (cached).
        agent_pools: dict[int, LayoutAgentPool] = {}

        def get_agent_pool(layout_idx: int) -> LayoutAgentPool:
            if layout_idx not in agent_pools:
                agent_pools[layout_idx] = LayoutAgentPool(
                    args.rl_agents_root, layout_id_of(layout_idx), device=args.device,
                    optimal_weight=args.rl_optimal_weight, stochastic=args.rl_stochastic)
            return agent_pools[layout_idx]

        policies, weights = [], None
        for li in rl_layout_indices:          # fail fast on the layouts we will use
            get_agent_pool(li)
        print(f"Loaded RL agents for {len(rl_layout_indices)} layouts from {args.rl_agents_root} "
              f"(optimal {args.rl_optimal_weight:.0%} / sub-optimal {1 - args.rl_optimal_weight:.0%}, "
              f"{'stochastic' if args.rl_stochastic else 'argmax'}).")
    else:
        # Heuristic pool with explicit random mass.
        n_greedy = len(args.epsilons)
        greedy_each = (1.0 - args.random_fraction) / max(n_greedy, 1)
        weights = np.array([args.random_fraction] + [greedy_each] * n_greedy, dtype=np.float64)
        policies, weights = make_policy_pool(tuple(args.epsilons), tuple(weights))

    # Cache one env per (layout, ghost-count) — layout parsing is the expensive part.
    env_cache: dict[tuple[int, int], PacmanEnv] = {}

    def get_env(layout_idx: int, num_ghosts: int) -> PacmanEnv:
        key = (layout_idx, num_ghosts)
        if key not in env_cache:
            lay = train_layouts[layout_idx]
            env_cache[key] = PacmanEnv(
                layout_path=str(ROOT / lay["file"]),
                num_ghosts=num_ghosts,
                ghost_epsilon=args.ghost_epsilon,
                randomize_spawn=True,
                min_spawn_dist=args.min_spawn_dist,
                max_steps=args.max_steps,
            )
        return env_cache[key]

    all_states, all_actions, all_rewards = [], [], []
    all_continues, all_is_first, all_layout_ids, all_ghost_counts = [], [], [], []
    total_events = {k: 0 for k in EVENT_KEYS}
    from collections import defaultdict
    policy_counts: dict[str, int] = defaultdict(int)
    rl_tag_counts = {"optimal": 0, "suboptimal": 0}
    ghost_count_eps = {g: 0 for g in ghost_choices}
    total = 0
    ep_idx = 0
    n_episodes = 0

    while total < args.n_transitions:
        if use_rl:
            layout_idx = int(rng.choice(rl_layout_indices))
            policy, tag = get_agent_pool(layout_idx).choose(rng)
            rl_tag_counts[tag] += 1
        else:
            layout_idx = (int(rng.integers(len(train_layouts))) if only_layouts is None
                          else int(rng.choice(only_layouts)))
            policy = policies[int(rng.choice(len(policies), p=weights))]
        num_ghosts = int(rng.choice(ghost_choices))
        env = get_env(layout_idx, num_ghosts)

        ep, events = collect_episode(env, policy, seed=args.seed + ep_idx)
        ep_idx += 1
        n_episodes += 1
        n = len(ep["states"])

        all_states.append(ep["states"])
        all_actions.append(ep["actions"])
        all_rewards.append(ep["rewards"])
        all_continues.append(ep["continues"])
        all_is_first.append(ep["is_first"])
        all_layout_ids.append(np.full(n, layout_idx, dtype=np.int32))
        all_ghost_counts.append(np.full(n, num_ghosts, dtype=np.int8))

        for k in EVENT_KEYS:
            total_events[k] += events[k]
        policy_counts[policy.name] += 1
        ghost_count_eps[num_ghosts] += 1
        total += n - 1  # transitions, not states

        if n_episodes % 200 == 0:
            print(f"  {total}/{args.n_transitions} transitions ({n_episodes} eps)")

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    continues = np.concatenate(all_continues, axis=0)
    is_first = np.concatenate(all_is_first, axis=0)
    layout_ids = np.concatenate(all_layout_ids, axis=0)
    ghost_counts = np.concatenate(all_ghost_counts, axis=0)

    np.save(out_dir / "states.npy", states)
    np.save(out_dir / "actions.npy", actions)
    np.save(out_dir / "rewards.npy", rewards)
    np.save(out_dir / "continues.npy", continues)
    np.save(out_dir / "is_first.npy", is_first)
    np.save(out_dir / "layout_ids.npy", layout_ids)
    np.save(out_dir / "ghost_counts.npy", ghost_counts)

    # Reward-event coverage tally (spec §3.5). Rare events that never appear
    # cannot be learned by the two-hot reward head.
    n_steps = len(states)
    # "step_only" = transitions whose reward is just the step penalty (no event).
    # is_first steps carry a placeholder reward of 0 and are excluded.
    step_penalty = -0.01  # pacman_env.reward.RewardConfig default (not overridden here)
    step_only = int((~is_first & np.isclose(rewards, step_penalty)).sum())
    coverage = {
        "pellet": total_events["ate_pellet"],
        "power_pellet": total_events["ate_power"],
        "ghost_eat": total_events["ate_ghosts"],
        "death": total_events["died"],
        "win": total_events["won"],
        "step_only": step_only,
    }
    reward_stats = {
        "min": float(rewards.min()), "max": float(rewards.max()),
        "mean": float(rewards.mean()), "nonzero_frac": float((rewards != 0).mean()),
    }

    manifest = {
        "dataset": args.dataset,
        "layout_pool": str(pool_dir.relative_to(ROOT)) if pool_dir.is_relative_to(ROOT) else str(pool_dir),
        "pool_split": args.pool_split,
        "layout_split": {"train_n": pool.get("n_train", 0), "test_n": pool.get("n_test", 0),
                         "n_layouts_used": len(train_layouts)},
        "n_transitions": int(total),
        "n_steps_stored": int(n_steps),
        "n_episodes": int(n_episodes),
        "ghost_choices": list(ghost_choices),
        "ghost_count_episodes": {str(g): c for g, c in ghost_count_eps.items()},
        "ghost_epsilon": args.ghost_epsilon,
        "max_steps": args.max_steps,
        "policy_mix": (
            {
                "source": "rl",
                "rl_agents_root": args.rl_agents_root,
                "optimal_weight": args.rl_optimal_weight,
                "stochastic": bool(args.rl_stochastic),
                "checkpoint_episode_counts": dict(rl_tag_counts),
                "episode_counts": dict(policy_counts),
            }
            if use_rl else
            {
                "source": "heuristic",
                "epsilons": list(args.epsilons),
                "random_fraction": args.random_fraction,
                "weights": {pol.name: float(w) for pol, w in zip(policies, weights)},
                "episode_counts": dict(policy_counts),
            }
        ),
        "reward_event_coverage": coverage,
        "reward_stats": reward_stats,
        "seed": args.seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSaved dataset '{args.dataset}': {total} transitions "
          f"({n_steps} stored steps, {n_episodes} episodes) -> {out_dir}")
    print("Reward-event coverage:")
    for k, v in coverage.items():
        print(f"  {k:14s}: {v}")
    print(f"Reward stats: {reward_stats}")
    print(f"Policy episode counts: {dict(policy_counts)}")
    if use_rl:
        print(f"RL checkpoint episodes: {rl_tag_counts}")
    print(f"Ghost-count episodes: {ghost_count_eps}")


if __name__ == "__main__":
    main()
