"""Train one PPO agent per TRAIN-pool layout, for RL-driven data collection (spec §3.3).

For each layout in the world-model TRAIN layout pool this trains a dedicated PPO
agent (rsl_rl, same architecture as `pacman_rl`), then exports two slim,
rsl_rl-free checkpoints used by `scripts/wm_collect_dataset.py --policy-source rl`:

    checkpoints/rl_agents/<layout_id>/optimal.pt      # fully-trained agent
    checkpoints/rl_agents/<layout_id>/suboptimal.pt   # ~half-trained agent

The collector samples the OPTIMAL agent 70% of episodes and the SUB-OPTIMAL one
30%, so the dataset stays off the narrow optimal manifold (the user's requirement:
the collection agent is intentionally not always optimal).

Each slim file holds only the actor weights + an `actor_cfg`, so data collection
loads them in pure torch with no rsl_rl dependency.

Usage
-----
    # all 30 train-pool layouts, 500 iterations each
    python scripts/wm_train_rl_agents.py --iterations 500 --device cuda --headless

    # smoke: first 2 layouts, few iterations
    python scripts/wm_train_rl_agents.py --max-layouts 2 --iterations 8 --num-envs 8 --headless
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsl_rl.runners import OnPolicyRunner

from pacman_env import STATE_DIM
from pacman_rl.config import load_yaml, make_env_cfg, make_train_cfg, resolve_path
from pacman_rl.vec_env import RslPacmanVecEnv


def actor_cfg_from_config(config: dict) -> dict:
    """Extract the bare actor-MLP spec so the slim checkpoint is self-describing."""
    actor = config["train"]["actor"]
    dist = actor.get("distribution_cfg", {})
    return {
        "hidden_dims": list(actor["hidden_dims"]),
        "activation": str(actor["activation"]),
        "obs_normalization": bool(actor.get("obs_normalization", False)),
        "in_dim": STATE_DIM,
        "n_actions": int(dist.get("num_categories", 5)),
    }


def _iter_of(path: Path) -> int:
    m = re.search(r"model_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1


def _select_checkpoints(ckpt_dir: Path) -> tuple[Path, Path]:
    """Return (optimal, suboptimal) = (highest-iter, closest-to-half-iter) checkpoints."""
    models = sorted((p for p in ckpt_dir.glob("model_*.pt") if _iter_of(p) >= 0),
                    key=_iter_of)
    if not models:
        raise FileNotFoundError(f"No model_*.pt checkpoints under {ckpt_dir}")
    optimal = models[-1]
    half_target = _iter_of(optimal) // 2
    # closest to half, preferring a distinct checkpoint from the optimal one
    candidates = [m for m in models if m != optimal] or models
    suboptimal = min(candidates, key=lambda m: abs(_iter_of(m) - half_target))
    return optimal, suboptimal


def _export_slim(model_pt: Path, out_pt: Path, layout_id: str, actor_cfg: dict) -> int:
    ck = torch.load(str(model_pt), map_location="cpu", weights_only=False)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor_state_dict": ck["actor_state_dict"],
        "actor_cfg": actor_cfg,
        "iter": int(ck.get("iter", _iter_of(model_pt))),
        "layout_id": layout_id,
        "source_checkpoint": str(model_pt),
    }, str(out_pt))
    return int(ck.get("iter", _iter_of(model_pt)))


def _redirect_save(runner: OnPolicyRunner, checkpoint_dir: Path) -> None:
    """Mirror pacman_rl/train.py: write rsl_rl checkpoints into checkpoint_dir."""
    original_save = runner.save

    def save_to_dir(path: str, infos=None) -> None:
        original_save(str(checkpoint_dir / Path(path).name), infos=infos)

    runner.save = save_to_dir


def train_one_layout(layout_id: str, layout_file: str, config: dict, args,
                     agents_root: Path, runs_root: Path) -> dict:
    iterations = int(args.iterations)
    env_cfg = make_env_cfg(config, layout_file=layout_file,
                           num_envs=args.num_envs, seed=args.seed)
    env_cfg.setdefault("ghost", {})
    if args.num_ghosts is not None:
        env_cfg["ghost"]["num_ghosts"] = int(args.num_ghosts)

    train_cfg = make_train_cfg(config, run_name=layout_id)
    train_cfg["save_interval"] = max(1, args.save_interval or iterations // 10)

    train_dir = runs_root / layout_id / "train"
    ckpt_dir = train_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or config.get("device", "cuda")
    env = RslPacmanVecEnv(env_cfg, device=device)
    try:
        runner = OnPolicyRunner(env, train_cfg, log_dir=str(train_dir), device=device)
        _redirect_save(runner, ckpt_dir)
        runner.learn(num_learning_iterations=iterations)
    finally:
        env.close()

    actor_cfg = actor_cfg_from_config(config)
    optimal_pt, sub_pt = _select_checkpoints(ckpt_dir)
    out_dir = agents_root / layout_id
    opt_iter = _export_slim(optimal_pt, out_dir / "optimal.pt", layout_id, actor_cfg)
    sub_iter = _export_slim(sub_pt, out_dir / "suboptimal.pt", layout_id, actor_cfg)
    print(f"  [{layout_id}] exported optimal(iter={opt_iter}) + suboptimal(iter={sub_iter}) "
          f"-> {out_dir}")
    return {"layout_id": layout_id, "layout_file": layout_file,
            "optimal_iter": opt_iter, "suboptimal_iter": sub_iter,
            "iterations": iterations, "num_ghosts": args.num_ghosts}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pacman_rl/configs/pacman_ppo.yaml")
    p.add_argument("--layout-pool", default="layouts/wm_pool")
    p.add_argument("--pool-split", default="train", choices=["train", "test"])
    p.add_argument("--agents-root", default="checkpoints/rl_agents",
                   help="Where slim per-layout agents are written.")
    p.add_argument("--runs-root", default="logs/pacman_rl/wm_agents",
                   help="Where full rsl_rl runs (checkpoints, tensorboard) are written.")
    p.add_argument("--iterations", type=int, default=None,
                   help="PPO iterations per layout (default: config.iterations).")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-ghosts", type=int, default=None,
                   help="Ghost count for agent training (default: config ghost.num_ghosts).")
    p.add_argument("--save-interval", type=int, default=None,
                   help="rsl_rl checkpoint interval (default: iterations//10).")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--layouts", type=int, nargs="+", default=None,
                   help="Only train these layout indices (default: all).")
    p.add_argument("--max-layouts", type=int, default=None,
                   help="Train only the first N layouts (smoke).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip layouts that already have optimal.pt + suboptimal.pt.")
    args = p.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = load_yaml(args.config)
    if args.iterations is None:
        args.iterations = int(config.get("iterations", 500))

    pool = json.loads((resolve_path(args.layout_pool) / "manifest.json").read_text())
    layouts = pool[args.pool_split]
    if not layouts:
        sys.exit(f"Layout pool '{args.pool_split}' is empty.")

    indices = list(range(len(layouts)))
    if args.layouts is not None:
        indices = [i for i in args.layouts if 0 <= i < len(layouts)]
    if args.max_layouts is not None:
        indices = indices[: args.max_layouts]

    agents_root = resolve_path(args.agents_root)
    runs_root = resolve_path(args.runs_root)
    print(f"Training {len(indices)} agent(s) | iterations={args.iterations} "
          f"num_envs={args.num_envs} ghosts={args.num_ghosts or 'config'} -> {agents_root}")

    summary = []
    for n, i in enumerate(indices):
        entry = layouts[i]
        layout_id = entry.get("layout_id", f"{args.pool_split}_{i:03d}")
        layout_file = str(resolve_path(entry["file"]))
        out_dir = agents_root / layout_id
        if args.skip_existing and (out_dir / "optimal.pt").exists() and (out_dir / "suboptimal.pt").exists():
            print(f"[{n + 1}/{len(indices)}] {layout_id}: skip (exists)")
            continue
        print(f"[{n + 1}/{len(indices)}] training {layout_id} ({entry['file']})")
        summary.append(train_one_layout(layout_id, layout_file, config, args, agents_root, runs_root))

    manifest_path = agents_root / "agents_manifest.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"agents": []}
    by_id = {a["layout_id"]: a for a in prior.get("agents", [])}
    for a in summary:
        by_id[a["layout_id"]] = a
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "config": args.config, "pool_split": args.pool_split,
        "optimal_weight_at_collection": 0.7,
        "agents": sorted(by_id.values(), key=lambda a: a["layout_id"]),
    }, indent=2))
    print(f"\nDone. {len(summary)} agent(s) trained. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
