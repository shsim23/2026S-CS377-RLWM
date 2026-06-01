"""Generate the train/test layout pools for world-model data collection.

Spec §3.2: the held-out *test* layouts are generated at the same time as the
training layouts and kept strictly separate — the test pool never contributes a
transition and exists only for later cross-layout generalization evaluation.
Generator seeds / layout IDs are persisted so the split is reproducible and
auditable.

Usage
-----
    python scripts/wm_generate_layouts.py --n-train 30 --n-test 5 --seed 0

    # smoke
    python scripts/wm_generate_layouts.py --n-train 4 --n-test 2 --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maze_generator import generate_maze
from maze_generator.constants import MAX_RETRIES
from world_model.data_pipeline.layouts import write_layout

# Seeds are spaced so a layout's internal retry range (seed..seed+MAX_RETRIES-1)
# never overlaps the next layout's range — keeps the pools reproducible & disjoint.
SEED_STRIDE = max(64, MAX_RETRIES * 8)


def _wall_hash(maze: dict) -> str:
    return hashlib.sha1(maze["walls"].tobytes()).hexdigest()[:16]


def _build_pool(name: str, n: int, seed_start: int, out_dir: Path,
                connectivity: float, num_ghosts: int) -> list[dict]:
    entries = []
    for i in range(n):
        seed = seed_start + i * SEED_STRIDE
        maze = generate_maze(seed=seed, connectivity=connectivity, num_ghosts=num_ghosts)
        layout_id = f"{name}_{i:03d}"
        fname = f"layout_{i:03d}.txt"
        write_layout(maze, out_dir / fname)
        entries.append({
            "layout_id": layout_id,
            "pool": name,
            "file": str((out_dir / fname).relative_to(ROOT)),
            "seed": seed,
            "food_count": int(maze["food_count"]),
            "wall_hash": _wall_hash(maze),
        })
    return entries


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="layouts/wm_pool")
    p.add_argument("--n-train", type=int, default=30)
    p.add_argument("--n-test", type=int, default=5)
    p.add_argument("--seed", type=int, default=0, help="Base seed for the train pool.")
    p.add_argument("--connectivity", type=float, default=0.3)
    p.add_argument("--num-ghosts", type=int, default=1)
    args = p.parse_args()

    out_root = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    train_dir = out_root / "train"
    test_dir = out_root / "test"

    # Train pool seeds occupy [seed, seed + n_train*STRIDE); the test pool starts
    # after that range so the two never share an internal retry seed.
    train_seed_start = args.seed
    test_seed_start = args.seed + (args.n_train + 1) * SEED_STRIDE

    train = _build_pool("train", args.n_train, train_seed_start, train_dir,
                        args.connectivity, args.num_ghosts)
    test = _build_pool("test", args.n_test, test_seed_start, test_dir,
                       args.connectivity, args.num_ghosts)

    # Audit: train and test wall layouts must be disjoint.
    train_hashes = {e["wall_hash"] for e in train}
    test_hashes = {e["wall_hash"] for e in test}
    overlap = train_hashes & test_hashes
    if overlap:
        raise RuntimeError(f"Train/test layout overlap detected (wall_hash): {overlap}")

    manifest = {
        "generator": "maze_generator.generate_maze",
        "seed_base": args.seed,
        "seed_stride": SEED_STRIDE,
        "connectivity": args.connectivity,
        "num_ghosts": args.num_ghosts,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "train": train,
        "test": test,
    }
    (out_root).mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(train)} train + {len(test)} test layouts to {out_root}")
    print(f"  train seeds: {[e['seed'] for e in train]}")
    print(f"  test  seeds: {[e['seed'] for e in test]}")
    print(f"  disjoint wall layouts: OK ({len(train_hashes)} train / {len(test_hashes)} test unique)")
    print(f"  manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
