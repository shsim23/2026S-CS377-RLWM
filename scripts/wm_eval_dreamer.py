"""Standalone intrinsic evaluation of a DreamerV3 world-model checkpoint (spec §10).

Usage
-----
    # in-distribution (train-layout dataset)
    python scripts/wm_eval_dreamer.py --checkpoint checkpoints/dreamer_wm/main/best.pt \
        --dataset main

    # cross-layout generalization (collect a test-pool dataset first):
    #   python scripts/wm_collect_dataset.py --dataset test_eval --pool-split test --n-transitions 20000
    python scripts/wm_eval_dreamer.py --checkpoint checkpoints/dreamer_wm/main/best.pt \
        --dataset main --test-dataset test_eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from world_model.dreamer import DreamerWorldModel, WorldModelConfig, SequenceReplay, evaluate


def _print_metrics(title: str, m: dict) -> None:
    print(f"\n=== {title} ===")
    for k in sorted(m):
        if k.endswith("_curve"):
            continue
        print(f"  {k}: {m[k]:.5f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="In-distribution dataset name.")
    p.add_argument("--test-dataset", default=None, help="Held-out test-layout dataset (cross-layout).")
    p.add_argument("--data-root", default="data/replay")
    p.add_argument("--config", default="configs/world_model/dreamer_v3.yaml")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    cfg = yaml.safe_load(open(ROOT / args.config))["world_model"]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    wm_cfg = WorldModelConfig(**ckpt["cfg"])
    model = DreamerWorldModel(wm_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.checkpoint} (step {ckpt.get('step', '?')})")

    window = cfg["context"] + cfg["seq_length"]

    replay = SequenceReplay(str(ROOT / args.data_root / args.dataset), length=window, seed=0)
    m = evaluate(model, replay, context=cfg["context"], horizon=cfg["k_step"],
                 n_windows=cfg["n_eval_windows"], device=device, seed=cfg["eval_seed"])
    _print_metrics(f"In-distribution ({args.dataset})", m)

    if args.test_dataset:
        test_replay = SequenceReplay(str(ROOT / args.data_root / args.test_dataset),
                                     length=window, seed=0)
        mt = evaluate(model, test_replay, context=cfg["context"], horizon=cfg["k_step"],
                      n_windows=cfg["n_eval_windows"], device=device, seed=cfg["eval_seed"])
        _print_metrics(f"Cross-layout / held-out ({args.test_dataset})", mt)


if __name__ == "__main__":
    main()
