"""Standalone evaluation of a trained world model checkpoint."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model import EnsembleWorldModel, SequenceReplayBuffer, evaluate_k_step_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-dir", default="data/replay")
    parser.add_argument("--split", default="val")
    parser.add_argument("--k-step", type=int, default=10)
    parser.add_argument("--n-trajs", type=int, default=100)
    args = parser.parse_args()

    ensemble = EnsembleWorldModel.load(args.checkpoint)
    ensemble.eval()

    buffer = SequenceReplayBuffer(args.data_dir, split=args.split)
    metrics = evaluate_k_step_rollout(ensemble, buffer, K=args.k_step, N=args.n_trajs)

    print("\n=== World Model Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
