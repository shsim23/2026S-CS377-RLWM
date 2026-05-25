from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


class SequenceReplayBuffer:
    """Episode-per-file NPZ replay buffer with sequence sampling."""

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
        rng = np.random.RandomState(bootstrap_seed) if bootstrap_seed is not None else np.random
        chosen_files = rng.choice(self.episode_files, size=batch_size, replace=True)

        states_b, actions_b, rewards_b, dones_b = [], [], [], []
        for path in chosen_files:
            ep = np.load(path)
            T = len(ep["states"])
            if T < seq_length:
                pad = seq_length - T
                s = np.concatenate([ep["states"],  np.tile(ep["states"][-1:], (pad, 1))], axis=0)
                a = np.concatenate([ep["actions"], np.full(pad, 4, dtype=np.int64)], axis=0)
                r = np.concatenate([ep["rewards"], np.zeros(pad, dtype=np.float32)], axis=0)
                d = np.concatenate([ep["dones"],   np.ones(pad, dtype=bool)], axis=0)
            else:
                start = rng.randint(0, T - seq_length + 1)
                s = ep["states"] [start: start + seq_length]
                a = ep["actions"][start: start + seq_length]
                r = ep["rewards"][start: start + seq_length]
                d = ep["dones"]  [start: start + seq_length]
            states_b.append(s); actions_b.append(a); rewards_b.append(r); dones_b.append(d)

        return {
            "states":  torch.from_numpy(np.stack(states_b)).float(),
            "actions": torch.from_numpy(np.stack(actions_b)).long(),
            "rewards": torch.from_numpy(np.stack(rewards_b)).float(),
            "dones":   torch.from_numpy(np.stack(dones_b)),
        }

    def sample_trajectories(self, N: int, seed: Optional[int] = None) -> List[dict]:
        rng = np.random.RandomState(seed) if seed is not None else np.random
        chosen = rng.choice(
            self.episode_files, size=min(N, len(self.episode_files)), replace=False
        )
        trajs = []
        for path in chosen:
            ep = np.load(path)
            trajs.append({k: ep[k] for k in ep.keys()})
        return trajs
