"""Sequence replay over the concatenated offline step-stream (spec §7).

The dataset (written by `scripts/wm_collect_dataset.py`) is a single contiguous
stream of steps with a per-step `is_first` flag marking episode boundaries.
Sampling draws length-L windows uniformly over ALL valid start positions,
ignoring episode boundaries (DreamerV3 convention): a window may span a boundary,
and `is_first` lets the model reset its recurrent state mid-window. The leading
`context` steps of each window warm up h_0 from real history and are excluded
from the loss by the training loop.

`states.npy` is memory-mapped so large (1–2M step) datasets need not fit in RAM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


class SequenceReplay:
    def __init__(self, dataset_dir: str, length: int, seed: Optional[int] = None):
        self.dir = Path(dataset_dir)
        self.length = int(length)
        self.states = np.load(self.dir / "states.npy", mmap_mode="r")     # (N, 901)
        self.actions = np.load(self.dir / "actions.npy")
        self.rewards = np.load(self.dir / "rewards.npy")
        self.continues = np.load(self.dir / "continues.npy")
        self.is_first = np.load(self.dir / "is_first.npy")
        self.layout_ids = np.load(self.dir / "layout_ids.npy")
        self.N = self.states.shape[0]
        if self.N < self.length:
            raise ValueError(f"Dataset has {self.N} steps < window length {self.length}")
        self.max_start = self.N - self.length
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.N

    def sample_batch(self, batch_size: int, device=None) -> dict:
        starts = self.rng.integers(0, self.max_start + 1, size=batch_size)
        L = self.length
        idx = starts[:, None] + np.arange(L)[None, :]      # (B, L)

        states = np.asarray(self.states)[idx]              # (B, L, 901)
        out = {
            "states": torch.from_numpy(states).float(),
            "actions": torch.from_numpy(self.actions[idx]).long(),
            "rewards": torch.from_numpy(self.rewards[idx]).float(),
            "continues": torch.from_numpy(self.continues[idx]).float(),
            "is_first": torch.from_numpy(self.is_first[idx]),
        }
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out

    def iter_eval_windows(self, n_windows: int, device=None, seed: int = 0):
        """Deterministic set of windows for evaluation (fixed across calls)."""
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, self.max_start + 1, size=n_windows)
        L = self.length
        for s in starts:
            idx = np.arange(s, s + L)
            yield {
                "states": torch.from_numpy(np.asarray(self.states)[idx]).float(),
                "actions": torch.from_numpy(self.actions[idx]).long(),
                "rewards": torch.from_numpy(self.rewards[idx]).float(),
                "continues": torch.from_numpy(self.continues[idx]).float(),
                "is_first": torch.from_numpy(self.is_first[idx]),
            }
