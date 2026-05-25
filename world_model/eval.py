from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .ensemble import EnsembleWorldModel
    from .replay_buffer import SequenceReplayBuffer


@torch.no_grad()
def evaluate_k_step_rollout(
    ensemble: "EnsembleWorldModel",
    val_buffer: "SequenceReplayBuffer",
    K: int = 10,
    N: int = 100,
    seed: int = 0,
) -> dict:
    latent_errs, reward_errs, done_errs, sigma_values = [], [], [], []

    device = next(ensemble.parameters()).device

    for traj in val_buffer.sample_trajectories(N, seed=seed):
        states  = torch.from_numpy(traj["states"]).float().to(device)
        actions = torch.from_numpy(traj["actions"]).long().to(device)
        rewards = traj["rewards"]
        dones   = traj["dones"]
        T = len(states)
        if T < K + 1:
            continue

        z, h = ensemble.encode(states[0:1])

        for t in range(K):
            a = actions[t: t + 1]
            out = ensemble.imagine_step(z, h, a)

            z_true, _ = ensemble.encode(states[t + 1: t + 2])
            latent_errs.append(((out["z_next"] - z_true) ** 2).mean().item())
            reward_errs.append((out["reward"].item() - float(rewards[t])) ** 2)
            done_errs.append(abs(out["done"].item() - float(dones[t])))
            sigma_values.append(out["sigma"].item())

            if out["done"].item() > 0.5:
                break

            z, h = out["z_next"], out["h_next"]

    return {
        "k_step_latent_mse": float(np.mean(latent_errs))   if latent_errs  else float("inf"),
        "k_step_reward_mse": float(np.mean(reward_errs))   if reward_errs  else float("inf"),
        "k_step_done_err":   float(np.mean(done_errs))     if done_errs    else float("inf"),
        "sigma_mean":        float(np.mean(sigma_values))  if sigma_values else 0.0,
    }
