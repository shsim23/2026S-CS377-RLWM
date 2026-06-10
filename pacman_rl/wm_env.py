from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from pacman_env import PacmanEnv, RewardConfig, STATE_DIM
from pacman_env.constants import Action
from world_model.dreamer import DreamerWorldModel, WorldModelConfig
from world_model.dreamer.nn import OneHotCategoricalST
from world_model.dreamer.rssm import FOOD_SLICE, PAC_SLICE, WALL_SLICE


NOOP_ACTION = int(Action.NOOP)


from pacman_rl.wm_rewards import DecodedStateRewardComputer
from pacman_rl.wm_uncertainty import (
    SelfEnsembleStats,
    RunningMean,
    component_weighted_decoded_state_variance,
    self_ensemble_stats,
)


class RslPacmanDreamerVecEnv(VecEnv):
    """rsl_rl VecEnv that trains PPO from Dreamer imagined transitions."""

    def __init__(self, env_cfg: dict[str, Any], wm_cfg: dict[str, Any], device: str = "cpu") -> None:
        self.cfg = dict(env_cfg)
        self.wm_cfg = dict(wm_cfg)
        self.device = torch.device(device)
        self.num_envs = int(self.cfg.get("num_envs", 16))
        self.num_actions = 1
        self.action_space_n = int(self.cfg.get("action_space_n", 5))
        self.max_episode_length = int(self.cfg.get("max_steps", self.cfg.get("max_episode_length", 500)))
        self.step_dt = 1.0
        self.unwrapped = self

        self.use_uncertainty = bool(self.wm_cfg.get("use_uncertainty_aware_methods", False))
        self.confidence_alpha = float(self.wm_cfg.get("confidence_alpha", 0.5))
        self.confidence_weight_scale = float(self.wm_cfg.get("confidence_weight_scale", 2.0))
        self.self_ensemble_inferences = max(1, int(self.wm_cfg.get("self_ensemble_inferences", 5)))
        self.self_ensemble_threshold = float(self.wm_cfg.get("self_ensemble_threshold", 2.0))
        self.self_ensemble_component_weights = self._self_ensemble_component_weights(self.wm_cfg)
        self.continue_threshold = float(self.wm_cfg.get("continue_threshold", 0.5))
        self.reward_source = str(self.wm_cfg.get("reward_source", "reward_head"))
        if self.reward_source not in {"reward_head", "decoded_rules"}:
            raise ValueError("world_model.reward_source must be one of: reward_head, decoded_rules")
        self.log_reward_sources = bool(self.wm_cfg.get("log_reward_sources", True))
        momentum = float(self.wm_cfg.get("running_mean_momentum", 0.99))
        self.uncertainty_mean = RunningMean(momentum=momentum, device=self.device)
        self.decoded_reward_computer = DecodedStateRewardComputer(self.cfg)

        self.envs = [self._make_seed_env() for _ in range(self.num_envs)]
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._obs = torch.zeros(self.num_envs, STATE_DIM, dtype=torch.float32, device=self.device)

        self.model = self._load_model().to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._h, self._z = self.model.initial_state(self.num_envs, self.device)
        self._e = torch.zeros(self.num_envs, self.model.cfg.e_dim, dtype=torch.float32, device=self.device)
        self._wall = torch.zeros(self.num_envs, 441, dtype=torch.float32, device=self.device)
        self._initial_power = torch.zeros(self.num_envs, 441, dtype=torch.float32, device=self.device)
        self._total_pellets = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        base_seed = int(self.cfg.get("seed", 0))
        for i in range(self.num_envs):
            self._reset_one(i, seed=base_seed + i)

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self._obs.clone()}, batch_size=[self.num_envs], device=self.device)

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_ids = actions.detach().to(self.device).long().view(self.num_envs, -1)[:, 0].clamp(0, self.action_space_n - 1)
        prev_obs = self._obs.clone()

        h_next = self.model.seq(self._h, self.model._flat(self._z), action_ids, self._e)
        prior_logits = self.model.prior(h_next)
        dist = OneHotCategoricalST(prior_logits, self.model.cfg.unimix)
        deterministic_latent = bool(self.wm_cfg.get("deterministic_latent", False))
        ensemble_inferences = self.self_ensemble_inferences if self.use_uncertainty else 1
        z_samples = []
        pred_obs_samples = []
        for _ in range(ensemble_inferences):
            z_sample = dist.mode() if deterministic_latent else dist.sample_st()
            z_samples.append(z_sample)
            pred_dyn_sample = self.model.decode_state(h_next, self.model._flat(z_sample))
            pred_obs_samples.append(self._full_state(pred_dyn_sample))

        z_next = z_samples[0]
        z_flat = self.model._flat(z_next)
        reward_head = self.model.reward_from_logits(self.model.reward_head(h_next, z_flat)).float()
        cont_logits = self.model.cont_head(h_next, z_flat)
        cont = torch.sigmoid(cont_logits)
        pred_obs = pred_obs_samples[0]

        uncertainty = self._self_ensemble_uncertainty(torch.stack(pred_obs_samples, dim=0))
        wm_done = cont < self.continue_threshold
        timeout = self.episode_length_buf + 1 >= self.max_episode_length
        base_dones = wm_done | timeout
        if self.use_uncertainty and bool(self.wm_cfg.get("adaptive_rollout_truncation", True)):
            base_dones = base_dones | uncertainty.truncate

        decoded_reward = self.decoded_reward_computer.compute(
            prev_obs,
            pred_obs,
            episode_ended=base_dones,
            total_pellets=self._total_pellets,
            initial_power=self._initial_power,
        )
        reward = decoded_reward.reward if self.reward_source == "decoded_rules" else reward_head
        dones = base_dones
        if self.reward_source == "decoded_rules":
            dones = dones | decoded_reward.win.bool() | decoded_reward.death.bool()

        self._episode_returns += reward
        self.episode_length_buf += 1
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_wins: list[float] = []
        completed_deaths: list[float] = []
        completed_timeouts: list[float] = []
        completed_pellets_remaining: list[float] = []

        self._obs = pred_obs
        self._h = h_next
        self._z = z_next

        pellets_remaining = decoded_reward.remaining_pellets
        for i in torch.nonzero(dones, as_tuple=False).flatten().tolist():
            completed_returns.append(float(self._episode_returns[i].item()))
            completed_lengths.append(int(self.episode_length_buf[i].item()))
            completed_wins.append(float(decoded_reward.win[i].item()))
            completed_deaths.append(float(decoded_reward.death[i].item()))
            completed_timeouts.append(float(timeout[i].item()))
            completed_pellets_remaining.append(float(pellets_remaining[i].item()))
            self._reset_one(i)

        extras: dict[str, Any] = {
            "time_outs": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "wm_confidence": uncertainty.confidence,
            "log": {
                "/wm/self_ensemble_uncertainty": uncertainty.uncertainty,
                "/wm/self_ensemble_uncertainty_norm": uncertainty.uncertainty_norm,
                "/wm/confidence": uncertainty.confidence,
                "/wm/self_ensemble_truncation_rate": uncertainty.truncate.float(),
                "/pacman/win_rate": decoded_reward.win,
                "/pacman/death_rate": decoded_reward.death,
                "/pacman/timeout_failure_rate": timeout.float(),
                "/pacman/pellets_remaining": pellets_remaining,
            },
        }
        if self.log_reward_sources:
            extras["log"].update({
                "/wm/reward_head": reward_head,
                "/wm/reward_decoded_rules": decoded_reward.reward,
                "/wm/reward_delta_head_minus_rules": reward_head - decoded_reward.reward,
                "/wm/reward_source_is_decoded": torch.full(
                    (self.num_envs,), float(self.reward_source == "decoded_rules"), dtype=torch.float32, device=self.device
                ),
            })
        if completed_returns:
            extras["episode"] = {
                "return": torch.tensor(completed_returns, dtype=torch.float32, device=self.device),
                "length": torch.tensor(completed_lengths, dtype=torch.float32, device=self.device),
                "win": torch.tensor(completed_wins, dtype=torch.float32, device=self.device),
                "death": torch.tensor(completed_deaths, dtype=torch.float32, device=self.device),
                "timeout": torch.tensor(completed_timeouts, dtype=torch.float32, device=self.device),
                "pellets_remaining": torch.tensor(completed_pellets_remaining, dtype=torch.float32, device=self.device),
            }

        return self.get_observations(), reward, dones, extras

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _load_model(self) -> DreamerWorldModel:
        ckpt_path = Path(str(self.wm_cfg.get("checkpoint", "checkpoints/dreamer_wm/rl_single_L0_twohot/latest.pt")))
        if not ckpt_path.is_absolute():
            ckpt_path = Path(__file__).resolve().parents[1] / ckpt_path
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        model = DreamerWorldModel(WorldModelConfig(**ckpt["cfg"]))
        model.load_state_dict(ckpt["model"])
        return model

    def _make_seed_env(self) -> PacmanEnv:
        reward_cfg = RewardConfig(**self.cfg.get("reward", {}))
        ghost_cfg = self.cfg.get("ghost", {})
        power_cfg = self.cfg.get("power_pellet", {})
        episode_cfg = self.cfg.get("episode", {})
        return PacmanEnv(
            layout_path=self.cfg["layout_file"],
            num_ghosts=int(ghost_cfg.get("num_ghosts", self.cfg.get("num_ghosts", 1))),
            ghost_epsilon=float(ghost_cfg.get("epsilon", self.cfg.get("ghost_epsilon", 0.2))),
            ghost_policy=str(ghost_cfg.get("policy", self.cfg.get("ghost_policy", "chase_stochastic"))),
            ghost_speed_ratio=float(ghost_cfg.get("speed_ratio", self.cfg.get("ghost_speed_ratio", 1.0))),
            power_pellet_enabled=bool(power_cfg.get("enabled", self.cfg.get("power_pellet_enabled", False))),
            frightened_duration=int(power_cfg.get("frightened_duration", self.cfg.get("frightened_duration", 30))),
            max_steps=int(episode_cfg.get("max_steps", self.cfg.get("max_steps", 500))),
            reward_config=reward_cfg,
            randomize_spawn=bool(self.cfg.get("randomize_spawn", True)),
            min_spawn_dist=int(self.cfg.get("min_spawn_dist", 3)),
        )

    @torch.no_grad()
    def _reset_one(self, idx: int, seed: int | None = None) -> None:
        obs_np, _ = self.envs[idx].reset(seed=seed)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        h0, z0 = self.model.initial_state(1, self.device)
        wall = obs[:, WALL_SLICE]
        e = self.model.embed_layout(wall)
        noop = torch.full((1,), NOOP_ACTION, dtype=torch.long, device=self.device)
        h, z = self.model.encode(obs, h0, noop, e, z_prev=z0)
        self._obs[idx] = obs.squeeze(0)
        self._wall[idx] = wall.squeeze(0)
        self._total_pellets[idx] = float(getattr(self.envs[idx], "_total_pellets", (obs[:, FOOD_SLICE] > 0.5).sum().item()))
        self._initial_power[idx] = torch.as_tensor(
            self.envs[idx].layout.to_padded_arrays()["initial_power"].reshape(-1), dtype=torch.float32, device=self.device
        )
        self._e[idx] = e.squeeze(0)
        self._h[idx] = h.squeeze(0)
        self._z[idx] = z.squeeze(0)
        self.episode_length_buf[idx] = 0
        self._episode_returns[idx] = 0.0

    def _full_state(self, dyn: torch.Tensor) -> torch.Tensor:
        return torch.cat([dyn[:, :459], self._wall, dyn[:, 459:460]], dim=-1).clamp(-5.0, 5.0)

    @staticmethod
    def _self_ensemble_component_weights(wm_cfg: dict[str, Any]) -> dict[str, float]:
        weights = wm_cfg.get("self_ensemble_component_weights", {}) or {}
        if not isinstance(weights, dict):
            raise ValueError("world_model.self_ensemble_component_weights must be a mapping")
        allowed = {"pacman_position", "ghost_positions", "food_mask", "power_timer"}
        unknown = set(weights) - allowed
        if unknown:
            raise ValueError(f"Unknown self-ensemble component weights: {sorted(unknown)}")
        return {name: float(value) for name, value in weights.items()}

    def _self_ensemble_uncertainty(self, decoded_samples: torch.Tensor) -> SelfEnsembleStats:
        if not self.use_uncertainty:
            zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            ones = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            return SelfEnsembleStats(zeros, zeros, ones, zeros.bool())

        uncertainty = component_weighted_decoded_state_variance(decoded_samples, self.self_ensemble_component_weights)
        uncertainty_norm = self.uncertainty_mean.normalize(uncertainty)
        return self_ensemble_stats(
            uncertainty,
            uncertainty_norm,
            alpha=self.confidence_alpha,
            confidence_weight_scale=self.confidence_weight_scale,
            threshold=self.self_ensemble_threshold,
        )
