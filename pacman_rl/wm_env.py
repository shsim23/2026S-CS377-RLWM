from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from pacman_env import PacmanEnv, RewardConfig, STATE_DIM
from pacman_env.constants import Action
from world_model.dreamer import DreamerWorldModel, WorldModelConfig
from world_model.dreamer.nn import OneHotCategoricalST
from world_model.dreamer.rssm import FOOD_SLICE, PAC_SLICE, WALL_SLICE


NOOP_ACTION = int(Action.NOOP)


from pacman_rl.wm_reliability import ReliabilityStats, RuleBasedTransitionScorer, RunningMean


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
        self.use_prior_entropy = bool(self.wm_cfg.get("use_prior_entropy", True))
        self.confidence_alpha = float(self.wm_cfg.get("confidence_alpha", 0.5))
        self.min_confidence = float(self.wm_cfg.get("min_confidence", 0.1))
        self.rule_threshold = float(self.wm_cfg.get("rule_threshold", 2.0))
        self.secondary_rule_threshold = float(self.wm_cfg.get("secondary_rule_threshold", 1.0))
        self.secondary_prior_threshold = float(self.wm_cfg.get("secondary_prior_threshold", 2.0))
        self.continue_threshold = float(self.wm_cfg.get("continue_threshold", 0.5))
        momentum = float(self.wm_cfg.get("running_mean_momentum", 0.99))
        self.rule_mean = RunningMean(momentum=momentum, device=self.device)
        self.prior_mean = RunningMean(momentum=momentum, device=self.device)
        self.scorer = RuleBasedTransitionScorer(self.wm_cfg)

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
        z_next = dist.mode() if bool(self.wm_cfg.get("deterministic_latent", False)) else dist.sample_st()
        z_flat = self.model._flat(z_next)
        reward = self.model.reward_from_logits(self.model.reward_head(h_next, z_flat)).float()
        cont_logits = self.model.cont_head(h_next, z_flat)
        cont = torch.sigmoid(cont_logits)
        done_prob = 1.0 - cont
        pred_dyn = self.model.decode_state(h_next, z_flat)
        pred_obs = self._full_state(pred_dyn)

        reliability = self._reliability(prev_obs, pred_obs, action_ids, prior_logits, done_prob)
        wm_done = cont < self.continue_threshold
        timeout = self.episode_length_buf + 1 >= self.max_episode_length
        dones = wm_done | timeout
        if self.use_uncertainty and bool(self.wm_cfg.get("adaptive_rollout_truncation", True)):
            dones = dones | reliability.rule_truncate | reliability.secondary_truncate

        self._episode_returns += reward
        self.episode_length_buf += 1
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_timeouts: list[float] = []
        completed_pellets_remaining: list[float] = []

        self._obs = pred_obs
        self._h = h_next
        self._z = z_next

        pellets_remaining = (self._obs[:, FOOD_SLICE] > 0.5).float().sum(dim=1)
        for i in torch.nonzero(dones, as_tuple=False).flatten().tolist():
            completed_returns.append(float(self._episode_returns[i].item()))
            completed_lengths.append(int(self.episode_length_buf[i].item()))
            completed_timeouts.append(float(timeout[i].item()))
            completed_pellets_remaining.append(float(pellets_remaining[i].item()))
            self._reset_one(i)

        extras: dict[str, Any] = {
            "time_outs": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "wm_confidence": reliability.confidence,
            "log": {
                "/wm/u_rule": reliability.u_rule,
                "/wm/u_rule_norm": reliability.u_rule_norm,
                "/wm/u_total": reliability.u_total,
                "/wm/confidence": reliability.confidence,
                "/wm/rule_truncation_rate": reliability.rule_truncate.float(),
                "/wm/secondary_truncation_rate": reliability.secondary_truncate.float(),
                "/pacman/pellets_remaining": pellets_remaining,
            },
        }
        if reliability.u_prior_norm is not None:
            extras["log"]["/wm/u_prior_norm"] = reliability.u_prior_norm
        if completed_returns:
            extras["episode"] = {
                "return": torch.tensor(completed_returns, dtype=torch.float32, device=self.device),
                "length": torch.tensor(completed_lengths, dtype=torch.float32, device=self.device),
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
        self._e[idx] = e.squeeze(0)
        self._h[idx] = h.squeeze(0)
        self._z[idx] = z.squeeze(0)
        self.episode_length_buf[idx] = 0
        self._episode_returns[idx] = 0.0

    def _full_state(self, dyn: torch.Tensor) -> torch.Tensor:
        return torch.cat([dyn[:, :459], self._wall, dyn[:, 459:460]], dim=-1).clamp(-5.0, 5.0)

    def _reliability(
        self,
        prev_obs: torch.Tensor,
        pred_obs: torch.Tensor,
        actions: torch.Tensor,
        prior_logits: torch.Tensor,
        done_prob: torch.Tensor,
    ) -> ReliabilityStats:
        if not self.use_uncertainty:
            zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            ones = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            return ReliabilityStats(zeros, zeros, None, None, zeros, ones, zeros.bool(), zeros.bool())

        u_rule = self.scorer.score(prev_obs, pred_obs, actions, done_prob=done_prob)
        u_rule_norm = self.rule_mean.normalize(u_rule)
        u_prior = None
        u_prior_norm = None
        if self.use_prior_entropy and prior_logits is not None:
            probs = F.softmax(prior_logits, dim=-1)
            ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            u_prior = ent.mean(dim=-1)
            u_prior_norm = self.prior_mean.normalize(u_prior)
            u_total = 2.0 * u_rule_norm + u_prior_norm
            secondary = (u_rule_norm > self.secondary_rule_threshold) & (u_prior_norm > self.secondary_prior_threshold)
        else:
            u_total = u_rule_norm
            secondary = torch.zeros_like(u_rule_norm, dtype=torch.bool)

        confidence = torch.exp(-self.confidence_alpha * u_total).clamp(self.min_confidence, 1.0).detach()
        rule_truncate = u_rule_norm > self.rule_threshold
        return ReliabilityStats(u_rule, u_rule_norm, u_prior, u_prior_norm, u_total, confidence, rule_truncate, secondary)
