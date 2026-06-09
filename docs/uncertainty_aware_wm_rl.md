# Uncertainty-Aware WM-RL

This note explains the implemented uncertainty-aware PPO training path for Pac-Man world-model RL. PPO can train on transitions imagined by the frozen Dreamer world model, and the uncertainty-aware mode reduces gradient pressure from imagined transitions whose stochastic self-ensemble outcomes disagree.

## Modes

The PPO config has two separate switches:

| Config | Meaning |
|---|---|
| `world_model.use_wm: false` | Train PPO on the ground-truth Pac-Man env. |
| `world_model.use_wm: true` | Train PPO from Dreamer imagined transitions. |
| `world_model.use_uncertainty_aware_methods: false` | Vanilla WM-PPO ablation: no confidence-weighted PPO loss and no uncertainty-triggered rollout truncation. |
| `world_model.use_uncertainty_aware_methods: true` | Enable self-ensemble confidence weighting and adaptive imagined-rollout truncation. |

The vanilla WM-PPO ablation remains important because it isolates whether the world model itself is useful before adding uncertainty-aware safeguards.

## Self-Ensemble Uncertainty

The Dreamer world model has stochastic latent factors. At each imagined step, the WM env computes the next recurrent state once, then samples `world_model.self_ensemble_inferences` candidate stochastic latents from the same prior distribution. Each candidate is decoded into a next state. The first sample advances the actual rollout, while all samples are used to estimate uncertainty.

The scalar uncertainty for each vectorized env transition is a component-weighted mean of decoded-state variance across self-ensemble samples:

```text
u_component = mean(var(decoded_component_samples, dim=ensemble), dim=component_dims)
u = weighted_mean([u_pacman_position, u_ghost_positions, u_food_mask, u_power_timer])
u_norm = u / (running_mean_u + 1e-8)
```

The weights are configured under `world_model.self_ensemble_component_weights`. Wall-mask dimensions are ignored because they are static layout context, not stochastic predictions.

With `self_ensemble_inferences: 1`, the variance is zero, so uncertainty-aware weighting is effectively neutral.

## Confidence-Weighted PPO Updates

Transition confidence is computed from normalized self-ensemble uncertainty:

```text
confidence = exp(-confidence_alpha * u_norm).clamp(min_confidence, 1.0).detach()
```

PPO policy and value losses are weighted by confidence, but entropy loss is not weighted:

```text
policy_loss = mean(confidence * policy_loss_per_sample)
value_loss = mean(confidence * value_loss_per_sample)
loss = policy_loss + value_coef * value_loss - entropy_coef * mean(entropy)
```

This lets stable imagined transitions contribute normally while reducing gradient pressure from high-disagreement transitions.

## Adaptive Imagined-Rollout Truncation

When `world_model.adaptive_rollout_truncation: true`, uncertainty-aware WM-PPO truncates imagined rollouts with:

```text
truncate = u_norm > self_ensemble_threshold
```

The default threshold is `2.0`. Episode termination from the WM continue head and max episode length still apply independently.

## Logging

Training logs the self-ensemble uncertainty signal alongside normal PPO and Pac-Man metrics:

- Mean raw self-ensemble uncertainty.
- Mean normalized self-ensemble uncertainty.
- Mean transition confidence.
- Fraction of transitions truncated by self-ensemble uncertainty.
- PPO return, episode length, win, death, timeout, and pellet metrics as usual.
