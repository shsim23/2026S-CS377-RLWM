# Uncertainty-Aware WM-RL

This note explains the implemented reliability-aware PPO training path for Pac-Man world-model RL. The goal is to train the PPO agent on transitions imagined by the Dreamer world model, while reducing the damage from model errors that look plausible to the policy optimizer.

## Modes

The PPO config has two separate switches:

| Config | Meaning |
|---|---|
| `world_model.use_wm: false` | Current baseline: train PPO on the ground-truth Pac-Man env. |
| `world_model.use_wm: true` | Train PPO from Dreamer imagined transitions. |
| `world_model.use_uncertainty_aware_methods: false` | Vanilla WM-PPO ablation: no confidence-weighted PPO loss and no uncertainty-triggered rollout truncation. |
| `world_model.use_uncertainty_aware_methods: true` | Enable reliability-aware PPO weighting and adaptive imagined-rollout truncation. |

The vanilla WM-PPO ablation is important. It tells us whether the world model itself is useful before adding uncertainty-aware safeguards.

## Why Rule-Based Reliability Is the Main Signal

Pac-Man has hard transition rules: Pac-Man moves at most one cell, walls block movement, ghosts cannot move through walls, and food cannot randomly appear. A learned world model can still produce transitions that violate those rules, especially during open-loop rollout. These violations are often more useful than generic neural uncertainty because they directly identify impossible imagined experience.

This matters because world models can be confidently wrong. A single Dreamer model may assign high confidence to an impossible next state if that state lies on a familiar-looking but incorrect manifold. A rule-based checker catches errors that model confidence alone may miss.

The reliability score uses hard-rule violations as the main signal:

- Pac-Man action consistency against the chosen action and walls.
- Pac-Man speed, wall, and bounds checks.
- Ghost speed, wall, and bounds checks.
- Food count cannot increase.
- At most one pellet should disappear per transition.
- Removed food should disappear at Pac-Man's predicted next position.
- Optional collision/done consistency when done probability is available.

The score is one scalar per vectorized env transition.

## Why Prior Entropy Is Only Weak Auxiliary Signal

Dreamer prior categorical entropy can still be useful, but it should not dominate the decision. Prior entropy is a model-internal uncertainty proxy, not a direct proof that a transition is invalid.

There are two uncertainty types to keep separate:

- Aleatoric uncertainty: real randomness in the environment. In this project, ghost behavior is intentionally stochastic with epsilon-greedy movement. High uncertainty about which legal ghost move happens is not necessarily a world-model failure.
- Epistemic uncertainty: uncertainty from model ignorance or insufficient training data. This is the uncertainty we would like to avoid training on too heavily.

A single prior-entropy number can mix these together. If we treated ghost stochasticity as the main uncertainty signal, we could incorrectly truncate valid transitions just because the ghost had several legal stochastic choices. For this reason, ghost entropy is not a main signal, and prior entropy is only used as a weak auxiliary term when prior logits are already available.

## Why Not an Ensemble First

An ensemble can estimate epistemic uncertainty better than a single model, but it is expensive here:

- Multiple Dreamer models would need to be trained and checkpointed.
- WM inference cost would multiply during PPO rollout collection.
- The first WM-RL comparison needs a clean baseline: ground-truth PPO vs vanilla WM-PPO vs reliability-aware WM-PPO.

The first implementation therefore uses one trained Dreamer checkpoint plus deterministic rule checks. An ensemble can be added later if the single-model reliability method is not enough.

## Confidence-Weighted PPO Updates

For each imagined transition, compute:

```text
u_rule_norm = u_rule / (running_mean_rule + 1e-8)
u_prior_norm = u_prior / (running_mean_prior + 1e-8), if prior entropy is available
```

Then combine:

```text
u_total = 2.0 * u_rule_norm + 1.0 * u_prior_norm
```

If prior entropy is unavailable:

```text
u_total = u_rule_norm
```

Transition confidence is:

```text
confidence = exp(-alpha * u_total).clamp(min=0.1, max=1.0).detach()
```

Default `alpha` is `0.5`. PPO policy and value losses are weighted by confidence, but entropy loss is not weighted:

```text
policy_loss = mean(confidence * policy_loss_per_sample)
value_loss = mean(confidence * value_loss_per_sample)
loss = policy_loss + value_coef * value_loss - entropy_coef * mean(entropy)
```

This lets useful imagined transitions still contribute while reducing gradient pressure from suspicious transitions.

## Adaptive Imagined-Rollout Truncation

The main truncation condition uses rule violations:

```text
truncate = u_rule_norm > rule_threshold
```

Default `rule_threshold` is `2.0`.

If prior entropy is available, a secondary condition may also truncate:

```text
truncate = truncate or ((u_rule_norm > 1.0) and (u_prior_norm > 2.0))
```

This avoids truncating solely because the ghost has stochastic legal choices. A rollout should mainly stop when the WM starts violating hard Pac-Man dynamics.

## Logging

Training should log:

- Mean raw rule violation score.
- Mean normalized rule violation score.
- Mean normalized prior entropy, if available.
- Mean combined uncertainty score.
- Mean transition confidence.
- Fraction of transitions truncated by rule violation.
- Fraction of transitions truncated by the secondary prior-entropy condition.
- PPO return, episode length, and success metrics as usual.
