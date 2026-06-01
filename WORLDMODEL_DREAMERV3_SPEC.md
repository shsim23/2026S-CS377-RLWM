# World Model Implementation Spec — DreamerV3-style State-based RSSM

This document specifies the target world model architecture for the Pac-Man project and how to implement it. It is the single source of truth for this phase.

## 0. First task: gap analysis against the current code

**Before writing or changing anything, do this first:**

1. Read the existing world model implementation in this repository (model definition, loss, training loop, config, replay buffer).
2. Produce a written gap analysis comparing the current implementation to the target architecture in this document. For each component below (encoder, latent representation, sequence model, dynamics predictor, reward head, continue head, decoder, loss terms, optimizer, training settings), state: (a) what the current code does, (b) what this spec requires, (c) the concrete change needed.
3. Propose an incremental change plan ordered by risk (lowest first), following §11.
4. Only after the gap analysis is reviewed, begin implementation.

Do not assume the current code matches any earlier design. Treat the repository code as ground truth and this document as the target.

---

## 1. Scope

Build **only the world model**. There is no actor, no critic, no value head, no return computation, and no planning in this phase. The world model's job is to predict, from `(state, action)` sequences, the next latent state, the reward, and the episode-continuation flag, and to reconstruct the dynamic part of the state.

The model is trained on real environment transitions from a replay buffer. It must support open-loop "imagination" rollouts (predicting forward without access to observations), because a later phase will train a policy entirely inside these rollouts.

---

## 2. Environment I/O

- **State vector**: 901-dim flat. It is composed of a **dynamic part (~460-dim)** — Pac-Man coordinates, ghost slots (coordinates + alive/valid flags), food/pellet state — and a **static `wall_mask` (441-dim)** describing the maze layout.
- **Action**: discrete, 5 actions (UP, DOWN, LEFT, RIGHT, NOOP). Embed into a vector before feeding the sequence model.

Handling of the two state parts:
- **Encoder input** `x_t`: the full 901-dim vector (so the posterior can place entities relative to walls).
- **Decoder reconstruction target** `x_dyn`: the **dynamic ~460-dim only**. Do not reconstruct `wall_mask` — it is static and known, and reconstructing it wastes decoder capacity and dilutes the loss.
- **Layout conditioning** `e`: a small MLP maps `wall_mask → e` (low-dim, ~32) once per episode. Feed `e` to the sequence model so dynamics are layout-aware. This is what enables one world model to generalize across maze layouts (change `wall_mask`/`e`, reuse the model).

---

## 3. Data collection pipeline (offline fixed dataset)

The world model is trained on a **fixed, pre-collected offline dataset** of real environment transitions. There is no online data collection during world-model training in this phase. (The interface in §12 leaves room to add online iterative collection in a later phase, but it is out of scope here.)

The two existing assets are used to build the dataset: the **map generator** and the **trained RL agent**.

### 3.1 Why coverage matters (read before implementing)

A world model is only accurate inside the state distribution it was trained on. Two failure modes must be avoided:

- **Narrow-policy collapse**: if data comes only from a well-trained agent, the dataset contains only competent "success" trajectories. The model never sees deaths, cornered states, or inefficient movement, so it cannot predict those dynamics. In the later phase the policy starts out behaving badly and immediately enters exactly those unseen states, where the model produces garbage (model exploitation). Collecting only good-agent data is therefore harmful.
- **No rare events**: the two-hot reward head can only learn a reward bucket if some samples land in it. If ghost-eating, power-pellet, win, and death events are nearly absent, those rewards are never learned.

The fix is a **mixed-policy, multi-layout** dataset that deliberately maximizes state-space coverage — recreating offline the "novice-to-expert" distribution that online DreamerV3 would get for free.

### 3.2 Layout pool — train/test split fixed up front

**Generate the test (held-out) layouts at the same time as the training layouts, and keep them strictly separate.** The test layouts must never contribute any transition to the training dataset — they exist only to evaluate cross-layout zero-shot generalization later. Mixing them in would invalidate the generalization evaluation.

- Use the map generator to produce a **training layout pool** (e.g. 20–50 layouts) and a separate, disjoint **test layout pool** (e.g. a handful) in one step.
- Persist the generator **seeds / layout IDs** for both pools so the split is reproducible and auditable.
- Record which pool each layout belongs to. The collection process samples **only from the training pool**. The test pool is written to disk, labeled, and otherwise untouched in this phase.

### 3.3 Policy pool — mixed for coverage

- Include several **checkpoints of the trained RL agent** at different competence levels (e.g. ~25% / ~50% / 100% of training) so the dataset spans novice through expert behavior.
- Include a **pure-random policy**.
- Inject extra exploration via **ε-greedy** on the agent policies.
- Suggested starting mix: ~20–30% pure-random episodes, ~70–80% agent-checkpoint episodes (spread across checkpoints). Tune from coverage diagnostics (§3.5).

### 3.4 Episode loop

1. Sample a `(layout, policy)` pair — **layout from the training pool only**, policy from the policy pool.
2. Roll out the simulator, applying ε-greedy exploration.
3. Store each transition as `(s_t, a_t, r_t, c_t, layout_id)` where `c_t = 1 - done`.
4. Save episode-by-episode, but index at the **step level** so the replay can uniformly sample length-`L` subsequences across episode boundaries (DreamerV3 convention). Mask so subsequences do not bleed across a `done` into the next episode's first step in a way that corrupts the transition.

### 3.5 Size target and coverage diagnostics

- **Start at ~500K transitions.** Train the world model, check k-step prediction accuracy, and increase toward **1–2M** if prediction is weak — especially if rare-event reward prediction is poor. The replay capacity (§7, 1e6) is consistent with this range.
- **Coverage diagnostic**: tally the frequency of each reward event (pellet, power-pellet, ghost-eat, death, win, step penalty) in the collected data. If any rare event is too sparse for the reward head to learn its bucket, raise the proportion of policies/situations that induce it.

### 3.6 Dataset artifacts to produce

- The **training transition dataset** (mixed-policy, training layouts only).
- The **train layout pool** and the **test layout pool**, each with seeds/IDs, stored separately and clearly labeled. The test pool carries no transitions.
- A small **metadata/manifest** recording: layout split, policy mix and checkpoints used, ε schedule, total transitions, and the reward-event coverage tally.

---

## 4. RSSM architecture

Implement the world model as a Recurrent State-Space Model with these six components:

```
Sequence model:       h_t  = f_φ(h_{t-1}, z_{t-1}, a_{t-1}, e)
Encoder (posterior):  z_t  ~ q_φ(z_t | h_t, x_t)
Dynamics pred (prior):ẑ_t  ~ p_φ(ẑ_t | h_t)
Reward predictor:     r̂_t  ~ p_φ(r_t | h_t, z_t)
Continue predictor:   ĉ_t  ~ p_φ(c_t | h_t, z_t)
Decoder:              x̂_t  ~ p_φ(x_dyn | h_t, z_t)
```

- The model state carried forward is `s_t = {h_t, z_t}` (deterministic recurrent state + stochastic latent).
- `encoder` and `dynamics predictor` are the posterior and prior over the same latent `z`. They are matched by the KL losses in §6. At imagination time only the prior `p_φ(ẑ_t | h_t)` is used (no `x_t`).

### 4.1 Latent representation (structural collapse prevention)

- `z` is a **vector of categoricals: 32 groups × 32 classes**. Sample one-hot per group with **straight-through gradients** (one-hot on forward, softmax gradient on backward).
- Apply **1% unimix**: parameterize each categorical as `0.99 · softmax(logits) + 0.01 · uniform`. This makes the distribution unable to become near-deterministic and keeps the KL well-scaled.
- This categorical structure plus the free-bits KL (§6) is the collapse-prevention mechanism. **Do not add a variance regularizer.**

### 4.2 Network primitives

- Encoder, decoder, dynamics predictor, reward head, continue head: **MLPs** (the input is low-dimensional state, so no CNN).
- Use **RMSNorm** and **SiLU** activations. Use a **GRU** for the sequence model (block-GRU as in DreamerV3 if convenient; a standard GRU is acceptable for this scale).
- Squash encoder inputs with `symlog` (continuous coordinate components).

---

## 5. Prediction heads

### 5.1 Reward head — two-hot categorical regression

- Apply `symlog` to the raw reward target, then **two-hot encode** over `K = 255` equally spaced buckets on the support `B = [-20, +20]`. (Our reward magnitudes are small, so the support is generous; you may shrink to `K = 51`, `B = [-5, +5]` as an optimization, but start with the standard `255 / [-20, 20]`.)
- The head outputs a softmax over buckets; the scalar prediction is `symexp(E_bucket[b])`.
- Loss: categorical cross-entropy against the two-hot soft label (stop-gradient on the target).
- **Zero-initialize the output layer** to avoid large initial reward predictions that delay learning.

### 5.2 Continue head — Bernoulli

- Predicts `continue = 1 - done`. Binary cross-entropy loss.

### 5.3 Decoder — dynamic state reconstruction

- Reconstructs the ~460-dim dynamic state from `{h_t, z_t}`.
- Continuous components (coordinates): `symlog` + squared error.
- Binary components (alive / valid / food-present flags): Bernoulli / BCE.
- (This head also serves as the representation-shaping signal, the role the DreamerV3 decoder plays. If a dual-path variant — reconstructing from both the encoder path and the dynamics path — is already present in the codebase, it may be retained as an enhancement.)

---

## 6. Training objective (world model loss)

```
L(φ) = E_q [ Σ_t  β_pred · L_pred + β_dyn · L_dyn + β_rep · L_rep ]

β_pred = 1.0,  β_dyn = 0.5,  β_rep = 0.1

L_pred = -ln p(x_dyn | z_t, h_t)  -  ln p(r_t | z_t, h_t)  -  ln p(c_t | z_t, h_t)
L_dyn  = max(1, KL[ sg(q(z_t | h_t, x_t)) ‖     p(z_t | h_t)  ])
L_rep  = max(1, KL[     q(z_t | h_t, x_t)  ‖ sg(p(z_t | h_t)) ])
```

- `sg(·)` is stop-gradient. The two KL terms differ only in which side is stopped and in their weight (KL balancing).
- **Free bits**: the `max(1, ·)` clips each KL below 1 nat (≈1.44 bits), disabling the term once it is already small so the model focuses on prediction.
- No KL annealing, no weight decay, no dropout.

---

## 7. Training settings

| Setting | Value |
|---|---|
| Batch size (sequences) | 16 |
| Sequence length | 64 (may use ~50 to fit episode structure) |
| Replay buffer capacity | 1e6 transitions |
| Replay sampling | uniform over all subsequences of length = sequence length, ignoring episode boundaries |
| Train ratio (train steps / env steps) | start low (Pac-Man simulation is cheap; data is not the bottleneck), tune upward only if data efficiency matters |
| Teacher forcing | drive the GRU with posterior `z_t` during training |
| Initial `h_0` | accumulate from real history at the start of each sampled sequence |
| Optimizer | LaProp (RMSProp then momentum) |
| Gradient clipping | adaptive gradient clipping (AGC) |
| Learning rate (world model) | 1e-4 |
| Normalization / activation | RMSNorm / SiLU |

---

## 8. Model size

Start at or below DreamerV3 "XS". The task is much simpler than DreamerV3's benchmarks, so this is a ceiling, not a target.

| Knob | Start value |
|---|---|
| GRU recurrent units (h dim) | 256 |
| Dense / MLP hidden units | 256 |
| MLP layers | 1–2 |
| Latent | 32 groups × 32 classes |
| Layout embedding `e` dim | 32 |

Scale up only if k-step rollout accuracy is the bottleneck.

---

## 9. Deviations from standard DreamerV3 (be explicit in code comments)

Dropped for this phase (all are actor/critic concerns):
- Actor network, critic network, value head, λ-returns, percentile return normalization, entropy regularizer, imagination-horizon rollout for policy.
- CNN encoder/decoder (we are state-based, MLP only).

Changed / specialized:
- Decoder reconstructs the **dynamic state (~460-dim) only**, not the full observation; `wall_mask` enters as conditioning `e`, not as a reconstruction target.
- Sequence model is conditioned on the layout embedding `e` for cross-layout generalization.

Kept faithfully:
- Categorical latent (32×32) + straight-through + 1% unimix.
- `L_pred + 0.5·L_dyn + 0.1·L_rep` with free bits = 1 nat.
- Two-hot symlog reward head (zero-init), Bernoulli continue head, symlog decoder.
- LaProp + AGC, RMSNorm, SiLU.

---

## 10. Intrinsic evaluation (no policy needed)

Measure world-model prediction quality directly:

- **k-step open-loop rollout**: from a real context window, roll the **prior** forward N steps (no observations), decode, and report dynamic-state reconstruction error, reward MSE (in raw space via symexp), and continue accuracy as a function of horizon (e.g. N up to one sequence length).
- **One-step metrics**: decoder reconstruction error, reward two-hot accuracy and raw MSE, continue accuracy.
- **Collapse metrics**: per-group categorical entropy and the count of groups collapsed to a near-deterministic class. Healthy = no collapsed groups, entropy well above zero.
- **Cross-layout generalization**: the same metrics on held-out (unseen) layouts, changing only `wall_mask` / `e`.

Primary success signal for this phase: break the reward-MSE plateau the previous lineage was stuck at, and reach low k-step rollout error with no collapsed latent groups.

---

## 11. Implementation order (lowest risk first)

1. **Two-hot symlog reward head** (with zero-init). Confirm reward learning improves over the current reward mechanism before changing anything else. Keep the current reward head available as an ablation baseline.
2. **Categorical latent (32×32) + straight-through + 1% unimix**, and switch the self-prediction objective to `L_dyn` / `L_rep` with free bits. Remove any variance regularizer. Verify training stability and no collapsed groups.
3. **Layout conditioning `e`** + multi-layout training. Measure cross-layout k-step accuracy.
4. **Tune**: free-bits, KL balancing weights, model size, train ratio. Lock the final config as the single runtime source of truth.

---

## 12. Interface contract for the later policy phase

Expose this so the policy phase can build on top without touching internals:

```python
class WorldModel:
    def encode(self, x, h, a_prev, e) -> (h, z)                      # posterior step (uses observation)
    def imagine_step(self, h, z, a, e) -> (h, z_next, reward, cont)  # prior step (no observation)
    def decode(self, h, z) -> x_dyn                                  # ~460-dim dynamic state (eval/debug)
```

Epistemic uncertainty for downstream variance-aware policy work is **not** produced by this model and must not influence its design. It will be sourced separately in the later phase.
