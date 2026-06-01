# DreamerV3-style State-based World Model (`world_model/dreamer/`)

Implements `WORLDMODEL_DREAMERV3_SPEC.md`: a state-based RSSM that predicts, from
`(state, action)` sequences, the next latent state, reward, and continue flag,
and reconstructs the dynamic part of the Pac-Man state. **World model only** — no
actor/critic/value/returns/planning (spec §1). Built as a fresh package; the
legacy v10c / JEPA model in the parent `world_model/` is kept untouched as the
ablation baseline (spec §11).

## Gap analysis (spec §0): legacy v10c → this RSSM

| Component | Legacy v10c (`world_model/single.py` …) | This implementation (spec target) |
|---|---|---|
| Latent | continuous 128-d vector | **categorical 32×32**, straight-through, 1% unimix (`nn.OneHotCategoricalST`) |
| Encoder | CNN/MLP → z | posterior `q(z\|h,x)` MLP, symlog obs (`rssm.Encoder`) |
| Dynamics | point predictor `h→ẑ` | prior `p(ẑ\|h)` (`rssm.DynamicsPredictor`), matched by KL |
| Sequence model | `GRU(z,a)` | layout-conditioned `GRU(h,z,a,e)` (`rssm.SequenceModel`) |
| Reward | deterministic `symlog(step+pellet·σ)` | **two-hot symlog**, K=255, B=[−20,20], zero-init (`rssm.RewardHead`) |
| Continue | DoneHead (done) | Bernoulli **continue=1−done** (`rssm.ContinueHead`) |
| Decoder | DynamicStateHead dual-path | decode `x_dyn`(460) from `{h,z}`; symlog+MSE / BCE (`rssm.Decoder`) |
| Layout | wall_mask in encoder only | MLP `wall_mask→e(32)` fed to GRU (`rssm.LayoutEmbedder`) |
| Loss | `MSE_latent + βr·MSE + βd·BCE + βvar·var (+aux)` | `L_pred + 0.5·L_dyn + 0.1·L_rep`, free bits=1 nat, **no var reg** (`loss.py`) |
| Optimizer | Adam, grad_clip 0.5 | **LaProp + AGC**, lr 1e-4 (`optim.py`) |
| Norm | LayerNorm | **RMSNorm** (`nn.RMSNorm`) |
| Replay | per-file random window | uniform length-L subseq over concatenated stream, `is_first` resets, h₀ warmup (`replay.py`) |
| Data | single layout | mixed-policy, multi-layout, train/test split (`data_pipeline/`, `scripts/wm_*`) |

## Files
- `nn.py` — RMSNorm, SiLU MLP, symlog/symexp, straight-through categorical + unimix, two-hot encode/decode, categorical KL.
- `rssm.py` — LayoutEmbedder, SequenceModel (GRU core), Encoder (posterior), DynamicsPredictor (prior), RewardHead, ContinueHead, Decoder + state-slice constants.
- `world_model.py` — `DreamerWorldModel`: `observe` (teacher-forced posterior rollout) + the spec §12 interface (`encode` / `imagine_step` / `decode`) + `initial_state` / `embed_layout`.
- `loss.py` — `L_pred + β_dyn·L_dyn + β_rep·L_rep`, free-bits KL balancing.
- `optim.py` — `LaProp` optimizer + `adaptive_grad_clip` (AGC).
- `replay.py` — `SequenceReplay` over the concatenated step-stream.
- `eval.py` — one-step / k-step open-loop / collapse / cross-layout metrics (spec §10).

## End-to-end usage

```bash
PY=/home/ubuntu/miniconda/envs/pacman-wm/bin/python   # env with gymnasium+torch+cuda

# 1. Layout pools (train + disjoint held-out test), reproducible seeds + manifest
$PY scripts/wm_generate_layouts.py --n-train 30 --n-test 5 --seed 0

# 2. Mixed-policy training dataset (~500K to start; raise toward 1–2M if weak)
$PY scripts/wm_collect_dataset.py --dataset main --n-transitions 500000
#    held-out cross-layout EVAL dataset (test pool — never used for training)
$PY scripts/wm_collect_dataset.py --dataset main_test --pool-split test --n-transitions 20000

# 3. Train the world model
$PY scripts/wm_train_dreamer.py --dataset main --device cuda

# 4. Intrinsic eval (in-distribution + cross-layout)
$PY scripts/wm_eval_dreamer.py --checkpoint checkpoints/dreamer_wm/main/best.pt \
    --dataset main --test-dataset main_test
```

Config (single source of truth): `configs/world_model/dreamer_v3.yaml`.

## Data alignment (DreamerV3 convention)
At stored index `t`: `states[t]=s_t`, `actions[t]=a_{t-1}` (action into s_t),
`rewards[t]=r_{t-1}`, `continues[t]=1−done_{t-1}`, `is_first[t]` marks episode
starts. The terminal state is stored. The replay samples length-L windows over
the whole concatenated stream (across episode boundaries); `is_first` resets the
recurrent carry mid-window. The leading `context` steps warm h₀ and are excluded
from the loss.

## Notes / deviations
- Coverage diagnostics: `scripts/wm_collect_dataset.py` prints a reward-event
  tally (pellet / power / ghost-eat / death / win / step-only). With 1 ghost and
  power pellets disabled, ghost-eat/power/win are absent — raise ghosts /
  enable power pellets, or include winning trajectories, if those reward buckets
  must be learned (spec §3.5).
- PPO checkpoints are not in the policy pool (the available agent was trained on
  a single layout and does not transfer to generated mazes); the pool is
  random + greedy-BFS with ε spanning the novice→expert spectrum (spec §3.3).
