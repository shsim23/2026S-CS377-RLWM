# Pac-Man PPO Baseline (`pacman_rl`)

This package trains PPO directly on the ground-truth `pacman_env.PacmanEnv` by default. The WM-RL path uses the same PPO runner interface and collects rollouts from the Dreamer world model when `world_model.use_wm: true` in `pacman_rl/configs/pacman_ppo.yaml`. Observations stay the same 901-dimensional state vector produced by `StateBuilder`.

Run output layout:

```text
logs/pacman_rl/<run_name>/
  train/
    events.out.tfevents...      # TensorBoard
    checkpoints/model_*.pt      # rsl_rl checkpoints
    videos/*.mp4                # training rollout videos, if --video
  play/
    *.mp4                       # playback videos, if --video
```

If `--run-name` is omitted, the current datetime is used as the run folder name.

## Install

For the `pacman_rl` conda environment on a CUDA 12.2 driver:

```bash
conda activate pacman_rl
cd /home/ubuntu/wonjae/world_model

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
# Do not run: python -m pip install -e ".[rl]"
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

## World-Model PPO

`pacman_ppo.yaml` now includes a top-level `world_model` section next to `env` and `train`:

- `world_model.use_wm: false` keeps the current ground-truth environment training path.
- `world_model.use_wm: true` trains PPO from Dreamer imagined transitions using the same `env` layout, spawn, action-space, and episode-length settings for WM inference setup.
- `world_model.use_uncertainty_aware_methods: false` is the vanilla WM-PPO ablation: imagined rollouts are used, but PPO losses are not confidence-weighted and rollouts are not truncated by uncertainty.
- `world_model.use_uncertainty_aware_methods: true` enables self-ensemble confidence-weighted PPO updates and adaptive imagined-rollout truncation.

The uncertainty-aware method is described in [`docs/uncertainty_aware_wm_rl.md`](../docs/uncertainty_aware_wm_rl.md).

WM-PPO smoke / ablation example:

```bash
python pacman_rl/train.py \
    --config pacman_rl/configs/pacman_ppo.yaml \
    --device cuda \
    --headless \
    --run-name wm_vanilla_1
```

Set these config values for vanilla WM-PPO:

```yaml
world_model:
  use_wm: true
  use_uncertainty_aware_methods: false
```

Set these config values for self-ensemble uncertainty-aware WM-PPO:

```yaml
world_model:
  use_wm: true
  use_uncertainty_aware_methods: true
  self_ensemble_inferences: 5
  self_ensemble_threshold: 2.0
  self_ensemble_component_weights:
    pacman_position: 1.0
    ghost_positions: 1.0
    food_mask: 1.0
    power_timer: 1.0
```

## Train

```bash
python pacman_rl/train.py --iterations 1 --num-envs 2 --device cuda --headless
```

Longer run:

```bash
python pacman_rl/train.py \
    --config pacman_rl/configs/pacman_ppo.yaml \
    --device cuda \
    --headless \
    --run-name medium_classic_1
```

TensorBoard:

```bash
tensorboard --logdir logs/pacman_rl/gt_ppo_medium/train
```


Resume from a checkpoint and train additional iterations:

```bash
python pacman_rl/train.py \
    --resume logs/pacman_rl/simple_open_1/train/checkpoints/model_100.pt \
    --iterations 200 \
    --device cuda \
    --headless
```

If `--run-name` is omitted, checkpoints under `logs/pacman_rl/<run>/train/checkpoints/` continue in the same run folder.

Record videos during training:

```bash
python pacman_rl/train.py \
    --device cuda \
    --headless \
    --video \
    --video-every 100 \
    --run-name simple_open_1
```

## Play / Eval

```bash
python pacman_rl/play.py \
    --checkpoint logs/pacman_rl/gt_ppo_medium/train/checkpoints/model_50.pt \
    --episodes 5 \
    --device cuda \
    --headless \
    --video
```

Remove `--headless` to render with a Pygame window when a display is available. Pass `--video` to save `.mp4` files under the run folder.

## Discrete Action Handling

rsl_rl's `MLPModel` emits five logits and `CategoricalDistribution` samples one integer action id:

| id | action |
|---|---|
| 0 | UP |
| 1 | DOWN |
| 2 | LEFT |
| 3 | RIGHT |
| 4 | NOOP |

The action tensor shape remains `(num_envs, 1)` so PPO storage stays compatible with rsl_rl.

## Tests

```bash
python -m pytest -q tests/test_rl_discrete.py tests/test_rl_vec_env.py
```
