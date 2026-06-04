# 2026S CS377 — RL with World Models

Pac-Man 환경에서 **JEPA-style world model**을 학습하고, 그 위에서 모델 기반 policy를 학습하는 프로젝트입니다. 현재는 world model을 쓰기 전 baseline으로, ground-truth `pacman_env`에서 직접 PPO를 학습하는 `pacman_rl/`도 포함합니다.

이 문서는 세 가지를 다룹니다:
1. **맵(layout) 작성 방법**
2. **월드 모델 학습 방법** (데이터 수집 → 학습 → 평가)
3. **GT Pac-Man PPO baseline 학습 방법** (`pacman_rl/`)

월드 모델의 설계가 어떻게 진화해왔는지(reward head 제거, dynamic state head 추가, FoodEatenHead 도입 등)는 [`world_model_implementation.md`](world_model_implementation.md)를 참고하세요.

> 🟡 **단일맵(map 000) WM eval & 시각화 빠른 가이드:** [`docs/single_map_eval.md`](docs/single_map_eval.md) — 학습된 단일맵 월드모델 평가(수치/이미지/GIF) 방법과 텍스트 → 팩맨 렌더링 사용법을 핵심만 정리.

---

## 1. 환경 설치

```bash
# conda env (CUDA 포함). 이미 만들어진 환경: pacman-wm
conda env create -f environment.yml -n pacman-wm
conda activate pacman-wm
```

모든 world-model 명령은 conda env `pacman-wm`을 활성화한 상태에서 실행해야 합니다.

### 1.1 RL baseline 환경 설치 (`pacman_rl`)

PPO baseline은 별도 conda env `pacman_rl`에서 실행하는 것을 권장합니다. CUDA 12.2 driver에서는 rsl_rl 5.3.0의 `torch>=2.6.0` 요구사항 때문에 PyTorch 2.6.0 `cu118` wheel을 사용합니다.

```bash
conda activate pacman_rl
cd /home/ubuntu/wonjae/world_model

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
# Do not run: python -m pip install -e ".[rl]"
```

설치 확인:

```bash
python - <<'PY'
import torch, rsl_rl, tensordict, gymnasium, pygame, pacman_env
print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

---

## 2. Pac-Man 맵 만들기

### 2.1 파일 형식

맵은 단순한 **ASCII grid 텍스트 파일**입니다. `layouts/` 디렉터리에 저장합니다.

| 문자 | 의미 |
|---|---|
| `%` | 벽 (wall) |
| `.` | 일반 pellet (food) |
| `o` | power pellet |
| `P` | 팩맨 시작 위치 (정확히 1개) |
| `G` | 고스트 시작 위치 (1개 이상) |
| (space) | 빈 칸 |

예시 — `layouts/train/pacman_classic.txt`:

```
%%%%%%%%%%%%%%%%%%%
%........%........%
%.%%.%%%.%.%%%.%%.%
%.................%
%.%%.%.%%%%%.%.%%.%
%....%...%...%....%
%%%%.%%%.%.%%%.%%%%
%....%.......%....%
%.%%.%.%%G%%.%.%%.%
%....%.......%....%
%.%%.%.%%%%%.%.%%.%
%....%...%...%....%
%%%%.%%%.%.%%%.%%%%
%....%...%...%....%
%.%%.%.%%%%%.%.%%.%
%.................%
%.%%.%%%.%.%%%.%%.%
%........P........%
%.................%
%%%%%%%%%%%%%%%%%%%
```

### 2.2 크기 제약

`pacman_env/constants.py`에 grid 최대 크기가 정의되어 있습니다:

```python
MAX_GRID_H = 21
MAX_GRID_W = 21
MAX_FOOD_POSITIONS = 21 * 21 = 441
```

- 모든 맵은 (21, 21) 이하여야 함
- 더 작은 맵은 walls로 padding되어 state vector(901-dim)에 들어감
- state layout: `[2 pacman + 16 ghost + 441 food + 441 wall + 1 power_timer] = 901`

### 2.3 맵 검증

새 맵을 만든 뒤 random/human 플레이로 동작을 확인:

```bash
python scripts/play_random.py --layout layouts/train/my_new_map.txt
python scripts/play_human.py  --layout layouts/train/my_new_map.txt
```

### 2.4 이미 준비된 맵

| 경로 | 용도 | 크기 |
|---|---|---|
| `layouts/train/pacman_classic.txt` | 메인 학습 맵 | 19×21 |
| `layouts/train/medium_classic.txt` | 중간 크기 | — |
| `layouts/train/small_open.txt` | 디버깅용 작은 맵 | 11×11 |
| `layouts/train/corridor.txt` | 좁은 통로 토폴로지 | 15×15 |
| `layouts/eval/unseen_size.txt` | OOD eval (크기 변화) | — |
| `layouts/eval/unseen_topology.txt` | OOD eval (구조 변화) | — |

---

## 3. 환경 설정 파일 (`configs/env/`)

환경 hyperparameter는 YAML로 분리되어 있습니다.

```yaml
# configs/env/default.yaml
env:
  layout_file: "layouts/train/medium_classic.txt"
  ghost:
    num_ghosts: 1
    policy: "chase_stochastic"      # or "random"
    epsilon: 0.2                     # 확률적 noise
    speed_ratio: 1.0                 # ghost moves per Pac-Man step; 0.5 = every other step
    personality: "homogeneous"
  power_pellet:
    enabled: false                   # MVP에서는 비활성화
    frightened_duration: 30
  reward:
    pellet: 1.0
    power_pellet: 5.0
    ghost_eaten: 10.0
    death: -10.0
    win: 50.0
    sparse_remaining_pellet_penalty: 0.0
    dense_remaining_pellet_ratio_penalty: 0.0
  episode:
    max_steps: 500
  render_mode: null
```

---

## 4. 월드 모델 학습 — End-to-End

### 4.1 단계 요약

```
[1] layout 작성        → layouts/train/*.txt
[2] 데이터 수집        → scripts/collect_data.py     → data/replay/<name>/
[3] 월드모델 학습      → scripts/train_world_model.py → checkpoints/<name>/best.pt
[4] 평가              → scripts/eval_world_model.py + eval_policy_readiness.py
```

### 4.2 Step 1 — 데이터 수집

`scripts/collect_data.py`는 Pac-Man env를 직접 굴려서 transition을 NPZ 형식으로 저장합니다.

```bash
python scripts/collect_data.py \
    --layout layouts/train/pacman_classic.txt \
    --num-transitions 70000 \
    --policy mixed --p-greedy 0.1 \
    --output-dir data/replay/pacman_classic \
    --randomize-spawn --min-spawn-dist 2 \
    --num-ghosts 1 --ghost-epsilon 0.2 \
    --val-fraction 0.1 --seed 0
```

주요 옵션:

| 옵션 | 의미 |
|---|---|
| `--policy` | `random` 또는 `mixed`(가끔 pellet 쪽으로 greedy) |
| `--p-greedy` | mixed policy에서 greedy step의 비율 |
| `--randomize-spawn` | 매 episode마다 팩맨/고스트 시작 위치 랜덤 |
| `--val-fraction` | val split 비율 (e.g. 0.1 = 10%) |

출력 구조:
```
data/replay/pacman_classic/
  train/episode_000000.npz, ...
  val/episode_000000.npz, ...
```

각 `.npz`는 `states`(T+1, 901), `actions`(T,), `rewards`(T,), `dones`(T,)를 담습니다.

### 4.3 Step 2 — 월드 모델 학습

기본 학습 (fresh, K=1 single model):

```bash
python scripts/train_world_model.py \
    --data-dir data/replay/pacman_classic \
    --checkpoint-dir checkpoints/pacman_classic \
    --wandb --wandb-name fresh_v10c
```

**핵심 CLI 옵션** (모두 config의 값을 override):

| 옵션 | 의미 |
|---|---|
| `--config` | YAML 설정 (default: `configs/world_model/jepa_default.yaml`) |
| `--data-dir` | 데이터 디렉터리 |
| `--checkpoint-dir` | 체크포인트 저장 경로 |
| `--max-train-steps` | 최대 step (config default: 50000) |
| `--eval-every` | eval 주기 (config default: 500) |
| `--learning-rate` | LR (config default: 3e-4) |
| `--burnin-min` / `--burnin-max` | train-time burnin 범위 |
| `--beta-reward` / `--beta-done` | loss term weights |
| `--pos-weight-done` | done BCE의 positive class weight |
| `--seed` | 재현 가능성 |
| `--wandb` | WandB 로깅 활성화 (offline 가능) |

#### Resume + head fine-tuning

기존 체크포인트에서 backbone을 freeze하고 head만 학습:

```bash
python scripts/train_world_model.py \
    --data-dir data/replay/pacman_classic \
    --resume-from checkpoints/pacman_classic_v82/best.pt \
    --extract-member 0 \
    --freeze-dynamics \
    --max-train-steps 3000 --eval-every 200 \
    --learning-rate 1e-4 \
    --wandb --wandb-name v10b_head_only
```

`--freeze-dynamics`는 encoder, action_embedder, dynamics, dynamic_state_head를 freeze하고 `food_eaten_head` + `done_head`만 학습합니다 (`scripts/train_world_model.py:155`).

#### Low-LR full fine-tune (v10c 전략)

backbone까지 살짝 미세 조정하지만 LR을 낮춰서 안정화:

```bash
python scripts/train_world_model.py \
    --data-dir data/replay/pacman_classic \
    --resume-from checkpoints/pacman_classic_v82/best.pt \
    --extract-member 0 \
    --learning-rate 5e-5 \
    --checkpoint-dir checkpoints/pacman_classic_v10c \
    --wandb --wandb-name v10c_fullft_lr5e5
```

### 4.4 Step 3 — Loss 구성 (`configs/world_model/jepa_default.yaml`)

학습은 다음 loss term들의 합을 최소화합니다:

```
L_total = L_latent
        + β_reward     · L_reward
        + β_done       · L_done
        + β_var        · L_var
        + β_dyn_state  · L_dyn          # state reconstruction aux
        + β_count_δ    · L_count_delta  # (v9에서 시도, 현재 0)
        + β_food_eaten · L_food_eaten   # FoodEatenHead BCE (v10)
```

현재 default 값 (`jepa_default.yaml`):

```yaml
beta_reward: 1.0          # L_reward = MSE(symlog(r_pred), symlog(r_true))
beta_done: 1.0
beta_var: 0.1             # latent variance regularizer (collapse 방지)
beta_dynamic_state: 1.0   # DynamicStateHead aux
beta_count_delta: 0.0
beta_food_eaten: 1.0      # FoodEatenHead BCE (encoder+dynamics 양 path)
pos_weight_done: 5.0
target_std: 0.15
```

### 4.5 Step 4 — 평가

#### 표준 K-step MSE eval

```bash
python scripts/eval_world_model.py \
    --checkpoint checkpoints/pacman_classic_v10c/best.pt \
    --data-dir data/replay/pacman_classic \
    --split val \
    --k-step 10 --n-trajs 100
```

출력 metric:
- `k_step_latent_mse` — autoregressive z rollout 정확도
- `k_step_reward_mse` — symexp(predicted) vs raw reward
- `k_step_done_err` — |pred − true| done
- `sigma_mean` — ensemble disagreement (K=1이면 0)

**M4 milestone thresholds** (`jepa_default.yaml`):
- `latent_mse < 0.05`
- `reward_mse < 0.10`
- `done_err < 0.10`

#### Policy-readiness eval (정책 학습 적합성)

순위 보존 / 분류 가능성 / per-class separation을 추가로 측정:

```bash
python scripts/eval_policy_readiness.py \
    --checkpoint checkpoints/pacman_classic_v10c/best.pt \
    --data-dir data/replay/pacman_classic \
    --split val \
    --k-step 10 --n-trajs 100 \
    --warmup 5
```

추가 지표:
- `food_eaten_auc` — predicted reward로 food-event 분류 ROC-AUC
- `pearson_r`, `spearman_r` — pred vs true reward 상관계수
- `mean_pred_r_cls{0,1}` — food 안 먹음/먹음 클래스별 예측 reward 평균
- `separation_pred` vs `separation_true` — magnitude 보존 정도

### 4.6 Step 5 — 시각화 (선택)

학습된 모델의 imagination trajectory를 시각화:

```bash
python scripts/visualize_world_model.py \
    --checkpoint checkpoints/pacman_classic_v10c/best.pt \
    --layout layouts/train/pacman_classic.txt
```

---

## 5. Ground-Truth Pac-Man PPO Baseline (`pacman_rl/`)

`pacman_rl/`은 world model을 사용하지 않고 실제 `PacmanEnv`에서 PPO를 학습하는 baseline입니다. 관측은 world model output과 같은 901-dim state vector이고, reward는 `pacman_env/reward.py`의 `RewardComputer`가 반환하는 점수를 그대로 사용합니다.

Run output layout:

```text
logs/pacman_rl/<run_name>/
  train/
    events.out.tfevents...      # TensorBoard
    checkpoints/model_*.pt      # rsl_rl checkpoints
    videos/*.mp4                # train rollout videos, if --video
  play/
    *.mp4                       # playback videos, if --video
```

`--run-name`을 생략하면 현재 datetime이 run folder 이름으로 사용됩니다.

### 5.1 핵심 구현

| 파일 | 역할 |
|---|---|
| `pacman_rl/vec_env.py` | 여러 `PacmanEnv`를 rsl_rl `VecEnv` 인터페이스로 감싸는 adapter |
| `pacman_rl/discrete.py` | Pac-Man의 5개 discrete action을 위한 categorical distribution |
| `pacman_rl/train.py` | PPO 학습 entrypoint |
| `pacman_rl/play.py` | checkpoint evaluation / playback entrypoint |
| `pacman_rl/video.py` | headless video recording helper |
| `pacman_rl/configs/pacman_ppo.yaml` | PPO/env 기본 설정 |

Discrete action은 policy MLP가 5개 logit을 출력하고, `torch.distributions.Categorical`에서 action id를 sample하는 방식입니다:

| id | action |
|---|---|
| 0 | UP |
| 1 | DOWN |
| 2 | LEFT |
| 3 | RIGHT |
| 4 | NOOP |

### 5.2 학습

Smoke test:

```bash
python pacman_rl/train.py \
    --iterations 1 \
    --num-envs 2 \
    --device cuda \
    --headless
```

일반 학습:

```bash
python pacman_rl/train.py \
    --config pacman_rl/configs/pacman_ppo.yaml \
    --device cuda \
    --headless \
    --run-name gt_ppo_medium
```

TensorBoard:

```bash
tensorboard --logdir logs/pacman_rl/gt_ppo_medium/train
```

학습 중 rollout video 저장:

```bash
python pacman_rl/train.py \
    --device cuda \
    --headless \
    --run-name gt_ppo_medium_video \
    --video \
    --video-every 100
```

### 5.3 플레이 / 평가

```bash
python pacman_rl/play.py \
    --checkpoint logs/pacman_rl/gt_ppo_medium/train/checkpoints/model_50.pt \
    --episodes 5 \
    --device cuda \
    --headless \
    --video
```

`--headless`를 빼면 가능한 환경에서 Pygame 창으로 직접 볼 수 있습니다. `--video`를 켜면 run folder 아래에 `.mp4` 파일을 저장합니다.

### 5.4 테스트

```bash
python -m pytest -q tests/test_rl_discrete.py tests/test_rl_vec_env.py
```

---

## 6. 디렉터리 구조

```
.
├── configs/
│   ├── env/                          # 환경 hyperparameter YAML
│   └── world_model/jepa_default.yaml # WM 학습 설정
├── layouts/
│   ├── train/                        # 학습용 맵
│   └── eval/                         # OOD 평가용 맵
├── pacman_env/                       # gym 환경 (env, state, ghost, reward)
├── pacman_rl/                        # GT Pac-Man PPO baseline (rsl_rl)
│   ├── train.py                      # PPO 학습 entry
│   ├── play.py                       # checkpoint play/eval
│   ├── vec_env.py                    # rsl_rl VecEnv adapter
│   ├── discrete.py                   # categorical action distribution
│   ├── video.py                      # mp4 recording helper
│   └── configs/pacman_ppo.yaml       # PPO/env config
├── world_model/
│   ├── single.py                     # SingleWorldModel
│   ├── ensemble.py                   # EnsembleWorldModel wrapper
│   ├── loss.py                       # compute_world_model_loss
│   ├── eval.py                       # k-step rollout eval
│   ├── replay_buffer.py              # SequenceReplayBuffer
│   └── modules/                      # encoder, dynamics, action, heads
├── scripts/
│   ├── collect_data.py               # 데이터 수집
│   ├── train_world_model.py          # 학습 entry
│   ├── eval_world_model.py           # 표준 eval
│   ├── eval_policy_readiness.py      # 정책 학습 적합성 eval
│   ├── visualize_world_model.py
│   ├── diagnose_reward.py            # reward path 진단
│   └── verify_reward_limit.py        # sigmoid-sum 정밀도 진단
├── checkpoints/                      # 학습된 체크포인트
└── world_model_implementation.md     # WM 설계 진화 history
```

---

## 7. 빠른 reproduce 레시피

마지막 best 체크포인트(`v10c`)를 처음부터 재현하려면:

```bash
# 1) 데이터 수집 (70k transitions, mixed policy)
python scripts/collect_data.py \
    --layout layouts/train/pacman_classic.txt \
    --num-transitions 70000 --policy mixed --p-greedy 0.1 \
    --output-dir data/replay/pacman_classic \
    --randomize-spawn

# 2) v8.2 baseline (50k steps, full training)
python scripts/train_world_model.py \
    --data-dir data/replay/pacman_classic \
    --checkpoint-dir checkpoints/pacman_classic_v82 \
    --wandb --wandb-name v82_baseline

# 3) v10c — FoodEatenHead 추가 + low-LR full fine-tune
python scripts/train_world_model.py \
    --data-dir data/replay/pacman_classic \
    --resume-from checkpoints/pacman_classic_v82/best.pt \
    --extract-member 0 \
    --learning-rate 5e-5 \
    --checkpoint-dir checkpoints/pacman_classic_v10c \
    --wandb --wandb-name v10c

# 4) 평가
python scripts/eval_policy_readiness.py \
    --checkpoint checkpoints/pacman_classic_v10c/best.pt \
    --data-dir data/replay/pacman_classic \
    --warmup 5
```
