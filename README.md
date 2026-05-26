# 2026S CS377 — RL with World Models

Pac-Man 환경에서 **JEPA-style world model**을 학습하고, 그 위에서 모델 기반 policy를 학습하는 프로젝트입니다.

이 문서는 두 가지를 다룹니다:
1. **맵(layout) 작성 방법**
2. **월드 모델 학습 방법** (데이터 수집 → 학습 → 평가)

월드 모델의 설계가 어떻게 진화해왔는지(reward head 제거, dynamic state head 추가, FoodEatenHead 도입 등)는 [`world_model_implementation.md`](world_model_implementation.md)를 참고하세요.

---

## 1. 환경 설치

```bash
# conda env (CUDA 포함). 이미 만들어진 환경: pacman-wm
conda env create -f environment.yml -n pacman-wm
conda activate pacman-wm
```

모든 명령은 conda env `pacman-wm`을 활성화한 상태에서 실행해야 합니다.

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

## 2.5 무작위 맵 자동 생성 (`maze_generator/`)

World model이 *general* state transition을 학습하도록, per-map training 대신 다양한 21×21 맵을 자동 생성해서 데이터로 쓸 수 있습니다. 설계 사양은 [`maze_generator_spec.md`](maze_generator_spec.md)를 참고하세요.

### 2.5.1 한 줄 사용

```python
from maze_generator import generate_maze, ascii_render, render_image

maze = generate_maze(seed=42)        # dict (spec §7.2 형식)
print(ascii_render(maze))            # 콘솔에 ASCII로 보기
render_image(maze, "out.png")        # PNG로 저장
```

반환되는 `maze` dict의 주요 필드:

| 키 | 형식 | 의미 |
|---|---|---|
| `walls` | `np.ndarray (21,21) bool` | True = 벽 |
| `pacman_pos` | `(row, col)` | 항상 `(14, 10)` |
| `ghost_positions` | `list[(r, c)]` | ghost house 내부 좌표 |
| `food_positions` | `list[(r, c)]` | 펠릿 위치들 (variable length) |
| `ghost_only_tiles` | `list[(r, c)]` | gate 좌표 |
| `ghost_house_interior` | `list[(r, c)]` | 고스트 하우스 walkable |
| `seed`, `width`, `height` | — | 메타 |
| `_tile_grid` | `np.ndarray (21,21) int8` | 디버그용 raw 타일 코드 (벽=0, path=1, gate=2, interior=3) |

### 2.5.2 주요 파라미터

```python
generate_maze(
    connectivity = 0.3,   # cycle 밀도 [0=tree, 1=fully connected]
    num_ghosts   = 1,     # 1, 2, 3 — 고스트 하우스 안에 자동 배치
    seed         = None,  # 재현용. None이면 매번 새 맵
)
```

`width`/`height`/`symmetric`/`ghost_house`는 Phase 1에서 고정. Warp tunnel과 power pellet (`num_warp_tunnels`, `num_power_pellets`)은 Phase 3/4용 인자만 마련돼 있고 0 외 값을 주면 `NotImplementedError`.

### 2.5.3 CLI 데모 — 예시 맵 일괄 생성

10개 시드로 PNG/ASCII를 한 번에 만드는 데모 스크립트:

```bash
# stdout에 ASCII로 10개 출력
python -m maze_generator.demo

# PNG + .txt를 저장 (+ 컨택트 시트 all_seeds.png)
python -m maze_generator.demo --save-dir maze_generator/examples

# 더 dense하게, 시드 100부터 20개, 고스트 3마리
python -m maze_generator.demo \
    --count 20 --start-seed 100 \
    --connectivity 0.6 --num-ghosts 3 \
    --save-dir maze_generator/examples_dense
```

기본 예시는 [`maze_generator/examples/`](maze_generator/examples/)에 이미 들어 있습니다 (`maze_seed000.png` ~ `maze_seed009.png`, `all_seeds.png`).

### 2.5.4 보장사항

- **결정성**: 같은 `seed` → 항상 같은 맵 (`random.Random(seed)` 사용, 전역 RNG 안 건드림).
- **좌우 대칭**: `col=10` 축 기준 완전 대칭.
- **고정 영역**: 팩맨 시작 (14, 10), 고스트 하우스 (row 9–11, col 8–12) 모든 맵에서 동일.
- **No dead-end**: 모든 PATH 셀이 이웃 PATH ≥ 2개.
- **연결성**: BFS 단일 connected component, 고스트 하우스에서 gate 통과해 외부 도달 가능.

검증은 `maze_generator.validator.validate(grid)`로 직접 다시 돌릴 수 있고, 생성 단계에서 실패하면 다른 seed로 최대 5회 재시도합니다 (21×21에선 1회로 항상 성공해야 함).

### 2.5.5 기존 layout 포맷과의 관계

현 generator는 ASCII layout file을 만들지 **않고**, 환경에 바로 먹일 dict를 반환합니다. 기존 `pacman_env.LayoutParser`와 호환되는 ASCII로 dump하고 싶다면 별도 어댑터가 필요합니다 (Phase 2 이후 통합 예정).

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
    step_penalty: -0.01
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

## 5. 디렉터리 구조

```
.
├── configs/
│   ├── env/                          # 환경 hyperparameter YAML
│   └── world_model/jepa_default.yaml # WM 학습 설정
├── layouts/
│   ├── train/                        # 학습용 맵 (수작업 ASCII)
│   └── eval/                         # OOD 평가용 맵
├── maze_generator/                   # 21x21 랜덤 맵 생성기 (§2.5)
│   ├── generator.py                  # generate_maze() API + 6-stage 파이프라인
│   ├── carving.py                    # randomized DFS carving (left half)
│   ├── post_process.py               # dead-end 제거 + food 배치
│   ├── validator.py                  # 연결성/대칭/dead-end 검증
│   ├── visualizer.py                 # ASCII + matplotlib PNG 렌더
│   ├── demo.py                       # python -m maze_generator.demo
│   └── examples/                     # 시드 0~9 PNG/ASCII 샘플
├── pacman_env/                       # gym 환경 (env, state, ghost, reward)
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

## 6. 빠른 reproduce 레시피

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
