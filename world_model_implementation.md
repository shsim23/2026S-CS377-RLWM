# Pac-Man 월드 모델 구현 정리

이 문서는 Pac-Man 강화학습용 **월드 모델(World Model)**의 현재 구현을 정리합니다.
처음 보는 사람도 이해할 수 있도록:

1. **월드 모델이 무엇이고 무엇을 예측하는가**
2. **구현이 어떻게 진화해왔는가** (큰 맥락)
3. **현재 최종 아키텍처** (모듈/loss/forward 단위 상세)
4. **사용 방법** (학습 / 데이터 수집 / 평가 / 정책 학습 인터페이스)
5. **주의사항** (보상 스케일, 적용 범위)

---

## 0. 월드 모델이란?

월드 모델은 **환경의 시뮬레이터를 신경망으로 학습한 것**입니다. 실제 게임 엔진을 돌리지 않고도,
`(상태, 행동)` 시퀀스가 주어지면 **다음 상태 · 보상 · 종료여부**를 예측합니다.

```
(state_t, action_t) ──[World Model]──▶ 다음 latent 상태, reward_t, done_t
```

이걸 쓰면 정책(policy)을 **상상(imagination)** 속에서 학습할 수 있습니다 — 실제 환경 롤아웃 없이
모델이 만들어낸 가상의 궤적으로 PPO 등을 돌리는 것이 최종 목표입니다.

> **이 월드 모델의 범위(scope):** 월드 모델 *만* 구현합니다. actor/critic/value/planning은 포함하지
> 않습니다. "주어진 (상태, 행동)에서 다음 상태·보상·종료를 예측하고, 관측 없이 open-loop로 상상 롤아웃을
> 할 수 있다"가 전부입니다.

### 상태 표현 (901차원 벡터)

환경은 게임 화면을 다음 901차원 벡터로 인코딩합니다 (`pacman_env.state.StateBuilder`):

| 구간 | 차원 | 인덱스 | 내용 |
|---|---|---|---|
| Pacman 위치 | 2 | `[0:2]` | (x, y), [-1,1] 정규화 |
| Ghost 슬롯 | 16 | `[2:18]` | 고스트 4마리 × (x, y, alive, valid) |
| Food 마스크 | 441 | `[18:459]` | 21×21 격자, 각 칸에 펠릿 유무 (0/1) |
| **Wall 마스크** | 441 | `[459:900]` | 21×21 격자, 벽 유무 (0/1) — **에피소드 내내 고정** |
| Power timer | 1 | `[900:901]` | 파워펠릿 타이머 (small_open에선 항상 0) |

핵심: **Wall 마스크(441)는 에피소드 동안 변하지 않는 정적 정보**입니다. 그래서 매 스텝 재구성하는 대신,
**레이아웃 임베딩 `e`** 로 한 번만 인코딩해서 모델에 조건으로 넣습니다. 나머지 **동적(dynamic) 460차원**
`(pacman 2 + ghost 16 + food 441 + power 1)` 만 모델이 예측/재구성합니다.

---

## 1. 구현 진화의 큰 흐름

이 월드 모델은 다음 4단계를 거쳐 현재에 도달했습니다.

```
 [1] JEPA 구현            [2] DreamerV3 참고 구현      [3] two-hot 구현            [4] GRU 8-block + 모델 스케일업
 (legacy/baseline)   ──▶  (현재 코드베이스 토대)  ──▶  (위치/보상 예측 개선)  ──▶  (현재 최종, model B)
```

### [1] JEPA 구현 (legacy baseline)
- 직접 설계한 JEPA(Joint-Embedding Predictive Architecture) 스타일: encoder가 상태를 latent `z`로
  압축하고, `sg(z_target)`을 self-supervised 타깃으로 써서 **디코더 없이** dynamics를 학습.
- 보상 학습이 잘 안 돼서 `DynamicStateHead`(상태 재구성 aux), 학습된 RewardHead 폐기 →
  결정론적 보상, `FoodEatenHead`(펠릿 섭취를 binary 분류) 등으로 우회.
- 한계: reward MSE가 천장(plateau)에 갇히고, latent collapse 방지를 위한 variance regularizer 같은
  땜질이 많이 필요했음. 이 구현은 `world_model/single.py`에 ablation baseline으로 남아있습니다.

### [2] DreamerV3 참고 구현 (현재 코드베이스 토대)
- JEPA의 한계를 벗어나기 위해 **DreamerV3 논문의 RSSM 구조를 충실히 재구현** (`world_model/dreamer/`).
- 핵심 변경:
  - **범주형(categorical) latent** (32 groups × N classes, straight-through, 1% unimix)
  - **KL balancing + free-bits** 목적함수 → variance regularizer 없이 collapse 방지
  - RMSNorm + SiLU, symlog 입력 압축
  - **레이아웃 조건화**: wall_mask → `e` 임베딩 → 시퀀스 모델에 주입 (맵 간 일반화 목적)
- 단, 표준 DreamerV3와 다르게: **CNN 대신 MLP**(상태 기반), 디코더는 동적 460차원만 재구성.

### [3] two-hot 구현 (위치/보상 예측 개선)
- DreamerV3의 **two-hot symlog 보상 헤드**를 채택 (보상을 255개 bin에 대한 분포로 예측).
- 추가로, **엔티티 위치(PositionHead)도 two-hot 격자 분류로** 전환:
  - 문제: pacman/ghost 좌표 10차원이 460차원 재구성 MSE 안에서 441개 food BCE에 묻혀 거의 학습 안 됨.
  - 해결: 10개 좌표를 각각 **21개 격자 칸에 대한 범주형 분류**로 예측 → 위치 1-step 오차 대폭 감소,
    "벽 위에 있음" 같은 오류 ~0%.

### [4] GRU 8-block 업데이트 + 모델 스케일업 (현재 최종)
- 단일 에이전트 데이터에 overfit하는 문제를 **mix12 커리큘럼 데이터**(PPO 체크포인트 12개 혼합)로 완화.
- **모델 용량을 DreamerV3 `size12m` 프리셋으로 확장**: `deter=2048`.
- 문제: `deter=2048`에서 일반 GRU는 recurrent 가중치가 폭발 (~19M 파라미터).
- 해결: **Block-diagonal GRU (`blocks=8`)** — recurrent(h→h) 변환을 8개 블록으로 쪼개 대각 블록만 사용.
  recurrent 파라미터를 ~8배 줄여, 넓은 recurrent 상태를 유지하면서도 전체 **~8M**으로 억제.
- 이 단계의 결과물이 현재 권장 모델 **model B** (`size12m` + blocked GRU)입니다.

> **참고:** [3]까지는 model A(3.36M, full GRU, deter=256) 계열이고, [4]에서 model B(7.95M)가 만들어졌습니다.
> A/B 비교 결과 **B가 거의 모든 지표에서 우세**합니다 (§4 참조).

---

## 2. 현재 아키텍처 (상세)

코드 위치: `world_model/dreamer/` 패키지.

### 2.1 전체 구조도

```
                          wall_mask[441] ──[LayoutEmbedder]──▶ e[32]  (에피소드당 1회, 시간축 broadcast)
                                                                 │
   각 시점 t에서:                                                  │  (조건)
                                                                 ▼
   h_{t-1}, z_{t-1}, a_{t-1} ───────────────[SequenceModel (Block GRU)]──────────▶ h_t  (deterministic 상태)
                                                                                    │
                            ┌───────────────────────────────────────────────────────┤
                            │                                                       │
                  관측 있음 │ (posterior, 학습/encode)              관측 없음 │ (prior, imagine)
                            ▼                                                       ▼
   state_t[901] ──[Encoder]──▶ post_logits ──▶ z_t          [DynamicsPredictor]──▶ prior_logits ──▶ ẑ_t
        (q(z|h,x))           (32 groups ×          (sample_st)   (p(z|h))                    (sample_st)
                              classes)
                            │
                            ▼ z_t (straight-through one-hot, 1% unimix)
        ┌───────────────────┼─────────────────────────┬──────────────────────┐
        ▼                   ▼                         ▼                      ▼
   [Decoder]          [RewardHead]              [ContinueHead]         [PositionHead]
   (h,z)→460d        (h,z)→255 bins             (h,z)→1 logit          (h,z)→10×21 bins
   동적상태 재구성    two-hot symlog 보상         done(1-c) 확률         엔티티 좌표(격자 분류)
```

### 2.2 6개 RSSM 컴포넌트 (+ PositionHead)

DreamerV3 RSSM의 6개 구성요소를 그대로 따릅니다 (`world_model/dreamer/rssm.py`):

| 컴포넌트 | 수식 | 역할 | 비고 |
|---|---|---|---|
| **SequenceModel** | `h_t = f(h_{t-1}, z_{t-1}, a_{t-1}, e)` | recurrent 상태 갱신 | Block GRU (§2.3) |
| **Encoder** (posterior) | `z_t ~ q(z\|h_t, x_t)` | **관측을 보고** latent 추론 | 학습·encode 시 |
| **DynamicsPredictor** (prior) | `ẑ_t ~ p(ẑ\|h_t)` | **관측 없이** latent 예측 | imagine 시 |
| **RewardHead** | `r̂_t ~ p(r\|h_t, z_t)` | 보상 예측 (two-hot symlog) | §2.4 |
| **ContinueHead** | `ĉ_t ~ p(c\|h_t, z_t)` | continue=1−done (Bernoulli) | logit 출력 |
| **Decoder** | `x̂_t ~ p(x_dyn\|h_t, z_t)` | 동적 460차원 재구성 | 2-layer MLP |
| **PositionHead** | `좌표_t ~ p(coord\|h_t, z_t)` | 엔티티 10좌표 격자 분류 | two-hot 모드 전용 (§2.5) |

공통 빌딩블록(`world_model/dreamer/nn.py`): **RMSNorm + SiLU** MLP, **symlog** 입력 압축,
**straight-through 범주형 샘플링**(1% unimix).

#### latent `z`의 구조
- `z`는 연속 벡터가 아니라 **범주형 그룹들의 묶음**입니다: `(groups, classes)` 모양.
  - model A: 32 × 32 = 1024차원, model B: 32 × 16 = 512차원
- 각 그룹은 one-hot으로 샘플되고(straight-through로 역전파 가능), 1% unimix(균등분포 1% 섞기)로
  죽은 클래스를 방지합니다.
- collapse(붕괴) 방지는 **범주형 latent + KL free-bits**가 담당 — legacy 모델의 variance regularizer가
  더 이상 필요 없습니다.

### 2.3 Block-diagonal GRU (현재의 핵심 변경)

`deter`(recurrent 상태 h의 차원)를 2048로 키울 때, 일반 `nn.GRUCell`의 recurrent 가중치는 `deter²`에
비례해 폭발합니다. **BlockGRUCell**(`rssm.py`)은 h→h 변환만 **블록 대각(block-diagonal)**으로 만듭니다:

- h를 `blocks=8`개 블록으로 나누고, 블록 k의 recurrent 기여는 **오직 자기 블록 k에서만** 옵니다.
- 입력(input→h) 변환은 그대로 dense.
- 결과: recurrent 파라미터 ~8배 절감. `deter=2048` full GRU ≈ 19M → blocked ≈ 8M.
- `gru_blocks=1`이면 일반 GRU와 완전히 동일 (하위호환).

```python
# rssm.py BlockGRUCell.forward (핵심)
gx = self.in_w(x).view(B, blocks, 3*bs)                 # 입력: dense
gh = torch.einsum("bkc,kcd->bkd", hb, self.h_w) + bias  # recurrent: 블록 대각
r = sigmoid(gx_r + gh_r); z = sigmoid(gx_z + gh_z)
n = tanh(gx_n + r * gh_n)
h_new = (1 - z) * n + z * hb                            # 표준 GRU 게이팅
```

### 2.4 Two-hot symlog 보상 헤드

보상을 스칼라로 회귀하지 않고 **분포로 예측**합니다 (DreamerV3 §5.1):

- 보상을 `symlog`(부호보존 로그) 공간으로 옮긴 뒤, `[-20, 20]`을 **255개 bin**으로 균등 분할.
- 헤드는 255개 bin에 대한 logit을 출력. 타깃은 실제 보상을 가장 가까운 두 bin에 나눠 담은 **two-hot** 라벨.
- 추론: `softmax → 기대값(Σ p·bin) → symexp` 로 스칼라 보상 복원 (`reward_from_logits`).
- 출력층은 0으로 초기화 → 초기 보상 예측이 0에서 시작 (symexp(0)=0).

### 2.5 Two-hot 위치 헤드 (PositionHead)

엔티티 좌표 10개 = `pacman(x,y)` + `ghost 4마리 ×(x,y)` (동적 벡터 인덱스 `[0,1,2,3,6,7,10,11,14,15]`)를
각각 **21개 격자 칸에 대한 범주형 분류**로 예측합니다.

- bin은 정규화 좌표 `[-1,1]`의 21개 격자 중심에 위치 → 정확한 칸은 two-hot이 한 bin에 몰림(사실상 one-hot 분류).
- 왜? 좌표 10차원은 460차원 재구성 MSE에서 441개 food BCE에 묻혀 거의 학습이 안 됐음. 격자 분류로
  바꾸니 위치 1-step 오차 급감 + 그래디언트가 binary 블록에 죽지 않음.
- two-hot 모드(`position_mode=twohot`)에서 좌표는 재구성 연속블록에서 빠지고, 연속블록엔 `power`만 남습니다.

### 2.6 Forward 경로

#### (a) `observe(...)` — teacher-forced posterior 롤아웃 (학습/평가용)
실제 관측을 매 스텝 보면서 latent을 추론. `actions[:,t]`는 `states[:,t]`로 *이끈* 행동(a_{t-1}),
`is_first[:,t]`는 에피소드 경계에서 recurrent carry를 리셋.

```python
h, z = initial_state()
for t in range(L):
    h = seq(h, flat(z), action_t, e_t)          # SequenceModel
    prior_logits = prior(h)                      # 관측 없는 예측 (KL용)
    post_logits  = encoder(h, state_t)           # 관측 본 추론
    z = OneHotCategoricalST(post_logits).sample_st()
# 이후 h,z 시퀀스로 decoder/reward/continue/position 헤드 일괄 적용
```

#### (b) `imagine_step(...)` — prior 스텝 (관측 없는 상상, 정책 학습용)
```python
h_next = seq(h, flat(z), a, e)
z_next = OneHotCategoricalST(prior(h_next)).sample_st()   # 관측 대신 prior에서 샘플
reward = reward_from_logits(reward_head(h_next, z_next))
cont   = sigmoid(cont_head(h_next, z_next))
```

### 2.7 학습 목적함수 (loss)

`world_model/dreamer/loss.py`:

```
L = β_pred · L_pred  +  β_dyn · L_dyn  +  β_rep · L_rep
    β_pred = 1.0,        β_dyn = 0.5,      β_rep = 0.1

L_pred = 재구성 NLL + 보상 NLL + continue NLL
       - 재구성: 연속 dim = symlog+MSE, binary dim(food+ghost flag) = BCE,
                 위치 dim = two-hot 교차엔트로피 (beta_cont로 가중)
       - 보상: two-hot 교차엔트로피 (symlog 공간)
       - continue: BCE
L_dyn  = max(1, KL[ sg(post) ‖ prior ])    # prior(상상 경로)를 posterior로 끌어당김
L_rep  = max(1, KL[ post ‖ sg(prior) ])    # posterior를 prior 쪽으로 정규화
```

- **Free bits = 1 nat**: 각 KL을 1 nat 아래로는 내리지 않음 (과도한 정규화로 인한 collapse 방지).
- **`sg`** = stop-gradient. KL을 양방향(dyn/rep)으로 쪼개 prior와 posterior를 각각 학습 (DreamerV3 KL balancing).
- **`context`(=8) 워밍업 스텝은 모든 loss에서 제외** — h_0를 실제 히스토리로 데우는 구간.
- KL annealing 없음, weight decay 없음, dropout 없음, **variance regularizer 없음**.

### 2.8 최적화

- **LaProp** 옵티마이저 + **AGC**(Adaptive Gradient Clipping, `agc_clip=0.3`) (`world_model/dreamer/optim.py`)
- learning rate `1e-4`, betas `(0.9, 0.999)`

---

## 3. 하이퍼파라미터 (model A / model B)

| 항목 | model A (`small_open_3m.yaml`) | **model B (`small_open_12m.yaml`)** |
|---|---|---|
| 파라미터 수 | 3,363,342 (~3.36M) | **7,954,446 (~7.95M)** |
| latent (groups×classes) | 32×32 = 1024 | 32×16 = 512 |
| `deter` (h 차원) | 256 | **2048** |
| `gru_blocks` | 1 (full GRU) | **8 (block-diagonal)** |
| `hidden` (MLP) | 256 | 256 |
| `e_dim` / `action_emb` | 32 / 16 | 32 / 16 |
| 보상 헤드 | two-hot, 255 bins, `[-20,20]` | 동일 |
| `position_mode` / `pos_bins` | twohot / 21 | 동일 |
| `unimix` | 0.01 | 0.01 |
| β_pred/β_dyn/β_rep | 1.0 / 0.5 / 0.1 | 동일 |
| free_nats | 1.0 | 1.0 |
| optimizer | LaProp + AGC(0.3), lr 1e-4 | 동일 |
| batch_size | 32 | 32 |
| seq_length / context | 64 / 8 | 64 / 8 |
| max_train_steps | 20,000 | 50,000 |
| k_step / n_eval_windows | 32 / 64 | 32 / 64 |

두 모델은 **데이터·loss·twohot 레시피가 동일**하고 용량(A vs B)만 다릅니다 — 깔끔한 A/B 비교용.

---

## 4. A/B 비교 결과 — 어느 모델을 쓰나

데이터셋 `small_open_mix12_newrwd`(신규 보상, mix12 커리큘럼)에서 두 모델 `best.pt` 평가:

| 지표 | model A (3.36M) | **model B (7.95M, 권장)** |
|---|---|---|
| 1-step 재구성 정확도 (recon_bin_acc) | 0.99989 | **0.99999** |
| 1-step 보상 MSE (head) | 57.2 | **17.5** |
| 1-step continue 정확도 | 0.994 | **0.999** |
| 디코딩 상태 품질 (gt_done 1step MSE) | 3.20 | **0.93** |
| latent 붕괴 그룹 | 0/32 | 0/32 |

→ **model B가 우세.** 최종 권장 체크포인트:

```
checkpoints/dreamer_wm/small_open_mix12_newrwd_modelB_12m/
├── best.pt     ← 권장 (step 48000, 검증 최고점)
└── latest.pt   (step 50000, 마지막 — best보다 검증점수 약간 나쁨)
```

---

## 5. 사용 방법

### 5.1 체크포인트 로드

체크포인트에 `cfg`가 들어있어 별도 config 없이 그대로 복원됩니다.
단, **model B를 불러오려면 BlockGRUCell 코드가 있는 최신 `world_model/dreamer/`가 필요**합니다.

```python
import torch
from world_model.dreamer import DreamerWorldModel, WorldModelConfig

ck = torch.load("checkpoints/dreamer_wm/small_open_mix12_newrwd_modelB_12m/best.pt",
                map_location="cuda", weights_only=False)
model = DreamerWorldModel(WorldModelConfig(**ck["cfg"])).cuda()
model.load_state_dict(ck["model"])
model.eval()
```

### 5.2 학습

```bash
PY=/home/ubuntu/miniconda/envs/pacman-wm/bin/python
$PY scripts/wm_train_dreamer.py \
    --dataset small_open_mix12_newrwd \
    --config configs/world_model/small_open_12m.yaml \
    --checkpoint-dir checkpoints/dreamer_wm/small_open_mix12_newrwd_modelB_12m \
    --device cuda --seed 0 --layout-id 0 --save-every 5000
```

데이터 수집 → A/B 학습까지 한 번에 도는 파이프라인: `scripts/wm_two_model_pipeline.sh`.
데이터 수집만: `scripts/wm_collect_mix_checkpoints.py` (PPO 체크포인트 12개를 섞어 mix12 데이터셋 생성).

### 5.3 평가

```bash
# (1) intrinsic 메트릭 (학습 때와 동일: recon/reward/continue/k-step)
$PY scripts/wm_eval_dreamer.py \
    --checkpoint checkpoints/dreamer_wm/small_open_mix12_newrwd_modelB_12m/best.pt \
    --dataset small_open_mix12_newrwd \
    --config configs/world_model/small_open_12m.yaml --device cuda

# (2) 보상: 학습된 head vs 디코딩 상태에서 규칙 계산한 보상 비교 + done 정확도
$PY scripts/wm_eval_reward_decoded.py \
    --checkpoint checkpoints/dreamer_wm/small_open_mix12_newrwd_modelB_12m/best.pt \
    --datasets small_open_mix12_newrwd --context 8 --horizon 32 --device cuda
#   ※ 보상 스케일은 --reward-{pellet,win,death}로 덮어쓸 수 있음 (기본 10/200/-100, §6 참조)
```

### 5.4 정책 학습 인터페이스 (상상 롤아웃)

`DreamerWorldModel`은 정책 단계용 인터페이스를 제공합니다:

```python
# 관측을 보며 latent 추론 (롤아웃 시작점)
h, z = model.encode(x=state0, h=h0, a_prev=a0, e=model.embed_layout(state0))

# 이후 관측 없이 K스텝 상상
for t in range(K):
    out = model.imagine_step(h, z, a_t, e)
    # out["h"], out["z_next"], out["reward"], out["cont"]
    h, z = out["h"], out["z_next"]
```

---

## 6. 주의사항 (꼭 읽기)

### 6.1 보상 스케일 — 신규(latest) 버전
`*_newrwd` 모델은 **`pacman_rl/configs/pacman_ppo.yaml`의 신규 보상 스케일**로 학습됐습니다:

| 이벤트 | 값 |
|---|---|
| 펠릿 1개 | +10 |
| 승리(마지막 펠릿 포함) | +200 (실제 보상 +210) |
| 죽음 | −100 |
| 타임아웃(절단) | −50 (전체 done의 0.003%로 무시 가능) |

> ⚠️ 과거에 공유된 체크포인트(`small_open_mix12_twohot` 등)는 **구버전 fold-in 스케일**(펠릿=1, 승리=50,
> 죽음=−10)이었습니다. 신규 모델은 보상 출력이 ~10배 큰 스케일이니, 다운스트림(PPO 보상 처리/임계값/정규화)이
> 구버전 가정이면 어긋납니다. **현재 `pacman_ppo.yaml`을 그대로 쓰면 일관됩니다.**

### 6.2 imagination 보상은 reward_head 대신 decoded_rules 권장
- 학습된 reward head는 **1-step은 정확**하지만(MSE 17, R²≈0.91; MAE 0.28), **k-step open-loop에서 발산**하고
  희귀 종료보상(죽음/승리)을 과소예측합니다.
- 디코딩된 상태 + 규칙으로 계산한 보상(decoded_rules)은 k-step에서도 안정적 (MSE ~17).
- `pacman_ppo.yaml`이 이미 `reward_source: decoded_rules`로 설정되어 있습니다 — 이대로 쓰세요.

### 6.3 적용 범위 — small_open 단일 맵 전용
- 현재 공유 모델은 **small_open(11×11, 고스트 1마리) 단일 레이아웃에 overfit**되어 있습니다.
  다른 레이아웃엔 일반화되지 않습니다.
- 환경 설정을 맞춰야 함: `num_ghosts=1, ghost epsilon=0.4, speed_ratio=0.6, min_spawn_dist=5,
  max_steps=700, power_pellet disabled`.
- 아키텍처는 레이아웃 조건화(`e` 임베딩)를 지원하므로 다맵 학습은 가능하나, 그건 별도 데이터/학습이 필요합니다.

---

## 7. 핵심 파일 위치

| 컴포넌트 | 파일 |
|---|---|
| `DreamerWorldModel`, `WorldModelConfig` | `world_model/dreamer/world_model.py` |
| RSSM 컴포넌트 (SequenceModel, **BlockGRUCell**, Encoder, Prior, Heads, Decoder, PositionHead) | `world_model/dreamer/rssm.py` |
| NN 프리미티브 (RMSNorm, MLP, 범주형 ST, two-hot 인코딩/디코딩, symlog) | `world_model/dreamer/nn.py` |
| 학습 목적함수 (`compute_loss`) | `world_model/dreamer/loss.py` |
| intrinsic 평가 (`evaluate`) | `world_model/dreamer/eval.py` |
| 옵티마이저 (LaProp + AGC) | `world_model/dreamer/optim.py` |
| 시퀀스 리플레이 | `world_model/dreamer/replay.py` |
| 학습 스크립트 | `scripts/wm_train_dreamer.py` |
| 데이터 수집 (mix12) | `scripts/wm_collect_mix_checkpoints.py` |
| 평가 스크립트 | `scripts/wm_eval_dreamer.py`, `scripts/wm_eval_reward_decoded.py` |
| 2-모델 A/B 파이프라인 | `scripts/wm_two_model_pipeline.sh` |
| legacy JEPA baseline (ablation) | `world_model/single.py`, `world_model/modules/` |
