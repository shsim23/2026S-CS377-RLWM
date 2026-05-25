# 월드 모델 구현 진화 (v0 → v10c)

이 문서는 Pac-Man 월드 모델 구현이 **Dreamer 기반 baseline**에서 시작해서 현재의 **JEPA + FoodEatenHead** 구조까지 어떻게 발전해왔는지를 정리합니다.

각 버전마다:
- **What**: 무엇을 바꿨는가
- **Why**: 어떤 문제 때문에 바꿨는가
- **Result**: 결과 (k-step reward MSE / latent MSE / done err)

마지막 섹션에서는 **현재(v10c) 최종 구조**를 모듈/loss/forward path 단위로 정리합니다.

---

## 0. 출발점 — Dreamer-style baseline (v0)

DreamerV2 구조를 그대로 따른 초기 설계:

```
state ──Encoder──▶ z_t ──Dynamics(GRU)──▶ z_{t+1}, h_{t+1}
                                           │
                            ┌──────────────┼──────────────┐
                            ▼              ▼              ▼
                        RewardHead     DoneHead       (sampled latent)
                       (z_next, h)   (z_next, h)
```

### Loss (v0)

```
L_total = L_latent + β_r·L_reward + β_d·L_done + β_var·L_var

L_latent = MSE(z_pred, sg(z_target))            # JEPA-style 자기지도
L_reward = MSE(symlog(r_pred), symlog(r_true))
L_done   = BCE(d_pred, d_true)
L_var    = ReLU(target_std − std(z))            # 코드 collapse 방지
```

### v0의 문제점

| 증상 | 수치 |
|---|---|
| Reward MSE가 **22 부근에서 멈춤** | mean predictor 수준 (food 비율 1/N) |
| Latent collapse 일부 발생 | dead_dims 22/128 |
| Done err는 OK | < 0.05 |
| post-1k step에서 발산 | 학습 후반 spike |

핵심 진단: **RewardHead가 `(z_next, h)`만 받으면 "food를 먹었는가"를 표현할 수 없음**.
food eaten 신호 = `food_count_t − food_count_{t+1}` 이므로, **이전 state z_t와의 차이**가 필요했음.

---

## 1. v1 — head weight 증가 시도 (실패)

| What | β_reward=3, pos_weight_done=20 으로 reward/done head에 가중치 부여 |
| Why  | reward learning이 약하므로 신호를 강하게 만들자 |
| Result | **L_done이 L_latent보다 55배 커져서 encoder를 done-위주로 왜곡** |

→ Loss balancing은 해결책이 아니었음. **구조적 변화 필요**.

---

## 2. v2 — Variance regularizer 강화

| What | β_var: 0.01 → 0.1 (10×), target_std=0.3 |
| Why  | v0의 dead_dims 문제 해결 |
| Result | dead_dims 0/128 ✅, 하지만 reward 학습은 여전히 plateau |

---

## 3. v3–v4 — target_std 미세 조정

| What | target_std 0.3 → 0.15 |
| Why  | natural latent std(~0.05) 대비 너무 크면 ReLU(target − std)가 항상 양수라 encoder를 계속 밀어붙임 |
| Result | latent 안정화. reward 문제는 여전 |

---

## 4. v5 — FoodCountHead aux loss

| What | latent z로부터 **food count(scalar)** 예측하는 head 추가, MSE aux loss |
| Why  | encoder가 food count 정보를 보존하도록 강제하면 reward head가 사용할 수 있을 것 |
| Result | **Head가 mean-collapse**. food count는 ±0..1만 변하니 평균값으로 수렴해도 loss가 거의 낮아짐. encoder에 유효한 압력을 못 줌 |

---

## 5. v6 — RewardHead 입력 확장

| What | RewardHead 입력: `(z_next, h)` → `(z_t, z_next, h)` |
| Why  | food count delta 계산을 head가 직접 할 수 있도록 z_t를 제공 |
| Result | 약간 개선되었지만 **reward MSE 0.22 천장**은 여전 |

```python
class RewardHead(nn.Module):
    def forward(self, z_t, z_next, h):
        x = torch.cat([z_t, z_next, h], dim=-1)
        return self.net(x).squeeze(-1)
```

---

## 6. v7 — DynamicStateHead (multi-task reconstruction)

**가장 큰 구조적 변화**.

| What | FoodCountHead(1-dim) 폐기 → **DynamicStateHead(460-dim)** |
|      | latent z → (pacman pos[2], ghost slots[16], food mask[441], power timer[1]) 재구성 |
| Why  | Mean-collapse를 막으려면 target이 1-D scalar여서는 안 됨. 441차원 binary mask + 다중 task로 collapse를 비싸게 만들기 |
| Result | latent에 풍부한 정보 보존됨. reward MSE 점진 개선 (~0.2) |

```python
class DynamicStateHead(nn.Module):
    OUTPUT_DIM = 460   # state[901] - wall[441] (wall은 episode constant라 제외)

    def forward(self, z):
        return self.net(z)   # [pacman_pos, ghost, food_logits, power]
```

Loss는 component별로 다른 norm:
- pacman/ghost/power: MSE
- food mask (441): per-cell BCE-with-logits

---

## 7. v8 / v8.1 / v8.2 — 학습된 RewardHead 폐기

v0~v7 모두 learned RewardHead를 사용했는데, **항상 mean-predictor에서 정체**.

### 진단 (`scripts/diagnose_reward.py`):
- `food_count_t − food_count_{t+1}`은 latent에 보존되어 있음 (DynamicStateHead로 확인)
- 하지만 RewardHead는 이 신호를 추출하지 못함 — gradient가 mean prediction을 향해 휨

### v8.1 — Deterministic reward

| What | **RewardHead 완전 제거**. reward를 deterministic하게 계산: |
| | `r_raw = step_penalty + pellet_value × (count_t − count_{t+1})` |
| | `r_pred = symlog(r_raw)` |
| Why  | learned head가 mean-collapse하니, 차라리 신뢰할 수 있는 DynamicStateHead의 count를 직접 사용 |
| Result | 처음으로 reward MSE가 plateau 아래로 (~0.19) |

### v8.2 — Dynamic state head dual-path

| What | DynamicStateHead를 **양쪽 path에 적용**: |
| | (a) encoder path: dyn_state(z_all)        vs state[t]   ← clean teaching |
| | (b) dynamics path: dyn_state(z_preds)     vs state[t+1] ← matches inference |
| Why  | rollout 시 reward는 z_preds에서 나오므로, z_preds도 state를 잘 재구성하도록 직접 강제 |
| Result | **reward MSE ~0.187, latent ~0.159, done ~0.003** ← M4 milestone에 가장 근접한 baseline |

```python
# (a) encoder path
pred_enc = self.dynamic_state_head(z_all)
# (b) dynamics path
pred_dyn = self.dynamic_state_head(z_preds)
# 양쪽 모두 MSE/BCE로 loss 계산
```

---

## 8. v9 — L_count_delta 명시 제약

v8.2에서도 reward MSE가 0.184에서 천장. 진단 결과:

- per-cell food BCE: **0.028** (잘 학습됨, cell calibration OK)
- 하지만 **441개 cell의 sigmoid 합**은 ±4 RMSE의 noise
- food_eaten = ±1 신호가 ±4 noise에 묻힘 → 39% False Negative, 8% sign error

### 해결 시도

| What | L_count_delta auxiliary loss 추가: |
| | `L_count_δ = MSE( (Σσ(food_logits_enc) − Σσ(food_logits_dyn)),  food_eaten_true )` |
| Why  | sum 자체에 명시적인 제약을 걸어서 cell 합이 정수 food_eaten count에 align되도록 |
| Result | 미미한 개선 (-0.003) — per-cell BCE가 여전히 dominate하고, sum constraint의 gradient는 441개 cell로 분산되어 약함 |

→ **구조적으로 sum이 reward 신호로 적합하지 않다는 결론**.

---

## 9. v10 — FoodEatenHead (현재 채택 구조)

**근본 해결책: sum 우회**.

| What | (z_t, z_next) 입력 → P(food eaten in this transition) 출력하는 **전용 binary classifier** 도입 |
| | reward 계산: `r_raw = step_penalty + pellet_value × P(food_eaten)`, then symlog |
| Why  | Sigmoid sum의 SNR 문제는 구조적. 직접 0/1 event를 학습하는 단일 head가 noise floor를 우회 |

```python
class FoodEatenHead(nn.Module):
    def forward(self, z_t, z_next):
        x = torch.cat([z_t, z_next], dim=-1)
        return self.net(x).squeeze(-1)   # logit
```

### Dual-path BCE loss

```python
L_fe_enc = BCE(food_eaten_head(z_all[t], z_all[t+1]),  food_eaten_true)  # clean
L_fe_dyn = BCE(food_eaten_head(z_all[t], z_preds[t]),  food_eaten_true)  # matches inference
L_food_eaten = L_fe_enc + L_fe_dyn
```

### v10 / v10b / v10c 실험

| 변종 | 전략 | 결과 |
|---|---|---|
| **v10** | v8.2 backbone에서 resume + full train at LR 3e-4 | step 500에서 잠깐 reward 0.164 (lucky), 그러나 encoder가 fresh head 충격으로 발산 (latent 0.18 → 0.38) |
| **v10b** | Encoder + dynamics + dyn_state_head freeze, food_eaten_head + done_head만 학습 | 안정. reward 0.184 @ step 3000. **v8.2의 latent 용량 한계 확인** |
| **v10c** | LR을 5e-5로 6배 낮춰서 full train | reward **0.184** @ step 1500, latent **0.158** (v8.2 동률 + 약간 개선), early stop @ step 9000 |

---

## 10. 결과 요약 표

| 버전 | 핵심 변경 | latent | reward | done |
|---|---|---|---|---|
| v0 | Dreamer baseline | — | ~0.22 (plateau) | ~0.05 |
| v1 | β_r=3 강화 | — | — | done이 dominate |
| v2–v4 | variance reg 강화 | dead_dims 0 ✅ | 0.22 천장 | OK |
| v5 | FoodCountHead 추가 | — | mean-collapse | — |
| v6 | RewardHead 입력 (z_t,z_next,h) | — | 약간 개선 | — |
| v7 | DynamicStateHead(460-dim) | 안정 | ~0.20 | OK |
| v8.1 | learned RewardHead 폐기, deterministic reward | — | ~0.19 | OK |
| **v8.2** | DynamicStateHead dual-path | **0.159** | **0.187** | **0.003** |
| v9 | L_count_delta aux | 0.158 | 0.184 (−0.003) | 0.003 |
| v10 | FoodEatenHead, full LR | encoder 발산 | (lucky 0.164) | — |
| v10b | head-only fine-tune | (frozen) | 0.184 | 0.003 |
| **v10c** | low-LR full fine-tune | **0.158** | **0.184** | **0.003** |

### Policy-readiness eval (v10c, warmup=5)

| 지표 | 값 | 해석 |
|---|---|---|
| food_eaten ROC-AUC | **0.860** | 모델이 food event를 86% 정확도로 ranking |
| Pearson r (pred vs true reward) | 0.633 | 중상위 |
| Spearman r | 0.508 | 순위 보존 보통 |
| separation (pred / true) | 0.41 / 1.00 | magnitude는 압축, 방향은 보존 |
| done err | 0.005 | M4 통과 |

**해석**: M4 reward MSE 임계값(0.10)에는 못 미치지만, **policy 학습에 필요한 ranking signal은 충분히 보존** (AUC 0.86).

---

## 11. 최종 구조 (v10c) 정리

### 11.1 모델 모듈 그래프

```
state[B,L,901] ─── StateEncoder ───▶ z_all[B,L,128]
                                       │
                                       ├──▶ DynamicStateHead ──▶ pred_enc[B,L,460]
                                       │     (state reconstruction aux)
                                       │
                                       ├──┐
                                       │  │
                                       │  ▼                  
action[B,L] ── ActionEmbedder ──┐    LatentDynamics (GRU)        
                                ▼                                  
                            (z_t, a_t, h_t) ──▶ (z_{t+1}, h_{t+1})
                                                  │       │
                                                  │       └──▶ DoneHead(z_next, h) ──▶ d_pred
                                                  │
                                                  ├──▶ DynamicStateHead(z_preds) ──▶ pred_dyn
                                                  │       (rollout state aux)
                                                  │
                                                  └──┬──▶ FoodEatenHead(z_t, z_next) ──▶ logit
                                                     ▼                                     │
                                            r_pred = symlog(                              │
                                                step_penalty +                            │
                                                pellet_value · σ(logit)                   │
                                            )                                              │
```

### 11.2 핵심 코드 위치

| 컴포넌트 | 파일 | 비고 |
|---|---|---|
| `SingleWorldModel` | `world_model/single.py` | top-level forward / imagine_step |
| `StateEncoder` | `world_model/modules/encoder.py` | 901 → 128, 2-layer MLP + LayerNorm + SiLU |
| `LatentDynamics` | `world_model/modules/dynamics.py` | GRUCell(160 → 256) + MLP(256 → 128) |
| `ActionEmbedder` | `world_model/modules/action.py` | Embedding(5, 32) |
| `DoneHead` | `world_model/modules/heads.py` | (z, h) → sigmoid |
| `DynamicStateHead` | `world_model/modules/heads.py` | z → 460-dim state recon |
| `FoodEatenHead` | `world_model/modules/heads.py` | (z_t, z_next) → logit |
| `compute_world_model_loss` | `world_model/loss.py` | 모든 loss term 합 |

### 11.3 Forward sequence (학습 시)

`SingleWorldModel.forward_sequence(states, actions, burnin)`:

```python
z_all = encoder(states)                          # (B, L, 128)
dyn_state_preds = dynamic_state_head(z_all)      # (B, L, 460)

h = zeros(B, 256)
z_preds, d_preds = [], []
for t in range(L − 1):
    a_emb = action_embedder(actions[:, t])
    z_next, h = dynamics(z_all[:, t], a_emb, h)
    d = done_head(z_next, h)
    z_preds.append(z_next); d_preds.append(d)
z_preds = stack(z_preds)                         # (B, L-1, 128)

dyn_state_z_preds = dynamic_state_head(z_preds)  # (B, L-1, 460)

food_eaten_logit_enc = food_eaten_head(z_all[:-1], z_all[1:])   # clean teaching
food_eaten_logit_dyn = food_eaten_head(z_all[:-1], z_preds)     # matches inference
r_preds = symlog(step_penalty + pellet_value · σ(food_eaten_logit_dyn))
```

### 11.4 Loss (전체)

`compute_world_model_loss` (`world_model/loss.py`):

```
L_total = L_latent
        + β_reward · L_reward        # MSE(symlog(r̂), symlog(r))
        + β_done   · L_done          # weighted BCE, pos_weight=5
        + β_var    · L_var           # ReLU(target_std − std(z))
        + β_dyn    · (L_dyn_enc + L_dyn_pred)   # MSE + per-cell BCE
        + β_count  · L_count_δ                  # (0 in v10c)
        + β_fe     · (L_fe_enc + L_fe_dyn)      # BCE on food event
```

burnin step은 모든 sequence loss에서 제외 (`outputs["burnin"]`).

### 11.5 Inference / rollout (`imagine_step`)

policy 학습 시 이렇게 사용:

```python
z, h = ensemble.encode(s0)
for t in range(K):
    out = ensemble.imagine_step(z, h, a_t)    # = SingleWorldModel.imagine_step
    # out["z_next"], out["h_next"], out["reward"], out["done"], out["sigma"]
    z, h = out["z_next"], out["h_next"]
```

내부적으로:

```python
def imagine_step(z, h, a):
    a_emb = action_embedder(a)
    z_next, h_next = dynamics(z, a_emb, h)
    food_eaten_prob = σ(food_eaten_head(z, z_next))
    r = symlog(step_penalty + pellet_value · food_eaten_prob)
    d = done_head(z_next, h_next)
    return {z_next, h_next, reward_symlog: r, done: d}
```

### 11.6 하이퍼파라미터 (현재 기본값)

| 항목 | 값 |
|---|---|
| state_dim | 901 |
| latent_dim | 128 |
| gru_hidden | 256 |
| hidden_dim (heads) | 256 (dyn), 128 (others) |
| action_emb_dim | 32 |
| batch_size | 64 |
| seq_length | 50 |
| burnin (min/max) | 0 / 5 |
| LR | 3e-4 (fresh), 5e-5 (v10c fine-tune) |
| grad_clip | 0.5 |
| max_train_steps | 50000 |
| eval_every | 500 |
| patience | 15 evals |
| β_reward, β_done, β_dyn, β_food_eaten | 1.0 |
| β_var | 0.1 |
| β_count_delta | 0.0 |
| target_std | 0.15 |
| pos_weight_done | 5.0 |
| K (k-step rollout eval) | 10 |
| N (eval trajectories) | 100 |
| ensemble K | 1 (disabled) |

---

## 12. 정리 — 어떤 idea가 핵심이었는가

1. **JEPA-style self-supervised latent target** (v0 시작)
   — encoder + sg(z_target)으로 decoder를 우회. 모든 후속 design의 기반.

2. **DynamicStateHead** (v7) — auxiliary state reconstruction
   — Mean-collapse가 비싸지도록 다차원/binary target으로 압력. encoder가 모든 reward/done 관련 state info를 보존하게 만듦.

3. **Learned RewardHead 폐기** (v8.1) — 결정론적 reward 도출
   — 다듬어진 latent에서 reward를 직접 계산해서 reward head의 mean-collapse 문제 자체를 우회.

4. **Dual-path DynamicStateHead** (v8.2) — encoder path + dynamics path 동시 학습
   — rollout 시 사용되는 z_preds도 state를 재구성하도록 직접 강제. inference-train gap을 줄임.

5. **FoodEatenHead** (v10) — sum-of-sigmoids 우회
   — 441-cell sigmoid 합의 SNR 한계를 인식하고, food event 자체를 binary classification으로 풀이. **현재 reward path의 중추**.

6. **Low-LR full fine-tune from existing best** (v10c) — 새 head를 추가할 때 encoder가 발산하지 않도록 LR을 6× 낮춤. 안정적인 성능 향상의 키.
