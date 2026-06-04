# 단일맵 World Model — Eval & 시각화 가이드 (map 000)

단일맵(`train_000`, layout 0)으로 학습한 DreamerV3 world model을 **같은 맵에서**
평가하고, 예측 결과를 실제 팩맨처럼 렌더링해 눈으로 확인하는 방법만 정리한
문서입니다.

> 모든 명령은 conda env `pacman-wm`에서 실행합니다.
> ```bash
> conda activate pacman-wm        # 또는 아래처럼 절대경로 파이썬 사용
> # /home/ubuntu/miniconda/envs/pacman-wm/bin/python <script>
> ```
> 작업 디렉터리: `/home/ubuntu/2026S-CS377-RLWM`

---

## 0. 대상 자산 위치

| 항목 | 경로 |
| --- | --- |
| **맵 (정본)** | `layouts/wm_pool/train/layout_000.txt` (= `train_000`, wall_hash `ed15b899340ffb27`) |
| **맵 그림** | `logs/wm_eval/train_000_layout.png` |
| **학습된 WM 체크포인트** | `checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt` |
| **같은 맵 데이터셋** | `data/replay/rl_single_L0` (150k transitions, ghost=1) |
| **설정 파일** | `configs/world_model/dreamer_v3.yaml` |

이 WM은 `(state, action) → next state`를 예측하는 모델이고, 그것을
**auto-regressive**(이전 예측을 다음 입력으로)하게 굴려서 imagination을 만듭니다.
평가는 항상 **단일 에피소드 윈도우**(`SingleEpisodeReplay`)에서 진행합니다 —
컨텍스트 구간만 실제 관측으로 posterior를 채우고(warm-up), 그 뒤로는 관측 없이
prior만 굴립니다.

---

## 1. 수치 평가 (어디서 에러가 쌓이는가)

호라이즌별 pacman/ghost cell-L1, exact-cell %, food IoU, reward |err|를
persistence baseline과 함께 측정합니다.

```bash
python scripts/wm_imagine_analysis.py \
    --checkpoint checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt \
    --test-dataset rl_single_L0 --layout-id 0 --n-windows 512
```

출력 → `logs/wm_eval/rl_single_L0_layout0_imagine_analysis/`
(CSV + JSON + `*.png`: 모델 vs persistence 곡선). 주요 인자: `--n-windows`(평가
윈도우 수), `--device`.

---

## 2. 정지 이미지 시각화 (GT vs imagine 사다리)

여러 호라이즌(h=1,2,4,8,16,32)에서 ground-truth와 open-loop 예측 격자를 나란히
렌더링합니다.

```bash
python scripts/wm_eval_visualize.py \
    --checkpoint checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt \
    --test-dataset rl_single_L0 --layout-id 0 --n-examples 4
```

출력 → `logs/wm_eval/rl_single_L0_layout0/`
(`rollout_qualitative.png`, `kstep_curves.png`, `metrics.json`).
`--viz-horizons`로 호라이즌 사다리, `--measure-windows`로 수치 측정 병행 가능.

---

## 3. 애니메이션(GIF) 시각화 — "텍스트 → 실제 팩맨"

전체 호라이즌을 프레임별로 GT(왼쪽) vs imagine(오른쪽) GIF로 만듭니다.

```bash
python scripts/wm_imagine_gif.py \
    --checkpoint checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt \
    --test-dataset rl_single_L0 --layout-id 0 --n-examples 3 --fps 3
```

출력 → `logs/wm_eval/rl_single_L0_layout0_imagine_gif/imagine_*.gif`

### 렌더링 규칙 (raw 901-d state → 아케이드 화면)

`scripts/wm_eval_visualize.py: draw_frame()`이 담당합니다.

- 좌표 역정규화: cell idx `[0,20] ↔ [-1,1]`,
  `round((coord+1)/2*20)`로 격자칸 복원 (21×21).
- **벽** = 파란 칸(static, `wall_flat`에서). **food** = 노란 점.
  **팩맨** = 진행 방향으로 입을 벌린 노란 wedge. **ghost** = 빨간 유령(눈 포함).
- 팩맨/유령 방향은 직전 프레임 대비 이동(`prev_dyn`)으로 계산.

---

## 4. 맵 그림만 다시 뽑기 (팀 공유용)

```bash
python - <<'PY'
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from pathlib import Path
lines=[l for l in Path("layouts/wm_pool/train/layout_000.txt").read_text().splitlines() if l]
H=len(lines); W=max(map(len,lines))
COL={'%':(.10,.10,.55)}
g=np.zeros((H,W,3))
for r,row in enumerate(lines):
    for c in range(W): g[r,c]=COL.get(row[c] if c<len(row) else '%',(0,0,0))
fig,ax=plt.subplots(figsize=(6,6),facecolor="#111"); ax.imshow(g,interpolation="nearest")
for r,row in enumerate(lines):
    for c,ch in enumerate(row):
        if ch=='.': ax.plot(c,r,'o',color="#ffd27f",ms=3)
        elif ch=='P': ax.plot(c,r,'o',color="yellow",ms=14)
        elif ch=='G': ax.plot(c,r,'o',color="red",ms=14)
ax.set_xticks([]); ax.set_yticks([])
Path("logs/wm_eval").mkdir(parents=True,exist_ok=True)
fig.tight_layout(); fig.savefig("logs/wm_eval/train_000_layout.png",dpi=130,facecolor="#111")
PY
```

---

## 한 줄 요약

```bash
# 수치
python scripts/wm_imagine_analysis.py --checkpoint checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt --test-dataset rl_single_L0 --layout-id 0
# 정지 이미지
python scripts/wm_eval_visualize.py   --checkpoint checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt --test-dataset rl_single_L0 --layout-id 0
# GIF
python scripts/wm_imagine_gif.py      --checkpoint checkpoints/dreamer_wm/rl_single_L0_twohot/best.pt --test-dataset rl_single_L0 --layout-id 0
```
