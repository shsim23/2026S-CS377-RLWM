# Pac-Man Maze Generator — Implementation Spec

## 1. 프로젝트 맥락

**상위 프로젝트**: Variance-Aware Policy Learning in State-Based World Models (CS377)

**이 모듈의 목적**: World model이 general state transition을 학습할 수 있도록 다양한 Pac-Man 맵을 자동 생성하는 generator. 기존엔 per-map training이었으나 데이터 다양성을 위해 random map generation으로 전환.

**전체 파이프라인에서의 위치**:
```
[맵 생성] → RL agent 학습 → 학습된 agent로 데이터 수집 → World model 학습
   ↑
  현재 작업
```

**단순화된 환경 (원본 Pac-Man 대비)**:
- Power pellet 없음 (Phase 4에서 도입 예정)
- Warp tunnel 없음 (Phase 3에서 도입 예정)
- 모든 ghost의 행동이 동일 (다양한 ghost personality 없음)

---

## 2. 단계별 도입 로드맵

| Phase | 환경 변화 | 비고 |
|---|---|---|
| **1 (현재)** | 1 ghost, ghost house O, warp/pellet X | Baseline state transition 학습 |
| **2** | N ghosts (최대 3), staggered release | Multi-agent dynamics |
| **3** | + Warp tunnel | Spatial non-locality, rare event |
| **4** | + Power pellet | Temporal hidden state (frightened mode), revive 메커니즘 |

**현재 구현 범위**: Phase 1을 지원하되, 파라미터로 Phase 2~4의 모든 옵션 기능을 켤 수 있도록 확장 가능한 구조로 설계.

---

## 3. 맵 사양

### 3.1 차원
- **고정 크기**: 21 × 21 (가로 × 세로, 정사각형)
- **좌우 대칭축**: col=10 (홀수 폭의 정확한 중앙)

### 3.2 좌표 시스템
- (row, col), 0-indexed
- row=0: 맵 상단, row=20: 맵 하단
- col=0: 맵 좌측, col=20: 맵 우측

### 3.3 Tile 종류
- `WALL`: 벽 (통과 불가)
- `PATH`: 통로 (모두 통과 가능)
- `GHOST_ONLY_PATH`: ghost만 통과 가능 (gate)
- `GHOST_HOUSE_INTERIOR`: ghost house 내부 walkable (ghost만, 일반 path 규칙과 별개)

(구현 시엔 int enum이나 별도 mask로 표현해도 됨)

---

## 4. Ghost House (고정 사양)

**위치와 크기는 모든 맵에서 고정** — world model이 ghost house 구조를 학습할 필요가 없도록.

```
        col:  8   9  10  11  12
row=9:        ■   ■   G   ■   ■    ← 상단 벽 + gate
row=10:       ■   X   X   X   ■    ← 내부 walkable, ghost 3마리 위치
row=11:       ■   ■   ■   ■   ■    ← 하단 벽
```

- **외곽 영역**: row 9~11, col 8~12 (5×3)
- **내부 walkable (`GHOST_HOUSE_INTERIOR`)**: (row=10, col 9~11) — 3 tile
- **Gate (`GHOST_ONLY_PATH`)**: (row=9, col=10) — 1 tile
- **나머지 8 tile**: 모두 `WALL`
- **Ghost 시작 위치 (최대 3마리)**:
  - 1마리: (row=10, col=10)
  - 2마리: (row=10, col=9), (row=10, col=11)
  - 3마리: (row=10, col=9), (row=10, col=10), (row=10, col=11)

**Phase 1 동작**: Ghost 1마리부터 시작. Ghost는 spawn 후 gate를 통해 즉시 외부로 진출 (staggered release는 Phase 2에서 도입).

---

## 5. Pacman 시작 위치

- **고정**: (row=14, col=10) — ghost house 아래 중앙, 4 tile 거리
- 이 타일은 carving grid 위에 있음 (둘 다 짝수) → maze에 자연 연결됨

---

## 6. 맵 생성 알고리즘

### 6.1 핵심 트릭: Even-Even Carving Grid

Path가 가능한 위치를 **(row, col) 둘 다 짝수**인 cell로만 제한.
- Candidate cells: row ∈ {2, 4, 6, 8, 10, 12, 14, 16, 18}, col ∈ {2, 4, ..., 18}
- 총 9 × 9 = **81개 candidate**

**이점**: 인접 path 사이에 자동으로 1-tile wall이 생겨 "1-tile thick path + 1-tile thick wall" 제약이 구조적으로 보장됨.

**Ghost house와의 호환**:
- Ghost house 내부 (row=10, col 9~11): col=10만 carving grid 위, col=9,11은 grid 밖
- → Ghost house는 carving grid 규칙 외부의 special region으로 취급. Carving 단계가 건드리지 않음.
- Gate 위쪽 (row=8, col=10)은 carving grid 위 → maze와 자연 연결 가능

**Pacman 시작점 (row=14, col=10)**: carving grid 위 ✓

### 6.2 6단계 파이프라인

#### Stage 1: Initialize & Reserve
```
모든 cell을 WALL로 초기화
외곽 border (row=0, row=20, col=0, col=20): permanent WALL

Ghost house region 예약:
  - row 9~11, col 8~12 영역 전체를 "ghost_house_region"으로 마킹 (carving 제외)
  - (row=10, col 9~11): GHOST_HOUSE_INTERIOR
  - (row=9, col=10): GHOST_ONLY_PATH (gate)
  - 나머지 8 tile: WALL

Pacman 시작점 (row=14, col=10): PATH 보장
```

#### Stage 2: Carve Left Half (좌측 절반)

대칭축이 col=10이므로 col 2~10 영역만 carve.

```
Candidate cells = {(r, c) | r ∈ {2,4,...,18}, c ∈ {2,4,...,10}, 
                            (r,c) ∉ ghost_house_region}

알고리즘:
  1. Start from Pacman 시작점 (14, 10) — root
  2. Randomized DFS:
     - Current cell 표시
     - 4방향 인접 cell (거리 2칸: ±2 row 또는 ±2 col) 셔플
     - 미방문 candidate cell로 이동
     - 이동 시: 현재 cell, 사이 wall (1칸), 다음 cell 모두 PATH로 변환
  3. Spanning tree 완성 → tree-like maze
  4. Connectivity 추가:
     - connectivity 파라미터 (기본값 0.3)에 따라
     - random하게 추가 wall (carving grid 사이의 1-tile wall)을 PATH로 변환
     - 단, ghost_house_region wall은 변환 금지
  5. Gate connectivity 보장:
     - Gate (9, 10) 위쪽 (8, 10)이 PATH가 되도록 강제
     - (8, 10)은 carving grid 위에 있으므로 spanning tree에 자동 포함되어야 함
```

#### Stage 3: Mirror

```
col 2~9의 모든 tile을 col=10 축 기준으로 우측에 복사:
  for r in 0..20:
      for c in 2..9:
          grid[r][20 - c] = grid[r][c]

col=10 (대칭축)은 그대로 유지
Ghost house는 이미 대칭이므로 변동 없음
```

#### Stage 4: Remove Dead-Ends

Pac-Man 맵 규칙: dead-end 없음.

```
loop:
    dead_ends = [(r,c) for all PATH cells with exactly 1 PATH neighbor]
    if dead_ends.empty: break
    
    for (r, c) in dead_ends:
        # 대칭성 유지를 위해 mirror도 함께 처리
        candidates = adjacent WALL tiles of (r, c)
        candidates 제외 대상:
          - border wall (row/col == 0 또는 20)
          - ghost_house_region wall
        
        if candidates.empty: 다음 dead_end로 (또는 raise)
        
        wall_to_break = random.choice(candidates)
        wall_to_break → PATH
        
        # mirror도 함께
        mirror_wall = (wall_to_break.row, 20 - wall_to_break.col)
        if mirror_wall != wall_to_break:  # 대칭축 위가 아니면
            mirror_wall → PATH

Edge case: 대칭축 위 (col=10) path tile이 dead-end인 경우
  → mirror가 자기 자신이므로 일반 케이스로 처리 (인접 wall 하나 뚫기)
  → 새 dead-end가 생길 수 있으므로 outer loop가 다시 처리
```

#### Stage 5: Validate

```
1. Connectivity check (BFS):
   - 모든 PATH cell이 single connected component인가?
   - Pacman 시작점 (14, 10)에서 gate 외부 (8, 10)까지 도달 가능한가?
   - Ghost house 내부에서 gate를 통해 외부 maze로 도달 가능한가?

2. Symmetry check:
   - for all (r, c): grid[r][c] == grid[r][20-c]?
   (Ghost house는 자체적으로 대칭이므로 통과)

3. No dead-end check:
   - 모든 PATH cell이 ≥ 2개의 PATH neighbor를 가지는가?

실패 시: 다른 seed로 retry (max 5회). 모두 실패하면 RuntimeError.
```

#### Stage 6: Place Food

```
food_positions = []
for (r, c) in all PATH cells:
    if (r, c) in {pacman 시작점, ghost house gate}: continue
    if (r, c) is GHOST_HOUSE_INTERIOR: continue
    food_positions.append((r, c))
```

좌우 대칭이라 food 분포도 자동 대칭.

---

## 7. API 사양

### 7.1 함수 시그니처

```python
def generate_maze(
    # --- 차원 (Phase 1에서는 고정) ---
    width: int = 21,
    height: int = 21,
    
    # --- 구조 파라미터 ---
    symmetric: bool = True,             # 좌우 대칭 (Phase 1에선 항상 True)
    connectivity: float = 0.3,          # cycle 밀도 [0=tree-like, 1=fully connected]
    
    # --- Ghost house (고정) ---
    ghost_house: bool = True,           # Phase 1에선 항상 True
    
    # --- Agents ---
    num_ghosts: int = 1,                # 1~3
    
    # --- Phase 3+ 옵션 기능 (현재는 모두 0) ---
    num_warp_tunnels: int = 0,          # 0, 1, 2
    num_power_pellets: int = 0,         # 0, 4
    
    # --- 재현성 ---
    seed: int | None = None,
) -> dict:
    """
    Returns a dict with maze structure and initial state.
    See section 7.2 for the return format.
    """
```

### 7.2 반환 형식 (State Vector)

```python
{
    # --- 정적 맵 구조 (episode 동안 불변) ---
    'walls': np.ndarray,                # shape (21, 21), dtype=bool
                                        # True = wall, False = walkable (any kind)
    'ghost_only_tiles': list[tuple],    # [(r, c), ...] ghost-only walkable tiles (gate)
    'ghost_house_interior': list[tuple],# [(r, c), ...] ghost house 내부 walkable
    
    # --- Agent 위치 ---
    'pacman_pos': tuple[int, int],      # (row, col) = (14, 10)
    'ghost_positions': list[tuple],     # [(r, c), ...] length=num_ghosts
    'ghost_in_house': list[bool],       # length=num_ghosts, Phase 2 staggered release용
    
    # --- Food ---
    'food_positions': list[tuple],      # variable length, [(r, c), ...]
    'food_count': int,                  # len(food_positions), 점수 계산용
    
    # --- Score / done flag ---
    'score': int,                       # 초기 0
    'done': bool,                       # 초기 False
    
    # --- Phase 3+ 옵션 ---
    'power_pellet_positions': list[tuple],  # 현재 빈 list
    'warp_tunnel_pairs': list[tuple],       # 현재 빈 list, [((r1,c1),(r2,c2)), ...]
    
    # --- 디버그 / 메타 ---
    'seed': int | None,
    'width': int,
    'height': int,
}
```

**State vector encoding 결정사항**:
- Food는 **variable-length list of coordinates** + count로 표현 (proposal에서 결정).
- Backbone이 transformer로 전환될 예정이라 variable-length가 자연스러움.
- Walls는 고정된 (21, 21) binary mask. 맵 크기 고정이라 OK.

### 7.3 Retry 메커니즘

```python
MAX_RETRIES = 5

def generate_maze(...):
    for attempt in range(MAX_RETRIES):
        try:
            current_seed = seed + attempt if seed is not None else None
            maze = _try_generate(current_seed, ...)
            if _validate(maze):
                return maze
        except Exception as e:
            last_error = e
            continue
    
    raise RuntimeError(
        f"Failed to generate valid maze after {MAX_RETRIES} attempts. "
        f"This likely indicates a bug — current constraints should always be "
        f"satisfiable on 21x21. Last error: {last_error}"
    )
```

21×21 맵에서 81개 candidate cell, 단순한 제약 조건이라 정상 동작 시 1번째 시도에서 항상 성공해야 함. 실패는 버그 신호로 취급.

---

## 8. 구현 권장사항

### 8.1 파일 구조 제안
```
maze_generator/
├── __init__.py
├── generator.py       # main API: generate_maze()
├── carving.py         # Stage 2 randomized DFS
├── post_process.py    # Stage 4 dead-end removal, Stage 6 food placement
├── validator.py       # Stage 5 validation
├── constants.py       # tile types, ghost house spec
└── visualizer.py      # debug용 ASCII / matplotlib 시각화
```

### 8.2 시각화 함수 (필수)
ASCII 출력 함수를 구현해서 디버그 시 맵 구조를 한눈에 볼 수 있도록 할 것.
예시:
- `■` = wall
- ` ` = path (food 없음)
- `·` = path with food
- `P` = Pacman
- `G` = Ghost
- `=` = ghost house gate
- `H` = ghost house interior (without ghost)

### 8.3 Reproducibility
- `seed` 인자를 받으면 동일 seed에 항상 동일 맵 생성. 
- `numpy.random.RandomState(seed)` 또는 `random.Random(seed)` 사용.
- 전역 random state를 건드리지 말 것.

### 8.4 Testing 권장
- `seed=None`으로 100번 generate → 모두 valid 한지 확인
- 동일 seed → 동일 맵 검증
- connectivity 파라미터 변경 시 cycle 개수가 단조 증가하는지 확인
- 각 ghost 수 (1, 2, 3)에 대해 ghost house 안에 정확히 그만큼 ghost가 배치되는지 확인

---

## 9. 향후 확장 시 고려사항 (Phase 2~4)

지금 당장 구현하지 않지만, 코드 구조가 이들을 쉽게 수용할 수 있어야 함.

### Phase 2: Staggered Release
- `ghost_in_house` flag를 시간에 따라 토글
- Dot counter 메커니즘 추가 (각 ghost마다 release threshold)
- Generator는 변경 없음, environment 쪽 변경

### Phase 3: Warp Tunnel
- `num_warp_tunnels` 파라미터 활성화 (1 또는 2)
- 대칭축 (col=10)을 가로지르는 horizontal tunnel 한 쌍을 생성
- Tunnel 위치는 좌우 대칭 보장 (자동: tunnel은 좌우 가장자리에서 같은 row에 위치)
- `warp_tunnel_pairs`에 [((r, 0), (r, 20)), ...] 형식으로 추가
- 외곽 border가 tunnel 위치에서 path로 변환됨

### Phase 4: Power Pellet
- `num_power_pellets` 파라미터 활성화 (보통 4)
- 맵의 네 모서리 근처 (대칭적) path tile에 배치
- 추가 state: ghost별 frightened mode timer, mode flag

---

## 10. 작업 시작 시 확인사항

1. 위 spec을 모두 이해했는지 확인
2. Phase 1 (num_ghosts=1, num_warp_tunnels=0, num_power_pellets=0) 만 우선 완성
3. ASCII visualizer로 생성된 맵을 사람이 검수 가능하게 출력
4. 다양한 seed로 10개 정도 맵을 출력해서 시각적으로 다양성 확인
5. 위 모든 검증을 통과하면 Phase 2~4용 hook이 잘 마련되어 있는지 마지막 점검

질문이 있으면 작업 시작 전에 정리해서 물어볼 것.
