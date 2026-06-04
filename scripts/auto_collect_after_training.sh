#!/usr/bin/env bash
# Watcher: wait until all per-layout RL TRAIN agents are trained, then automatically
# (1) train the 5 TEST-pool agents, (2) collect the RL TRAIN dataset, (3) collect the
# RL TEST dataset. Stops after collection (no world-model retrain). Meant to run
# unattended in its own tmux session while the train-pool training finishes overnight.
set -uo pipefail

ROOT="/home/ubuntu/2026S-CS377-RLWM"
PY="/home/ubuntu/miniconda/envs/pacman-wm/bin/python"
export PYTHONUNBUFFERED=1
cd "$ROOT"

AGENTS_ROOT="checkpoints/rl_agents"
TRAIN_DATASET="main_rl"
TEST_DATASET="main_rl_test"
N_TRANSITIONS="${N_TRANSITIONS:-500000}"
N_TEST_TRANSITIONS="${N_TEST_TRANSITIONS:-20000}"
GHOSTS="${GHOSTS:-1 2 3 4}"
RL_ITERATIONS="${RL_ITERATIONS:-300}"
RL_NUM_ENVS="${RL_NUM_ENVS:-64}"
DEVICE="${DEVICE:-cuda}"
POLL_SECS="${POLL_SECS:-60}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/pacman_rl/auto_collect_${STAMP}.log"
mkdir -p "$ROOT/logs/pacman_rl"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Count how many TRAIN-pool layouts still lack a finished agent (optimal+suboptimal).
missing_train_agents() {
    "$PY" - "$AGENTS_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
pool = json.loads(Path("layouts/wm_pool/manifest.json").read_text())
missing = 0
for e in pool["train"]:
    d = root / e["layout_id"]
    if not (d / "optimal.pt").exists() or not (d / "suboptimal.pt").exists():
        missing += 1
print(missing)
PY
}

log "=== auto-collect watcher started (log: $LOG) ==="
log "Waiting for TRAIN-pool agents under $AGENTS_ROOT (poll every ${POLL_SECS}s)..."

while true; do
    MISSING="$(missing_train_agents)"
    if [ "$MISSING" = "0" ]; then
        log "All TRAIN-pool agents present. Proceeding."
        break
    fi
    log "  still training: $MISSING train-pool agent(s) missing"
    sleep "$POLL_SECS"
done

log "[1/3] Training TEST-pool agents (${RL_ITERATIONS} iters each)..."
"$PY" scripts/wm_train_rl_agents.py --pool-split test --agents-root "$AGENTS_ROOT" \
    --iterations "$RL_ITERATIONS" --num-envs "$RL_NUM_ENVS" --device "$DEVICE" \
    --seed 0 --headless --skip-existing 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then log "ERROR: test-agent training failed. Aborting."; exit 1; fi

log "[2/3] Collecting RL TRAIN dataset '$TRAIN_DATASET' ($N_TRANSITIONS transitions)..."
"$PY" scripts/wm_collect_dataset.py --dataset "$TRAIN_DATASET" --pool-split train \
    --n-transitions "$N_TRANSITIONS" --ghost-choices $GHOSTS --seed 0 \
    --policy-source rl --rl-agents-root "$AGENTS_ROOT" --rl-optimal-weight 0.7 \
    --device cpu 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then log "ERROR: train-dataset collection failed. Aborting."; exit 1; fi

log "[3/3] Collecting RL TEST dataset '$TEST_DATASET' ($N_TEST_TRANSITIONS transitions)..."
"$PY" scripts/wm_collect_dataset.py --dataset "$TEST_DATASET" --pool-split test \
    --n-transitions "$N_TEST_TRANSITIONS" --ghost-choices $GHOSTS --seed 1 \
    --policy-source rl --rl-agents-root "$AGENTS_ROOT" --rl-optimal-weight 0.7 \
    --device cpu 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then log "ERROR: test-dataset collection failed. Aborting."; exit 1; fi

touch "$ROOT/logs/pacman_rl/auto_collect_${STAMP}.DONE"
log "=== DONE. RL datasets ready: data/replay/$TRAIN_DATASET , data/replay/$TEST_DATASET ==="
log "Next (manual): retrain WM with"
log "  $PY scripts/wm_train_dreamer.py --dataset $TRAIN_DATASET --device $DEVICE --seed 0"
