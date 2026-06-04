#!/usr/bin/env bash
# Quick experiment: collect a dataset from the RL agents trained SO FAR (partial
# set — whatever layouts already have an agent), then train a world model on it.
# For checking world-model quality on the agent-covered maps without waiting for
# all 30 agents. Runs unattended in tmux.
set -uo pipefail

ROOT="/home/ubuntu/2026S-CS377-RLWM"
PY="/home/ubuntu/miniconda/envs/pacman-wm/bin/python"
export PYTHONUNBUFFERED=1
cd "$ROOT"

DATASET="${DATASET:-rl_partial}"
AGENTS_ROOT="${AGENTS_ROOT:-checkpoints/rl_agents}"
N_TRANSITIONS="${N_TRANSITIONS:-150000}"
GHOSTS="${GHOSTS:-1 2 3 4}"
STEPS="${STEPS:-50000}"          # world-model train steps (50k = same as heuristic baseline)
SAVE_EVERY="${SAVE_EVERY:-5000}" # keep a persistent step_<N>.pt snapshot every N steps
DEVICE="${DEVICE:-cuda}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/wm_pipeline/rl_quick_${STAMP}.log"
mkdir -p "$ROOT/logs/wm_pipeline"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== RL quick WM experiment (dataset=$DATASET, steps=$STEPS) ==="

log "[1/2] Collecting RL dataset from agents trained so far (partial)..."
"$PY" scripts/wm_collect_dataset.py --dataset "$DATASET" --pool-split train \
    --n-transitions "$N_TRANSITIONS" --ghost-choices $GHOSTS --seed 0 \
    --policy-source rl --rl-agents-root "$AGENTS_ROOT" --rl-allow-partial \
    --rl-optimal-weight 0.7 --device cpu 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then log "ERROR: collection failed. Aborting."; exit 1; fi

log "[2/2] Training world model on '$DATASET' ($STEPS steps)..."
"$PY" scripts/wm_train_dreamer.py --dataset "$DATASET" --device "$DEVICE" \
    --seed 0 --max-steps "$STEPS" --save-every "$SAVE_EVERY" 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then log "ERROR: world-model training failed. Aborting."; exit 1; fi

touch "$ROOT/logs/wm_pipeline/rl_quick_${STAMP}.DONE"
log "=== DONE. Checkpoint: checkpoints/dreamer_wm/$DATASET/best.pt ==="
log "Visualize WM prediction on a trained map with:"
log "  $PY scripts/wm_eval_visualize.py --checkpoint checkpoints/dreamer_wm/$DATASET/best.pt --test-dataset $DATASET --layout-id 0"
