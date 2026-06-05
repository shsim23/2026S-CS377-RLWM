#!/usr/bin/env bash
# Single-map sanity pipeline: collect a dedicated dataset for ONE layout, train the
# two-hot world model on it with a proper (not fast-cycle) schedule, then measure
# prediction quality ON THE SAME map. Decouples "does the WM fit one map well?"
# from cross-layout generalization. Runs unattended; chain: collect -> train -> measure.
set -uo pipefail

ROOT="/home/ubuntu/2026S-CS377-RLWM"
PY="/home/ubuntu/miniconda/envs/pacman-wm/bin/python"
export PYTHONUNBUFFERED=1
cd "$ROOT"

LAYOUT_ID="${LAYOUT_ID:-0}"                       # pool index (0 = train_000)
DATASET="${DATASET:-rl_single_L${LAYOUT_ID}}"
N_TRANSITIONS="${N_TRANSITIONS:-150000}"          # plenty of coverage for one map
GHOSTS="${GHOSTS:-1 2 3 4}"
AGENTS_ROOT="${AGENTS_ROOT:-checkpoints/rl_agents}"  # per-layout PPO agents (optimal/suboptimal)
# --- WM training (proper single-map schedule, two-hot positions) ---
POSITION_MODE="${POSITION_MODE:-twohot}"
BETA_CONT="${BETA_CONT:-1.0}"
BATCH="${BATCH:-32}"
SEQ="${SEQ:-64}"
CONTEXT="${CONTEXT:-8}"
STEPS="${STEPS:-20000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
N_EVAL_WINDOWS="${N_EVAL_WINDOWS:-64}"
DEVICE="${DEVICE:-cuda}"
CKPT_DIR="${CKPT_DIR:-checkpoints/dreamer_wm/${DATASET}_${POSITION_MODE}}"
MEAS_DIR="${MEAS_DIR:-logs/wm_eval/${DATASET}_${POSITION_MODE}}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/wm_pipeline/single_map_${DATASET}_${STAMP}.log"
mkdir -p "$ROOT/logs/wm_pipeline"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== single-map pipeline: layout $LAYOUT_ID -> dataset $DATASET (agents=$AGENTS_ROOT) ==="
log "    train: position_mode=$POSITION_MODE beta_cont=$BETA_CONT batch=$BATCH seq=$SEQ steps=$STEPS"

log "[1/3] Collecting $N_TRANSITIONS transitions for layout $LAYOUT_ID ..."
"$PY" scripts/wm_collect_dataset.py --dataset "$DATASET" --pool-split train \
    --only-layouts "$LAYOUT_ID" --n-transitions "$N_TRANSITIONS" \
    --ghost-choices $GHOSTS --seed 0 \
    --policy-source rl --rl-agents-root "$AGENTS_ROOT" \
    --rl-optimal-weight 0.7 --device cpu 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then log "ERROR: collection failed."; exit 1; fi

log "[2/3] Training $POSITION_MODE world model on '$DATASET' ($STEPS steps) ..."
"$PY" scripts/wm_train_dreamer.py --dataset "$DATASET" --device "$DEVICE" --seed 0 \
    --position-mode "$POSITION_MODE" --beta-cont "$BETA_CONT" --layout-id "$LAYOUT_ID" \
    --batch-size "$BATCH" --seq-len "$SEQ" --context "$CONTEXT" \
    --max-steps "$STEPS" --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" \
    --n-eval-windows "$N_EVAL_WINDOWS" \
    --checkpoint-dir "$CKPT_DIR" 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then log "ERROR: training failed."; exit 1; fi

log "[3/3] Measuring position quality on the SAME map (layout $LAYOUT_ID) ..."
"$PY" scripts/wm_measure_position.py --dataset "$DATASET" --layout-id "$LAYOUT_ID" \
    --n-windows 256 --out-dir "$MEAS_DIR" \
    --checkpoints \
      "twohot_best=$CKPT_DIR/best.pt" \
      "twohot_latest=$CKPT_DIR/latest.pt" 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then log "ERROR: measurement failed."; exit 1; fi

touch "$ROOT/logs/wm_pipeline/single_map_${DATASET}_${STAMP}.DONE"
log "=== DONE. ckpt=$CKPT_DIR  measure=$MEAS_DIR ==="
log "Visualize: $PY scripts/wm_eval_visualize.py --checkpoint $CKPT_DIR/best.pt --test-dataset $DATASET --layout-id $LAYOUT_ID"
