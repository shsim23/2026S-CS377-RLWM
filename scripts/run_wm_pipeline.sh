#!/usr/bin/env bash
# Full DreamerV3 world-model pipeline: generate layouts -> (optionally train
# per-layout RL agents) -> collect data (ghost counts 1-4 mixed) -> train world
# model. Designed to run unattended in tmux.
set -euo pipefail

ROOT="/home/ubuntu/2026S-CS377-RLWM"
PY="/home/ubuntu/miniconda/envs/pacman-wm/bin/python"
export PYTHONUNBUFFERED=1   # live progress when piped through tee
cd "$ROOT"

# --- knobs (override via env) ---
DATASET="${DATASET:-main}"
TEST_DATASET="${TEST_DATASET:-main_test}"
N_TRAIN="${N_TRAIN:-30}"
N_TEST="${N_TEST:-5}"
N_TRANSITIONS="${N_TRANSITIONS:-500000}"
N_TEST_TRANSITIONS="${N_TEST_TRANSITIONS:-20000}"
GHOSTS="${GHOSTS:-1 2 3 4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

# --- collection policy source: 'heuristic' (default) or 'rl' ---
# 'rl' first trains one PPO agent per layout (train + test pools), then collects
# with a per-episode 70/30 optimal/sub-optimal checkpoint mix (deterministic
# argmax). Diversity comes from the checkpoint mix, randomized spawns, stochastic
# ghosts, and the ghost-count sweep.
POLICY_SOURCE="${POLICY_SOURCE:-heuristic}"
AGENTS_ROOT="${AGENTS_ROOT:-checkpoints/rl_agents}"
RL_ITERATIONS="${RL_ITERATIONS:-500}"
RL_NUM_ENVS="${RL_NUM_ENVS:-64}"
RL_OPTIMAL_WEIGHT="${RL_OPTIMAL_WEIGHT:-0.7}"

COLLECT_EXTRA=""
if [ "$POLICY_SOURCE" = "rl" ]; then
    COLLECT_EXTRA="--policy-source rl --rl-agents-root $AGENTS_ROOT --rl-optimal-weight $RL_OPTIMAL_WEIGHT"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/logs/wm_pipeline"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/pipeline_${STAMP}.log"

echo "=== WM pipeline @ $STAMP ===" | tee "$LOG"
echo "dataset=$DATASET ghosts=[$GHOSTS] n_transitions=$N_TRANSITIONS device=$DEVICE policy_source=$POLICY_SOURCE" | tee -a "$LOG"

echo -e "\n[1] Generating layout pools ($N_TRAIN train / $N_TEST test)..." | tee -a "$LOG"
$PY scripts/wm_generate_layouts.py --n-train "$N_TRAIN" --n-test "$N_TEST" --seed "$SEED" 2>&1 | tee -a "$LOG"

if [ "$POLICY_SOURCE" = "rl" ]; then
    echo -e "\n[2] Training per-layout PPO agents (train + test pools, $RL_ITERATIONS iters each)..." | tee -a "$LOG"
    $PY scripts/wm_train_rl_agents.py --pool-split train --agents-root "$AGENTS_ROOT" \
        --iterations "$RL_ITERATIONS" --num-envs "$RL_NUM_ENVS" --device "$DEVICE" \
        --seed "$SEED" --headless --skip-existing 2>&1 | tee -a "$LOG"
    $PY scripts/wm_train_rl_agents.py --pool-split test --agents-root "$AGENTS_ROOT" \
        --iterations "$RL_ITERATIONS" --num-envs "$RL_NUM_ENVS" --device "$DEVICE" \
        --seed "$SEED" --headless --skip-existing 2>&1 | tee -a "$LOG"
fi

echo -e "\n[3] Collecting TRAIN dataset '$DATASET' ($N_TRANSITIONS transitions, ghosts $GHOSTS, source=$POLICY_SOURCE)..." | tee -a "$LOG"
$PY scripts/wm_collect_dataset.py --dataset "$DATASET" --pool-split train \
    --n-transitions "$N_TRANSITIONS" --ghost-choices $GHOSTS --seed "$SEED" $COLLECT_EXTRA 2>&1 | tee -a "$LOG"

echo -e "\n[4] Collecting held-out TEST dataset '$TEST_DATASET' ($N_TEST_TRANSITIONS transitions, source=$POLICY_SOURCE)..." | tee -a "$LOG"
$PY scripts/wm_collect_dataset.py --dataset "$TEST_DATASET" --pool-split test \
    --n-transitions "$N_TEST_TRANSITIONS" --ghost-choices $GHOSTS --seed $((SEED + 1)) $COLLECT_EXTRA 2>&1 | tee -a "$LOG"

echo -e "\n[5] Training world model on '$DATASET'..." | tee -a "$LOG"
$PY scripts/wm_train_dreamer.py --dataset "$DATASET" --device "$DEVICE" --seed "$SEED" 2>&1 | tee -a "$LOG"

echo -e "\n=== Pipeline complete. Log: $LOG ===" | tee -a "$LOG"
echo "Run cross-layout eval with:" | tee -a "$LOG"
echo "  $PY scripts/wm_eval_dreamer.py --checkpoint checkpoints/dreamer_wm/$DATASET/best.pt --dataset $DATASET --test-dataset $TEST_DATASET" | tee -a "$LOG"
