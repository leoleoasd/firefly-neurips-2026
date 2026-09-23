#!/bin/bash
# Task configuration — tau2-bench TELECOM domain.
#
# Sourced by run_grpo_async.sh BEFORE the model config (so the model-knob
# overrides below take effect). Expects SCRIPT_DIR.
#
# Sets: ROLLOUT_ARGS, EVAL_ARGS, GRPO_ARGS, and exports TAU2_*/USER_SIM_MODEL.

# ── Model-knob overrides (read by models/*.sh) ───────────────────────────────
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
export SAVE_INTERVAL=${SAVE_INTERVAL:-5}
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-8}

# ── Tau2 domain (no reward penalties for telecom) ────────────────────────────
export TAU2_DOMAIN=${TAU2_DOMAIN:-"telecom"}
export TAU2_REWARD_TYPE=${TAU2_REWARD_TYPE:-"all"}
export TAU2_MAX_TURNS=${TAU2_MAX_TURNS:-20}

# ── User simulator — larger Qwen3-235B user-sim (override via env) ────────────
export USER_SIM_MODEL=${USER_SIM_MODEL:-"/data/checkpoints/tau/Qwen/Qwen3-235B-A22B-Thinking-2507"}

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_PATH=${DATA_PATH:-"${SCRIPT_DIR}/data/${TAU2_DOMAIN}_train_tasks.jsonl"}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-"${SCRIPT_DIR}/data/${TAU2_DOMAIN}_test_tasks.jsonl"}

# ── Rollout / GRPO hyperparameters ───────────────────────────────────────────
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-24}
OVER_SAMPLING_BS=${OVER_SAMPLING_BS:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-24}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
NUM_ROLLOUT=${NUM_ROLLOUT:-1000}

ROLLOUT_ARGS=(
  --prompt-data ${DATA_PATH}
  --input-key index
  --rollout-shuffle
  --num-rollout ${NUM_ROLLOUT}
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-response-len 8192
  --rollout-temperature 1.0
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --balance-data
  --over-sampling-batch-size ${OVER_SAMPLING_BS}
  --dynamic-sampling-filter-path
    slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

EVAL_ARGS=(
  --skip-eval-before-train
  --eval-interval 200
  --eval-prompt-data ${TAU2_DOMAIN}-test ${EVAL_DATA_PATH}
  --n-samples-per-eval-prompt 1
  --eval-max-response-len 8192
  --eval-top-k 1
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --entropy-coef 0.00
  --eps-clip 0.4
  --use-tis
)
