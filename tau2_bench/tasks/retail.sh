#!/bin/bash
# Task configuration — tau2-bench RETAIL domain.
#
# Sourced by run_grpo_async.sh AFTER the model config. Sets the domain, reward
# shaping, rollout/eval/GRPO args, and the user simulator. Expects SCRIPT_DIR.
#
# Sets: ROLLOUT_ARGS, EVAL_ARGS, GRPO_ARGS, and exports TAU2_*/USER_SIM_MODEL.

# ── Tau2 domain + reward shaping (env vars read by agent_moe.py) ──────────────
export TAU2_DOMAIN=${TAU2_DOMAIN:-"retail"}
export TAU2_REWARD_TYPE=${TAU2_REWARD_TYPE:-"all"}
export TAU2_MAX_TURNS=${TAU2_MAX_TURNS:-50}

# Reward penalties — see agent_moe.py final_reward_with_penalties().
# Override/disable any of these by exporting them before launching.
export TAU2_TOOL_FORMAT_PENALTY=${TAU2_TOOL_FORMAT_PENALTY:--0.1}
export TAU2_MAX_ASSISTANT_TOKENS=${TAU2_MAX_ASSISTANT_TOKENS:-4096}
export TAU2_ASSISTANT_LENGTH_PENALTY_PER_EXCESS_TOKEN=${TAU2_ASSISTANT_LENGTH_PENALTY_PER_EXCESS_TOKEN:--0.0001}
export TAU2_CONSECUTIVE_SAME_TOOL_PENALTY=${TAU2_CONSECUTIVE_SAME_TOOL_PENALTY:--0.15}
export TAU2_EXCESS_STEPS_THRESHOLD=${TAU2_EXCESS_STEPS_THRESHOLD:-30}
export TAU2_EXCESS_STEPS_PENALTY_PER_STEP=${TAU2_EXCESS_STEPS_PENALTY_PER_STEP:--0.05}

# ── User simulator (local sglang served via rm_router; override via env) ──────
# To use Azure OpenAI instead, source tasks/retail_azure.sh.
export USER_SIM_MODEL=${USER_SIM_MODEL:-"/data/checkpoints/tau/Qwen/Qwen3-30B-A3B-Thinking-2507-user-sim"}

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_PATH=${DATA_PATH:-"${SCRIPT_DIR}/data/${TAU2_DOMAIN}_train_tasks.jsonl"}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-"${SCRIPT_DIR}/data/${TAU2_DOMAIN}_test_tasks.jsonl"}

# ── Rollout / GRPO hyperparameters ───────────────────────────────────────────
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-24}
OVER_SAMPLING_BS=${OVER_SAMPLING_BS:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
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
  --use-kl-loss
  --kl-loss-coef ${KL_LOSS_COEF:-0.001}
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high ${EPS_CLIP_HIGH:-0.28}
  --use-tis
)
