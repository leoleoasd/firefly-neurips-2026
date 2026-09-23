#!/bin/bash
# Task configuration — TELECOM smoke test.
#
# Tiny end-to-end run on the SHARED running Ray cluster to prove the integration
# works. Reuses tasks/telecom.sh but shrinks batch/rollout sizes, disables eval,
# and runs wandb offline + weave disabled (no network needed). The user simulator
# is whatever sglang model is already registered under rm_router.

# Smoke-only runtime env (forwarded by run_grpo_async.sh when set).
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-/tmp/tau2_wandb}
export WEAVE_DISABLED=${WEAVE_DISABLED:-true}

# Tiny sizes (override telecom.sh defaults).
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
export OVER_SAMPLING_BS=${OVER_SAMPLING_BS:-4}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
export NUM_ROLLOUT=${NUM_ROLLOUT:-2}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export TAU2_MAX_TURNS=${TAU2_MAX_TURNS:-20}
# sglang ignores the served model id for single-model serving, so any registered
# user-sim routes correctly via rm_router.
export USER_SIM_MODEL=${USER_SIM_MODEL:-"/data/base_models/Qwen/Qwen3-30B-A3B-Thinking-2507"}

source "$(dirname -- "${BASH_SOURCE[0]}")/telecom.sh"

# Eval fully disabled for the smoke run.
EVAL_ARGS=(--skip-eval-before-train)
