#!/bin/bash
# Run tau2-bench GRPO training with fully async rollout.
# NOTE: This script only submits the job; it does NOT kill/start Ray.
# Prereqs: Ray cluster up, rm_router running, and (unless using Azure) a user-sim
# sglang server registered under "rm_router". See README.md.
#
# Usage: ./run_grpo_async.sh <model_config> <task_config> <run_name>
#   model_config: file under tau2_bench/models/ (without .sh), e.g. qwen3-30B-A3B
#   task_config:  file under tau2_bench/tasks/  (without .sh), e.g. retail | telecom | retail_azure
#   run_name:     wandb exp name / checkpoint subdirectory suffix
#
# The two config layers are independent:
#   tasks/*.sh   -> domain, reward shaping, rollout/eval/GRPO args, user simulator
#   models/*.sh  -> architecture, ckpt paths, perf/sglang/MoE args, cluster size
# The task is sourced first so it can override model knobs (MAX_TOKENS_PER_GPU,
# SAVE_INTERVAL, ROLLOUT_NUM_GPUS).

set -ex

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 <model_config> <task_config> <run_name>" >&2
  echo "  e.g. $0 qwen3-30B-A3B retail my_experiment" >&2
  exit 1
fi

MODEL_CONFIG_NAME="$1"
TASK_CONFIG_NAME="$2"
RUN_NAME="$3"

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

TASK_CONFIG_FILE="${SCRIPT_DIR}/tasks/${TASK_CONFIG_NAME}.sh"
MODEL_CONFIG_FILE="${SCRIPT_DIR}/models/${MODEL_CONFIG_NAME}.sh"
[ -f "${TASK_CONFIG_FILE}" ] || { echo "Error: task config not found: ${TASK_CONFIG_FILE}" >&2; exit 1; }
[ -f "${MODEL_CONFIG_FILE}" ] || { echo "Error: model config not found: ${MODEL_CONFIG_FILE}" >&2; exit 1; }

# Task first (sets domain + model-knob overrides), then model (builds arg arrays).
source "${TASK_CONFIG_FILE}"
source "${MODEL_CONFIG_FILE}"

# ── Shared task config (model- and domain-independent) ───────────────────────
LEARNING_RATE=${LEARNING_RATE:-"5e-6"}

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr ${LEARNING_RATE}
  --lr-decay-style constant
  --weight-decay 0.005
  --adam-beta1 0.9
  --adam-beta2 0.999
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_tau2_gym_moe.generate
  --rollout-function-path shared.fully_async_rollout.generate_rollout_fully_async
  --data-source-path shared.data_source.RolloutDataSourceWithExclusion
  --custom-rollout-log-function-path shared.rollout_log.log_rollout_data
)

WANDB_PROJECT=${WANDB_PROJECT:-"slime-tau2-bench"}
WANDB_GROUP=${WANDB_GROUP:-"qwen3-30b-a3b-${TAU2_DOMAIN}"}
WANDB_ARGS=(
  --use-wandb
  --wandb-project ${WANDB_PROJECT}
  --wandb-group ${WANDB_GROUP}
  --wandb-exp-name ${RUN_NAME}
  # slime ignores --wandb-exp-name for the run name and, with the random suffix
  # on, names runs "<group>_<id>-RANK_<rank>". Disable it so the run name is just
  # the (correct) group, instead of an unreadable RANK-suffixed id.
  --disable-wandb-random-suffix
)

# ── Runtime env (forward what the rollout workers need) ──────────────────────
# USER_SIM_MODEL / AZURE_* are passed through verbatim; empty means "unset" and
# tau2_env_workers.py:_resolve_user_sim() falls through to the next option.
join_by() { local d=$1; shift; printf '%s' "$1"; shift; printf '%s' "${@/#/$d}"; }

ENV_ENTRIES=(
  "\"PYTHONPATH\": \"${SCRIPT_DIR}:${REPO_ROOT}/thirdparty/tau2-bench/src\""
  "\"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\""
  "\"NCCL_SOCKET_IFNAME\": \"^lo,docker\""
  "\"RAY_DEDUP_LOGS\": \"1\""
  "\"TOOL_CALL_PARSER\": \"${TOOL_CALL_PARSER:-qwen25}\""
  "\"SLIME_DISABLE_PROCESSOR\": \"${SLIME_DISABLE_PROCESSOR:-0}\""
  "\"TAU2_DOMAIN\": \"${TAU2_DOMAIN}\""
  "\"TAU2_REWARD_TYPE\": \"${TAU2_REWARD_TYPE}\""
  "\"TAU2_MAX_TURNS\": \"${TAU2_MAX_TURNS}\""
  "\"TAU2_TOOL_FORMAT_PENALTY\": \"${TAU2_TOOL_FORMAT_PENALTY:-}\""
  "\"TAU2_MAX_ASSISTANT_TOKENS\": \"${TAU2_MAX_ASSISTANT_TOKENS:-}\""
  "\"TAU2_ASSISTANT_LENGTH_PENALTY_PER_EXCESS_TOKEN\": \"${TAU2_ASSISTANT_LENGTH_PENALTY_PER_EXCESS_TOKEN:-}\""
  "\"TAU2_CONSECUTIVE_SAME_TOOL_PENALTY\": \"${TAU2_CONSECUTIVE_SAME_TOOL_PENALTY:-}\""
  "\"TAU2_EXCESS_STEPS_THRESHOLD\": \"${TAU2_EXCESS_STEPS_THRESHOLD:-}\""
  "\"TAU2_EXCESS_STEPS_PENALTY_PER_STEP\": \"${TAU2_EXCESS_STEPS_PENALTY_PER_STEP:-}\""
  "\"USER_SIM_MODEL\": \"${USER_SIM_MODEL:-}\""
  "\"AZURE_USER_SIM_MODEL\": \"${AZURE_USER_SIM_MODEL:-}\""
  "\"AZURE_API_KEY\": \"${AZURE_API_KEY:-}\""
  "\"AZURE_API_BASE\": \"${AZURE_API_BASE:-}\""
  "\"AZURE_API_VERSION\": \"${AZURE_API_VERSION:-}\""
  "\"BEDROCK_USER_SIM_MODEL\": \"${BEDROCK_USER_SIM_MODEL:-}\""
  "\"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\""
)
# Optional: only forward when set (e.g. smoke runs export WANDB_MODE=offline).
# PYTORCH_CUDA_ALLOC_CONF: opt-in only. expandable_segments:True helps the
# Megatron trainer's fragmentation BUT breaks SGLang's custom all-reduce CUDA
# graph capture (it's a job-wide env, applied to the rollout engines too), so
# do NOT default it on. Export it explicitly if you know the rollout engines
# tolerate it.
[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && ENV_ENTRIES+=("\"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\"")
[ -n "${TAU2_WORKERS_PER_NODE:-}" ] && ENV_ENTRIES+=("\"TAU2_WORKERS_PER_NODE\": \"${TAU2_WORKERS_PER_NODE}\"")
[ -n "${WANDB_MODE:-}" ] && ENV_ENTRIES+=("\"WANDB_MODE\": \"${WANDB_MODE}\"")
[ -n "${WANDB_DIR:-}" ] && ENV_ENTRIES+=("\"WANDB_DIR\": \"${WANDB_DIR}\"")
[ -n "${WEAVE_DISABLED:-}" ] && ENV_ENTRIES+=("\"WEAVE_DISABLED\": \"${WEAVE_DISABLED}\"")
[ -n "${AWS_PROFILE:-}" ] && ENV_ENTRIES+=("\"AWS_PROFILE\": \"${AWS_PROFILE}\"")

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    $(join_by $',\n    ' "${ENV_ENTRIES[@]}")
  },
  \"excludes\": [\".git\", \"wandb\", \"*.pt\", \"thirdparty/tau2-bench/data\"]
}"

# ── Submit ───────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
RAY_BIN=${RAY_BIN:-"${REPO_ROOT}/.venv/bin/ray"}
[ -x "${RAY_BIN}" ] || RAY_BIN=ray

"${RAY_BIN}" job submit --address="auto" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 ./thirdparty/slime/train_async.py \
  --actor-num-nodes ${ACTOR_NUM_NODES} \
  --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \
  --rollout-num-gpus ${ROLLOUT_NUM_GPUS} \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${DISTRIBUTED_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${EVAL_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${CUSTOM_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${WANDB_ARGS[@]}
