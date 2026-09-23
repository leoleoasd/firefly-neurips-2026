#!/bin/bash
# Run GRPO training for web agent with browser environment
# NOTE: This script only submits the job, does NOT kill/start Ray
#
# Usage: ./run_grpo_async.sh <model_config> <run_name>
#   model_config: name of a file under web_agent/models/ (without .sh)
#   run_name:     wandb group / checkpoint subdirectory name

set -ex

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <model_config> <run_name>" >&2
  echo "  e.g. $0 qwen3-30B-A3B my_experiment" >&2
  exit 1
fi

MODEL_CONFIG_NAME="$1"
RUN_NAME="$2"

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
MAX_CONCURRENT_CONTAINER_LAUNCHES=80
MAX_CONCURRENT_CONTAINERS_RUNNING=1024

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# ── Model-specific config (sets MODEL_ARGS, CKPT_ARGS, PERF_ARGS, SGLANG_ARGS, MISC_ARGS, etc.) ──
MODEL_CONFIG_FILE="${SCRIPT_DIR}/models/${MODEL_CONFIG_NAME}.sh"
if [ ! -f "${MODEL_CONFIG_FILE}" ]; then
  echo "Error: model config not found: ${MODEL_CONFIG_FILE}" >&2
  exit 1
fi
source "${MODEL_CONFIG_FILE}"

# ── Task / dataset config (model-independent) ──

ROLLOUT_ARGS=(
   --rollout-function-path shared.fully_async_rollout.generate_rollout_fully_async
   --data-source-path shared.data_source.RolloutDataSourceWithExclusion
   --prompt-data "${SCRIPT_DIR}/data/train_shopping_shopping_admin_gitlab.jsonl"
   --input-key index
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 200
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 12
   --balance-data
   --rollout-batch-size 16
   --n-samples-per-prompt 12
   --over-sampling-batch-size 64
   --dynamic-sampling-filter-path
     slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --save-debug-rollout-data /data/debug_30b_web/${RUN_NAME}/data_{rollout_id}.pt
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project web-agent
   --wandb-group ${RUN_NAME}
)

CUSTOM_ARGS=(
   --custom-generate-function-path web_agent.generate.generate
   --custom-rollout-log-function-path shared.rollout_log.log_rollout_data
   # --debug-rollout-only
)

# ── Runtime environment ──

# Browser environment configuration
# Set INCUS_SERVER_URL to point to container orchestrator
# Set PROXY_SERVER for host rewriting proxy
# Set MAX_CONCURRENT_CONTAINER_LAUNCHES to limit concurrent container launches (default: 8)
# Set MAX_CONCURRENT_CONTAINERS_RUNNING to limit total running containers (default: 64)
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"INCUS_SERVER_URL\": \"${INCUS_SERVER_URL:-http://127.0.0.1:8001}\",
    \"PROXY_SERVER\": \"${PROXY_SERVER:-http://localhost:8080}\",
    \"PROXY_ENABLED\": \"${PROXY_ENABLED:-true}\",
    \"BROWSER_HEADLESS\": \"${BROWSER_HEADLESS:-true}\",
    \"MAX_CONCURRENT_CONTAINER_LAUNCHES\": \"${MAX_CONCURRENT_CONTAINER_LAUNCHES:-32}\",
    \"MAX_CONCURRENT_CONTAINERS_RUNNING\": \"${MAX_CONCURRENT_CONTAINERS_RUNNING:-512}\",
    \"AWS_PROFILE\": \"${AWS_PROFILE:-default}\",
    \"NUM_ASYNC_ROLLOUT_WORKERS\": \"${NUM_ASYNC_ROLLOUT_WORKERS:-16}\",
    \"MOONCAKE_PROTOCOL\": \"efa\",
    \"FI_PROVIDER\": \"efa\",
    \"FI_EFA_USE_DEVICE_RDMA\": \"1\",
    \"FI_EFA_FORK_SAFE\": \"1\",
    \"RDMAV_FORK_SAFE\": \"1\",
    \"NCCL_SOCKET_IFNAME\": \"^lo,docker\",
    \"TOOL_CALL_PARSER\": \"${TOOL_CALL_PARSER:-qwen25}\"
  },
  \"excludes\": [\".git\"]
}"

ray job submit --address=auto \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ./thirdparty/slime/train_async.py \
   --actor-num-nodes ${ACTOR_NUM_NODES:-2} \
   --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE:-8} \
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS:-8} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
