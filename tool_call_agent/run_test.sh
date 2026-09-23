#!/bin/bash
# Run GRPO training with fully async rollout
# NOTE: This script only submits the job, does NOT kill/start Ray
#
# Usage: ./run_grpo_async.sh <model_config> <run_name>
#   model_config: name of a file under tool_call_agent/models/ (without .sh)
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
   --prompt-data "/data/mcp-data/data_batches/3/training_data.jsonl"
   --input-key index
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 300
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 16
   --balance-data
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --over-sampling-batch-size 128
   --dynamic-sampling-filter-path
     slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
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
   --wandb-project tool-call-agent
   --wandb-group ${RUN_NAME}
)

CUSTOM_ARGS=(
   --custom-generate-function-path tool_call_agent.generate.generate
   --custom-rollout-log-function-path shared.rollout_log.log_rollout_data
   --use-distributed-post
   # --load-debug-rollout-data /tmp/instance_storage/debug/data_{rollout_id}.pt
   # --save-debug-rollout-data /tmp/instance_storage/debug1/data_{rollout_id}.pt
   # --custom-rm-path generate.reward_func
)

# ── Runtime environment ──

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"JUDGE_MODEL\": \"/data/base_models/Qwen/Qwen3-30B-A3B\",
    \"TRAJECTORY_PATH\": \"/data/mcp-data/data_batches/3\",
    \"SERVERS_PATH\": \"/data/mcp-data/mcp_servers_joined.json\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"EXCLUDE_ON_DROP_REASONS\": \"zero_std_1.0\",
    \"MAX_STEP_LAG\": \"2\",
    \"MOONCAKE_PROTOCOL\": \"efa\",
    \"FI_PROVIDER\": \"efa\",
    \"FI_EFA_USE_DEVICE_RDMA\": \"1\",
    \"FI_EFA_FORK_SAFE\": \"1\",
    \"RDMAV_FORK_SAFE\": \"1\",
    \"NCCL_SOCKET_IFNAME\": \"^lo,docker\",
    \"TOOL_CALL_PARSER\": \"${TOOL_CALL_PARSER:-qwen25}\",
    \"SLIME_DISABLE_PROCESSOR\": \"${SLIME_DISABLE_PROCESSOR:-0}\",
    \"SLIME_ROLLOUT_FULL_LOGPROBS\": \"${SLIME_ROLLOUT_FULL_LOGPROBS:-0}\",
    \"SLIME_ROLLOUT_FULL_ROUTING\": \"${SLIME_ROLLOUT_FULL_ROUTING:-0}\",
    \"LOCAL_WORLD_SIZE\": \"${ACTOR_NUM_GPUS_PER_NODE:-8}\",
    \"RAY_DEDUP_LOGS\": \"0\",
    \"UCCL_DEBUG\": \"INFO\",
    \"UCCL_DEBUG_SUBSYS\": \"ALL\",
    \"UCCL_DEBUG_VLOG_LEVEL\": \"2\"
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
