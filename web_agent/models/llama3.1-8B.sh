#!/bin/bash
# Model-specific configuration for Llama 3.1 8B (web agent, H100 80GB)
#
# Sourced by run_grpo_async.sh. Expects REPO_ROOT, RUN_NAME to be set.

MODEL_CONFIG=llama3.1-8B-Instruct
HF_MODEL_NAME=checkpoints/rl_web_agent/web_agent_sft_128k_llama-3.1-lr1e5-fix-chat-template-hf/iter_0000044_hf
source "${REPO_ROOT}/thirdparty/slime/scripts/models/${MODEL_CONFIG}.sh"

# Override tool call parser for Llama 3.1
export TOOL_CALL_PARSER=llama3

NUM_ASYNC_ROLLOUT_WORKERS=32

MODEL_DIR=/data/
CKPT_DIR=/data/checkpoints/rl_web_agent/

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}${HF_MODEL_NAME}/
   --ref-load /data/checkpoints/rl_web_agent/rl_base/llama3.1-lr1e5-fix-2epoch/
   # --load ...
   # --no-load-optim
   --save ${CKPT_DIR}/llama3.1-8B-${RUN_NAME}/
   --save-interval 20
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 8
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

SGLANG_ARGS=(
   --num-gpus-per-node 8
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-passrate

   --use-fault-tolerance
   --use-tis
)

# Training cluster size
ACTOR_NUM_NODES=4
ACTOR_NUM_GPUS_PER_NODE=8
ROLLOUT_NUM_GPUS=32
