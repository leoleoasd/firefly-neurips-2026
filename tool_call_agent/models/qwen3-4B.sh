#!/bin/bash
# Model-specific configuration for Qwen3-4B
#
# Sourced by run_grpo_async.sh. Expects REPO_ROOT, RUN_NAME to be set.

MODEL_CONFIG=qwen3-4B-Instruct-2507
HF_MODEL_NAME=Qwen/Qwen3-4B-Thinking-2507
source "${REPO_ROOT}/thirdparty/slime/scripts/models/${MODEL_CONFIG}.sh"

MODEL_DIR=/data/base_models/
CKPT_DIR=/data/checkpoints/mcp/

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}${HF_MODEL_NAME}/
   --ref-load /data/base_models/Qwen/Qwen3-4B-Thinking-2507_torch_dist/
   # --load ...
   # --no-load-optim
   --save ${CKPT_DIR}/${MODEL_CONFIG}-${RUN_NAME}/
   --save-interval 20
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
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
ACTOR_NUM_NODES=2
ACTOR_NUM_GPUS_PER_NODE=8
ROLLOUT_NUM_GPUS=16
