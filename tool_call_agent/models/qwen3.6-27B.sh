#!/bin/bash
# Model-specific configuration for Qwen3.6-27B (dense).
#
# Qwen3.6-27B shares the qwen3_5 dense architecture, so we reuse the upstream
# slime model script `qwen3.5-27B.sh` (megatron_to_hf handles qwen3_6 via the
# same convert_qwen3_5_to_hf path).
#
# Sourced by run_grpo_async.sh. Expects REPO_ROOT, RUN_NAME to be set.

MODEL_CONFIG=qwen3.6-27B
HF_MODEL_NAME=Qwen/Qwen3.6-27B
source "${REPO_ROOT}/thirdparty/slime/scripts/models/qwen3.5-27B.sh"

MODEL_DIR=/data/base_models/
CKPT_DIR=/data/checkpoints/mcp/

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}${HF_MODEL_NAME}/
   --ref-load ${MODEL_DIR}${HF_MODEL_NAME}_torch_dist/
   --save ${CKPT_DIR}/${MODEL_CONFIG}-${RUN_NAME}/
   --save-interval 20
   --save-debug-rollout-data /data/debug/mcp/${RUN_NAME}-36_27b_{rollout_id}.pt
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --decoder-last-pipeline-num-layers 30
   --context-parallel-size 2
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
   --qwen-gdn-backend flashqla
)

SGLANG_ARGS=(
   --num-gpus-per-node 8
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   --log-passrate

   --use-fault-tolerance
   --use-tis
)

# Training cluster size
ACTOR_NUM_NODES=2
ACTOR_NUM_GPUS_PER_NODE=8
ROLLOUT_NUM_GPUS=16

# sglang tool-call parser (read by run_grpo_async.sh into TOOL_CALL_PARSER env)
TOOL_CALL_PARSER=qwen3_coder
export TOOL_CALL_PARSER

# Qwen3.6-27B ships a VLM-style HF config, so AutoProcessor returns a real
# Qwen3VLProcessor even though we train text-only. Skip processor loading
# (gated by shared/data_source.py) for this text-only RL run.
SLIME_DISABLE_PROCESSOR=1
export SLIME_DISABLE_PROCESSOR
