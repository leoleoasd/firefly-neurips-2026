#!/bin/bash
# Model-specific configuration for Qwen3-30B-A3B
#
# Sourced by run_grpo_async.sh. Expects REPO_ROOT, RUN_NAME to be set.

MODEL_CONFIG=qwen3-30B-A3B
HF_MODEL_NAME=Qwen/Qwen3-30B-A3B-Thinking-2507
MODEL_ARGS_ROTARY_BASE=10000000 # thinking 2507
source "${REPO_ROOT}/thirdparty/slime/scripts/models/${MODEL_CONFIG}.sh"

MODEL_DIR=/data/base_models/
CKPT_DIR=/data/checkpoints/mcp/

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}${HF_MODEL_NAME}/
   --ref-load ${MODEL_DIR}/megatron/${MODEL_CONFIG}_torch_dist/
   --save-debug-rollout-data /data/debug/mcp/${RUN_NAME}-30ba3b_{rollout_id}.pt
   # --load-debug-rollout-data /data/debug/mcp/30ba3b_{rollout_id}.pt
   # --debug-train-only
   # --load /tmp/instance_storage/checkpoints/${MODEL_CONFIG}_slime/
   # --no-load-optim
   --save ${CKPT_DIR}/${MODEL_CONFIG}-${RUN_NAME}/
   --save-interval 20
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 8
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   # --recompute-num-layers 8
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   # NOTE: deep_ep / UCCL doesn't work on AWS EFA (only Mellanox IB).
   # Stick with the upstream default `alltoall` dispatcher (pure NCCL, no deep_ep).
   # --moe-token-dispatcher-type=alltoall
   # deepep disabled for now — re-enable by swapping the dispatcher type above back to
   # `flex` and uncommenting the backend line below.
   --moe-token-dispatcher-type=flex
   --moe-flex-dispatcher-backend=deepep
)

SGLANG_ARGS=(
   --num-gpus-per-node 8
   --rollout-num-gpus-per-engine 1
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

   --use-routing-replay
   --use-rollout-routing-replay
   --use-fault-tolerance

   --use-tis
)

# Rollout log-prob mode (read by shared/sample_helpers.py at import on the rollout workers,
# forwarded into the Ray runtime env by run_grpo_async.sh):
#   0 = incremental — keep each turn's generation-time logprobs (this variant).
#   1 = full update  — overwrite each turn from sglang's prefill input_token_logprobs.
# See qwen3-30B-A3B-full-logprobs.sh for the full-update variant.
SLIME_ROLLOUT_FULL_LOGPROBS=0
export SLIME_ROLLOUT_FULL_LOGPROBS

# Rollout routing-replay mode (independent of SLIME_ROLLOUT_FULL_LOGPROBS):
#   0 = incremental — keep each turn's actual generation-time routing, append the new tail.
#   1 = full update  — overwrite each turn with this turn's full-sequence recomputed routing.
SLIME_ROLLOUT_FULL_ROUTING=0
export SLIME_ROLLOUT_FULL_ROUTING

# Training cluster size
ACTOR_NUM_NODES=2
ACTOR_NUM_GPUS_PER_NODE=8
ROLLOUT_NUM_GPUS=32
