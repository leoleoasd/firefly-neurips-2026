#!/bin/bash
# Model-specific configuration for Qwen3-30B-A3B (web agent)
#
# Sourced by run_grpo_async.sh. Expects REPO_ROOT, RUN_NAME to be set.

MODEL_CONFIG=qwen3-30B-A3B
HF_MODEL_NAME=Qwen/Qwen3-30B-A3B-Thinking-2507
MODEL_ARGS_ROTARY_BASE=10000000 # thinking 2507
source "${REPO_ROOT}/thirdparty/slime/scripts/models/${MODEL_CONFIG}.sh"

NUM_ASYNC_ROLLOUT_WORKERS=32

MODEL_DIR=/data/base_models/
CKPT_DIR=/data/checkpoints/rl_web_agent/

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}${HF_MODEL_NAME}/
   # --ref-load ${MODEL_DIR}${HF_MODEL_NAME}_torch_dist/
   # --ref-load /data/checkpoints/rl_web_agent/rl_base/4b_3epoch/
   --ref-load /data/checkpoints/rl_web_agent/rl_base/30b_3epoch
   # --load /tmp/instance_storage/resume/
   # --no-load-optim
   --save ${CKPT_DIR}/${MODEL_CONFIG}-${RUN_NAME}/
   --save-interval 20
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 16
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --micro-batch-size 1
   # --use-dynamic-batch-size
   # --max-tokens-per-gpu 4096


   # --fine-grained-activation-offloading
   # --offload-modules core_attn moe_act expert_fc1
   --moe-token-dispatcher-type=flex
   --moe-flex-dispatcher-backend=deepep
)

SGLANG_ARGS=(
   # --rollout-num-gpus-per-engine 4
   # --sglang-mem-fraction-static 0.8


   # --sglang-moe-a2a-backend deepep
   # --sglang-deepep-mode auto
   # --prefill-num-servers 1


   --rollout-num-gpus-per-engine 32
   --sglang-mem-fraction-static 0.8
   --sglang-enable-dp-attention
   --sglang-dp-size 32
   --sglang-ep-size 32
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head
   --sglang-disable-cuda-graph

   --sglang-moe-a2a-backend deepep
   --sglang-deepep-mode auto
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-passrate

   --use-routing-replay
   --use-rollout-routing-replay
   --use-fault-tolerance
   --use-tis
)

# Training cluster size
ACTOR_NUM_NODES=4
ACTOR_NUM_GPUS_PER_NODE=8
ROLLOUT_NUM_GPUS=32
