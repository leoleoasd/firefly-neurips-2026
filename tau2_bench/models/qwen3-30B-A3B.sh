#!/bin/bash
# Model-specific configuration for Qwen3-30B-A3B (MoE) tau2-bench training.
#
# Sourced by run_grpo_async.sh. Expects RUN_NAME to be set.
# Sets: MODEL_ARGS, DISTRIBUTED_ARGS, CKPT_ARGS, PERF_ARGS, SGLANG_ARGS,
#       MISC_ARGS, MOE_ARGS, and the cluster-size vars.
#
# Task configs (tasks/*.sh) may export these before this file is sourced to
# override per-experiment knobs:
#   MAX_TOKENS_PER_GPU   default 24576   (telecom uses 16384)
#   SAVE_INTERVAL        default 2       (telecom uses 5)
#   ACTOR_NUM_NODES      default 1
#   ACTOR_NUM_GPUS_PER_NODE default 8
#   ROLLOUT_NUM_GPUS     default 16      (telecom uses 8)

# ── Checkpoint paths (override via env) ──────────────────────────────────────
CKPT_DIR=${CKPT_DIR:-"/data/checkpoints/tau/Qwen"}
AGENT_HF_CHECKPOINT=${AGENT_HF_CHECKPOINT:-"${CKPT_DIR}/Qwen3-30B-A3B-Thinking-2507"}
AGENT_MEGATRON_CHECKPOINT=${AGENT_MEGATRON_CHECKPOINT:-"${AGENT_HF_CHECKPOINT}_torch_dist"}
AGENT_SAVE_DIR=${AGENT_SAVE_DIR:-"${CKPT_DIR}/Qwen3-30B-A3B-Thinking-2507-${RUN_NAME}"}

# ── Model architecture — Qwen3-30B-A3B (MoE), matches HF config.json ──────────
NLAYERS=48
FIRST_K_DENSE_REPLACE=0
arr=()
for ((i = 0; i < NLAYERS; i++)); do
  if ((i < FIRST_K_DENSE_REPLACE)); then arr+=(0); else arr+=(1); fi
done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

MODEL_ARGS=(
  --disable-bias-linear
  --qk-layernorm
  --group-query-attention
  --num-attention-heads 32
  --num-query-groups 4
  --kv-channels 128
  --num-layers 48
  --hidden-size 2048
  --ffn-hidden-size 6144

  --normalization RMSNorm
  --position-embedding-type rope
  --norm-epsilon 1e-6
  --rotary-percent 1.0
  --swiglu
  --untie-embeddings-and-output-weights
  --vocab-size 151936

  --rotary-base "${MODEL_ARGS_ROTARY_BASE:-10000000}"

  # moe
  --moe-ffn-hidden-size 768
  --moe-router-score-function softmax
  --moe-token-dispatcher-type alltoall
  --moe-router-topk 8
  --moe-layer-freq $MOE_LAYER_FREQ
  --num-experts 128
  --moe-grouped-gemm
  --moe-token-drop-policy probs
  --moe-router-dtype fp32
  --moe-permute-fusion
  --moe-aux-loss-coeff 0
)

DISTRIBUTED_ARGS=(
  --distributed-backend nccl
  --use-distributed-optimizer
)

CKPT_ARGS=(
  --hf-checkpoint ${AGENT_HF_CHECKPOINT}
  --ref-load ${AGENT_MEGATRON_CHECKPOINT}
  --load ${AGENT_SAVE_DIR}
  --save ${AGENT_SAVE_DIR}
  --save-interval ${SAVE_INTERVAL:-2}
)

PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 4
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-24576}
)

SGLANG_ARGS=(
  --num-gpus-per-node 8
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.7
  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --log-passrate
)

# MoE routing replay
MOE_ARGS=(
  --use-rollout-routing-replay
  --use-routing-replay
  --use-slime-router
)

# ── Cluster size (override via env) ──────────────────────────────────────────
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-16}
