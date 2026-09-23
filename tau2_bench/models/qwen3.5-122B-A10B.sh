#!/bin/bash
# Model-specific configuration for Qwen3.5-122B-A10B (MoE, qwen3_5_moe) tau2-bench.
#
# Architecture derived from the HF config.json of /data/base_models/Qwen/Qwen3.5-122B-A10B
# (256 experts, top-8, 48 all-MoE layers, hidden 3072, heads 32 / kv 2, head_dim 256,
# moe_ffn 1024, shared-expert 1024, vocab 248320, rope 1e7, untied embeddings).
#
# Sourced by run_grpo_async.sh AFTER the task config. Sets MODEL_ARGS, CKPT_ARGS,
# PERF_ARGS, SGLANG_ARGS, MISC_ARGS, MOE_ARGS, DISTRIBUTED_ARGS + cluster vars.

# ── Checkpoint paths (override via env) ──────────────────────────────────────
CKPT_DIR=${CKPT_DIR:-"/data/checkpoints/tau/Qwen"}
AGENT_HF_CHECKPOINT=${AGENT_HF_CHECKPOINT:-"/data/base_models/Qwen/Qwen3.5-122B-A10B"}
AGENT_MEGATRON_CHECKPOINT=${AGENT_MEGATRON_CHECKPOINT:-"/data/base_models/Qwen/Qwen3.5-122B-A10B_torch_dist"}
AGENT_SAVE_DIR=${AGENT_SAVE_DIR:-"${CKPT_DIR}/Qwen3.5-122B-A10B-${RUN_NAME}"}

# ── Model architecture — Qwen3.5-122B-A10B (MoE) — all layers MoE ────────────
NLAYERS=48
FIRST_K_DENSE_REPLACE=0
arr=()
for ((i = 0; i < NLAYERS; i++)); do
  if ((i < FIRST_K_DENSE_REPLACE)); then arr+=(0); else arr+=(1); fi
done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

MODEL_ARGS=(
  --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
  --disable-bias-linear
  --qk-layernorm
  --group-query-attention
  --num-attention-heads 32
  --num-query-groups 2
  --kv-channels 256
  --num-layers 48
  --hidden-size 3072
  --ffn-hidden-size 1024
  --use-gated-attention

  --normalization RMSNorm
  --apply-layernorm-1p
  --position-embedding-type rope
  --norm-epsilon 1e-6
  --rotary-percent 0.25
  --swiglu
  --untie-embeddings-and-output-weights
  --vocab-size 248320

  --rotary-base 10000000

  # moe
  --moe-ffn-hidden-size 1024
  --moe-shared-expert-intermediate-size 1024
  --moe-router-score-function softmax
  --moe-token-dispatcher-type alltoall
  --moe-router-topk 8
  --moe-layer-freq $MOE_LAYER_FREQ
  --num-experts 256
  --moe-grouped-gemm
  --moe-token-drop-policy probs
  --moe-router-dtype fp32
  --moe-permute-fusion
  --moe-aux-loss-coeff 0

  # qwen3.5 specific
  --attention-output-gate
  --moe-shared-expert-gate
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
  --save-interval ${SAVE_INTERVAL:-5}
)

# Actor parallelism: TP4 · PP4 · EP8 (= 32 GPUs at DP2). Optimizer offloaded to CPU
# (122B won't keep Adam states on-GPU). decoder-last-pipeline-num-layers balances PP4
# over 48 layers (12 per stage).
PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 4
  --decoder-last-pipeline-num-layers 12
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-8192}
  --qwen-gdn-backend flashqla
)

# Rollout engines: TP4 per engine (122B ~244GB bf16 fits in 4×140GB @ mem-frac 0.7).
SGLANG_ARGS=(
  --num-gpus-per-node 8
  --rollout-num-gpus-per-engine 4
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

# MoE routing replay (needed so rollout token routing is reproduced in training).
MOE_ARGS=(
  --use-rollout-routing-replay
  --use-routing-replay
  --use-slime-router
)

# sglang parsers
TOOL_CALL_PARSER=qwen3_coder
export TOOL_CALL_PARSER

# Qwen3.5 ships a VLM-style HF config; skip processor loading for text-only RL.
SLIME_DISABLE_PROCESSOR=1
export SLIME_DISABLE_PROCESSOR

# ── Cluster size (override via env) — 32 actor + 28 rollout + 4 user-sim = 64 ─
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-4}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-28}
