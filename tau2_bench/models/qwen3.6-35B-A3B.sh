#!/bin/bash
# Model-specific configuration for Qwen3.6-35B-A3B (MoE, qwen3_5_moe) tau2-bench.
#
# Reuses slime's OFFICIAL MODEL_ARGS from scripts/models/qwen3.5-35B-A3B.sh
# (num-attention-heads 16, hidden 2048, 40 layers, 256 experts top-8, attn_output_gate,
# moe-shared-expert-gate) — verified to match /data/base_models/Qwen/Qwen3.6-35B-A3B
# config.json exactly. This is the known-good reference arch for get_qwen3_5_spec.
#
# Sourced by run_grpo_async.sh AFTER the task config. Sets CKPT/PERF/SGLANG/MISC/MOE
# + cluster vars; MODEL_ARGS come from the upstream script.

source "${REPO_ROOT}/thirdparty/slime/scripts/models/qwen3.5-35B-A3B.sh"

# ── Checkpoint paths (override via env) ──────────────────────────────────────
CKPT_DIR=${CKPT_DIR:-"/data/checkpoints/tau/Qwen"}
AGENT_HF_CHECKPOINT=${AGENT_HF_CHECKPOINT:-"/data/base_models/Qwen/Qwen3.6-35B-A3B"}
AGENT_MEGATRON_CHECKPOINT=${AGENT_MEGATRON_CHECKPOINT:-"/data/base_models/Qwen/Qwen3.6-35B-A3B_torch_dist"}
AGENT_SAVE_DIR=${AGENT_SAVE_DIR:-"${CKPT_DIR}/Qwen3.6-35B-A3B-${RUN_NAME}"}

CKPT_ARGS=(
  --hf-checkpoint ${AGENT_HF_CHECKPOINT}
  --ref-load ${AGENT_MEGATRON_CHECKPOINT}
  --load ${AGENT_SAVE_DIR}
  --save ${AGENT_SAVE_DIR}
  --save-interval ${SAVE_INTERVAL:-5}
)

DISTRIBUTED_ARGS=(
  --distributed-backend nccl
  --use-distributed-optimizer
)

# Actor parallelism mirrors the known-good upstream test
# (tests/test_qwen3.6_35B_A3B_pd_mooncake.py, training path): TP2 · PP1 · CP2 · EP8
# = 8 GPUs (1 node). Optimizer offloaded to CPU.
PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 2
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-8192}
  --qwen-gdn-backend flashqla
)

# Rollout engines: TP2 per engine (35B bf16 ~70GB fits in 2x140GB @ mem-frac 0.7).
SGLANG_ARGS=(
  --num-gpus-per-node 8
  --rollout-num-gpus-per-engine 2
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

# Qwen3.6 ships a VLM-style HF config; skip processor loading for text-only RL.
SLIME_DISABLE_PROCESSOR=1
export SLIME_DISABLE_PROCESSOR

# ── Cluster size (override via env) — 8 actor + 8 rollout + 4 user-sim ────────
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-8}
