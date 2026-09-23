#!/bin/bash
# Model-specific configuration for Qwen3.6-27B (dense) tau2-bench training.
#
# Qwen3.6-27B shares the qwen3_5 dense architecture, so we reuse the upstream
# slime model script `qwen3.5-27B.sh` for MODEL_ARGS (same as tool_call_agent's
# qwen3.6-27B config). Dense => no MoE routing replay (MOE_ARGS empty).
#
# Sourced by run_grpo_async.sh AFTER the task config. Expects REPO_ROOT, RUN_NAME.
# Task configs may export MAX_TOKENS_PER_GPU / SAVE_INTERVAL / ROLLOUT_NUM_GPUS
# to override the per-experiment knobs below.

source "${REPO_ROOT}/thirdparty/slime/scripts/models/qwen3.5-27B.sh"

# ── Checkpoint paths (override via env) ──────────────────────────────────────
CKPT_DIR=${CKPT_DIR:-"/data/checkpoints/tau/Qwen"}
AGENT_HF_CHECKPOINT=${AGENT_HF_CHECKPOINT:-"${CKPT_DIR}/Qwen3.6-27B"}
AGENT_MEGATRON_CHECKPOINT=${AGENT_MEGATRON_CHECKPOINT:-"${AGENT_HF_CHECKPOINT}_torch_dist"}
AGENT_SAVE_DIR=${AGENT_SAVE_DIR:-"${CKPT_DIR}/Qwen3.6-27B-${RUN_NAME}"}

CKPT_ARGS=(
  --hf-checkpoint ${AGENT_HF_CHECKPOINT}
  --ref-load ${AGENT_MEGATRON_CHECKPOINT}
  --load ${AGENT_SAVE_DIR}
  --save ${AGENT_SAVE_DIR}
  --save-interval ${SAVE_INTERVAL:-2}
)

DISTRIBUTED_ARGS=(
  --distributed-backend nccl
  --use-distributed-optimizer
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
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
  --qwen-gdn-backend flashqla
)

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

# Dense model — no MoE routing replay.
MOE_ARGS=()

# ── Cluster size (override via env) ──────────────────────────────────────────
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-16}

# sglang tool-call parser (forwarded into TOOL_CALL_PARSER by run_grpo_async.sh)
TOOL_CALL_PARSER=qwen3_coder
export TOOL_CALL_PARSER

# Qwen3.6-27B ships a VLM-style HF config, so AutoProcessor returns a real
# Qwen3VLProcessor even though we train text-only. Skip processor loading
# (gated by shared/data_source.py) for this text-only RL run.
SLIME_DISABLE_PROCESSOR=1
export SLIME_DISABLE_PROCESSOR
