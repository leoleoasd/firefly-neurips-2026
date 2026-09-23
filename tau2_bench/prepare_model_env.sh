#!/usr/bin/env bash
# Per-node setup for tau2-bench training. Run on EVERY node.
#
# Syncs the repo, fetches the agent (SFT) + user-simulator weights from S3,
# converts the agent checkpoint to Megatron torch_dist, and downloads the
# domain task data. Destinations line up with the defaults baked into
# models/qwen3-30B-A3B.sh and tasks/<domain>.sh, so a plain
#   bash prepare_model_env.sh && ./run_grpo_async.sh qwen3-30B-A3B retail <run>
# works with no path overrides.
#
# Usage: [TAU2_DOMAIN=retail|telecom] [CKPT_DIR=...] bash prepare_model_env.sh
#
# Replaces the old per-target scripts (prepare_model_env{,_retail,_kube}.sh);
# the kube variant's path differences collapse to CKPT_DIR, and launching the
# user-sim sglang server is a separate step (see README step 3).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

TAU2_DOMAIN=${TAU2_DOMAIN:-retail}
CKPT_DIR=${CKPT_DIR:-/data/checkpoints/tau/Qwen}   # must match models/qwen3-30B-A3B.sh
MODEL_CONFIG=${MODEL_CONFIG:-qwen3-30B-A3B}        # slime model script (for torch_dist conversion)

# S3 locations of the agent/user-sim weights and the task data. Required —
# there is no public default; host the artifacts yourself and point these at
# your bucket/prefix, e.g. MODEL_S3_ROOT=s3://my-bucket/tau2/models
MODEL_S3_ROOT="${MODEL_S3_ROOT:?set MODEL_S3_ROOT (s3:// bucket/prefix hosting the agent + user-sim weights)}"
DATA_S3_ROOT="${DATA_S3_ROOT:?set DATA_S3_ROOT (s3:// bucket/prefix hosting the task data)}"

AGENT_DIR="${CKPT_DIR}/Qwen3-30B-A3B-Thinking-2507"
DATA_DIR="${SCRIPT_DIR}/data"

# ── Sync repo + env ──────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"
git reset --hard && git clean -fd
git pull
uv sync
condax install s5cmd

# ── Fetch agent (SFT) + user-simulator weights (domain-specific) ─────────────
case "${TAU2_DOMAIN}" in
  retail)
    s5cmd cp --sp "${MODEL_S3_ROOT}/Qwen3-30B-A3B-Thinking-2507-alldatasft-fix-context-fix-data-exported/*" "${AGENT_DIR}/"
    s5cmd cp --sp "${MODEL_S3_ROOT}/Qwen3-30B-A3B-Thinking-2507/*" "${CKPT_DIR}/Qwen3-30B-A3B-Thinking-2507-user-sim/"
    ;;
  telecom)
    s5cmd cp --sp "${MODEL_S3_ROOT}/sft-base-model-400/*" "${AGENT_DIR}/"
    s5cmd cp --sp "${MODEL_S3_ROOT}/Qwen3-235B-A22B-Thinking-2507/*" "${CKPT_DIR}/Qwen3-235B-A22B-Thinking-2507/"
    ;;
  *) echo "Unknown TAU2_DOMAIN='${TAU2_DOMAIN}' (expected retail|telecom)" >&2; exit 1 ;;
esac

# ── Convert agent HF checkpoint -> Megatron torch_dist ───────────────────────
pushd "${REPO_ROOT}/thirdparty/slime" >/dev/null
source "scripts/models/${MODEL_CONFIG}.sh"
python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${AGENT_DIR}" \
  --save "${AGENT_DIR}_torch_dist"
popd >/dev/null

# ── Download task data into tau2_bench/data (tasks/*.sh read from here) ───────
case "${TAU2_DOMAIN}" in
  retail)
    aws s3 cp "${DATA_S3_ROOT}/retail/combined_rl_robust_tasks.jsonl" "${DATA_DIR}/retail_train_tasks.jsonl"
    cp "${DATA_DIR}/retail_train_tasks.jsonl" "${DATA_DIR}/retail_test_tasks.jsonl"
    aws s3 cp "${DATA_S3_ROOT}/retail/db.json" "${REPO_ROOT}/thirdparty/tau2-bench/data/tau2/domains/retail/db.json"
    ;;
  telecom)
    aws s3 cp "${DATA_S3_ROOT}/telecom_train_tasks.jsonl" "${DATA_DIR}/telecom_train_tasks.jsonl"
    aws s3 cp "${DATA_S3_ROOT}/telecom_test_tasks.jsonl"  "${DATA_DIR}/telecom_test_tasks.jsonl"
    ;;
esac

echo "prepare_model_env done: domain=${TAU2_DOMAIN} ckpt_dir=${CKPT_DIR}"
