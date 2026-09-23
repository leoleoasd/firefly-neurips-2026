#!/bin/bash
# Qwen3-30B-A3B, FULL-UPDATE rollout log-prob variant.
#
# Identical to qwen3-30B-A3B.sh except the rollout log-prob mode: instead of keeping each
# turn's generation-time logprobs (incremental), every turn OVERWRITES rollout_log_probs from
# sglang's prefill-recomputed input_token_logprobs (returned for free via logprob_start_len=0,
# no extra forward pass). Implemented in shared/sample_helpers.py, toggled by the env var below.
#
# Sourced by run_grpo_async.sh. Expects REPO_ROOT, RUN_NAME to be set.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Reuse the full incremental config (MODEL_ARGS, CKPT_ARGS, PERF_ARGS, SGLANG_ARGS, MISC_ARGS,
# cluster sizes, etc.) so the two variants can't drift apart.
source "${SCRIPT_DIR}/qwen3-30B-A3B.sh"

# Flip on full-update mode for log-probs only (overrides the =0 set by the sourced config).
# Routing replay (SLIME_ROLLOUT_FULL_ROUTING) is independent and stays at the base value;
# set it here too if you also want full-update routing.
SLIME_ROLLOUT_FULL_LOGPROBS=1
export SLIME_ROLLOUT_FULL_LOGPROBS
