#!/bin/bash
# Qwen3-30B-A3B, INCREMENTAL log-probs + FULL-UPDATE routing-replay variant.
#
# Identical to qwen3-30B-A3B.sh except rollout_routed_experts uses full-update mode: every turn
# OVERWRITES the routing with this turn's full-sequence recomputed routing
# (SLIME_ROLLOUT_FULL_ROUTING=1), while rollout_log_probs stays incremental
# (SLIME_ROLLOUT_FULL_LOGPROBS=0). The two toggles are independent; see shared/sample_helpers.py.
#
# Sourced by run_grpo_async.sh. Expects REPO_ROOT, RUN_NAME to be set.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Reuse the full incremental config (MODEL_ARGS, CKPT_ARGS, PERF_ARGS, SGLANG_ARGS, MISC_ARGS,
# cluster sizes, etc.) so the variants can't drift apart.
source "${SCRIPT_DIR}/qwen3-30B-A3B.sh"

# Incremental log-probs (explicit; matches the base default).
SLIME_ROLLOUT_FULL_LOGPROBS=0
export SLIME_ROLLOUT_FULL_LOGPROBS

# Full-update routing replay (overrides the =0 set by the sourced config).
SLIME_ROLLOUT_FULL_ROUTING=1
export SLIME_ROLLOUT_FULL_ROUTING
