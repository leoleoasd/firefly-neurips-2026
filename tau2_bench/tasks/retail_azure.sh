#!/bin/bash
# Task configuration — tau2-bench RETAIL domain with an Azure OpenAI user simulator.
#
# Same as tasks/retail.sh but the user simulator is served by Azure OpenAI
# (via litellm) instead of a local sglang server — no user-sim sglang job needed.
#
# Requires AZURE_API_KEY to be exported before launching. The Azure path is
# selected in tau2_env_workers.py:_resolve_user_sim() when AZURE_USER_SIM_MODEL
# is set and USER_SIM_MODEL is NOT.

# Reuse all retail rollout/reward/GRPO config.
source "$(dirname -- "${BASH_SOURCE[0]}")/retail.sh"

# Switch the user simulator from local sglang to Azure OpenAI.
unset USER_SIM_MODEL
export AZURE_USER_SIM_MODEL=${AZURE_USER_SIM_MODEL:-"Azure/gpt-5.4"}
export AZURE_API_KEY=${AZURE_API_KEY:?"AZURE_API_KEY must be set (export it before running)"}
export AZURE_API_BASE=${AZURE_API_BASE:?"AZURE_API_BASE must be set (your Azure OpenAI endpoint, e.g. https://<your-resource>.openai.azure.com/)"}
export AZURE_API_VERSION=${AZURE_API_VERSION:-"2024-12-01-preview"}
