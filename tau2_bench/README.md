# Tau2-Bench: Multi-Turn Agent RL Training

Train a Qwen3-30B-A3B (MoE) agent on Tau2-Bench multi-turn tasks with GRPO + SLIME on a multi-node Ray cluster.

## Quick Start

### 1. Prepare models & environment

Run on every node. `MODEL_S3_ROOT` and `DATA_S3_ROOT` are **required** (no public
defaults): point them at the S3 bucket/prefix hosting the agent + user-sim weights
and the task data. Pick the domain; destinations default to `/data/checkpoints/tau/Qwen`
(override with `CKPT_DIR=...`), matching `models/qwen3-30B-A3B.sh` and `tasks/<domain>.sh`:

```bash
export MODEL_S3_ROOT=s3://<your-bucket>/tau2/models   # agent + user-sim weights
export DATA_S3_ROOT=s3://<your-bucket>/tau2/data      # retail/telecom task data

TAU2_DOMAIN=retail  bash prepare_model_env.sh   # retail: alldatasft agent + 30B user-sim
TAU2_DOMAIN=telecom bash prepare_model_env.sh   # telecom: sft-base-400 agent + 235B user-sim
```

Then bring up the Ray cluster (also on every node):

```bash
bash launch_ray.sh
```

### 2. Start the RM router

On the head node, launch the reward-model / user-sim router. It registers itself in `sglang_registry` under `rm_router` so training can discover it.

```bash
python scripts/rm_router.py
```

### 3. Launch the user simulator (SGLang)

Skip this step if the user simulator is Azure OpenAI or Bedrock (see step 4).
Otherwise submit the user-sim SGLang server as a Ray job:

```bash
ray job submit --address=auto \
  --working-dir . \
  -- python scripts/sglang_job.py --num-gpus 8 --num-nodes 2 \
    --model /tmp/instance_storage/Qwen/Qwen3-30B-A3B-Thinking-2507-user-sim \
    --context-length 131072 --reasoning-parser deepseek-r1 --tool-call-parser qwen \
    --tp 8
```

Wait until you see `[RewardSGLang] registered: ...` before continuing. Tune `--num-nodes`, `--model`, and `--tp` for a different / larger user-sim (e.g. Qwen3-235B).

### 4. Start RL training

Config is split into two independent layers, like `tool_call_agent`:

- `models/<name>.sh` — architecture, checkpoint paths, perf/sglang/MoE args, cluster size
- `tasks/<name>.sh` — domain, reward shaping, rollout/eval/GRPO args, user simulator

```bash
# ./run_grpo_async.sh <model_config> <task_config> <run_name>
./run_grpo_async.sh qwen3-30B-A3B retail   my_experiment   # retail, local sglang user-sim
./run_grpo_async.sh qwen3-30B-A3B telecom  my_experiment   # telecom, 235B user-sim
AZURE_API_KEY=... AZURE_API_BASE=https://<your-resource>.openai.azure.com/ \
  ./run_grpo_async.sh qwen3-30B-A3B retail_azure my_experiment   # retail, Azure user-sim
./run_grpo_async.sh qwen3-30B-A3B telecom_smoke smoke      # tiny end-to-end check
```

Task configs under `tasks/`: `retail`, `telecom` (local sglang user-sim),
`retail_azure` (Azure OpenAI user-sim), `telecom_smoke` (tiny offline check).
Model configs under `models/`: `qwen3-30B-A3B`, `qwen3.5-122B-A10B`, `qwen3.6-27B`,
`qwen3.6-35B-A3B`.

**User simulator.** Exactly one of these env vars must reach the rollout workers
(`run_grpo_async.sh` forwards all of them through the Ray runtime env):

- `USER_SIM_MODEL` — local sglang-served model via `rm_router` (set by the `retail`/`telecom` task configs; needs step 3)
- `AZURE_USER_SIM_MODEL` — Azure OpenAI deployment; also requires `AZURE_API_KEY` and `AZURE_API_BASE` (set by `retail_azure`)
- `BEDROCK_USER_SIM_MODEL` — AWS Bedrock model / inference-profile id (uses the standard AWS credential chain)

Reward penalties and most hyperparameters are env vars with defaults in the task
config — override them inline, e.g. `TAU2_CONSECUTIVE_SAME_TOOL_PENALTY=-0.05 ./run_grpo_async.sh ...`
to reproduce the old `consecutive_tool_penalty` variant. See `tasks/retail.sh`
for the full list.

## Cleanup

```bash
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
```
