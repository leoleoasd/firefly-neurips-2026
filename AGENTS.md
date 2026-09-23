# Common Commands (run from /data/slime_uv)

## Convert HF checkpoint to Megatron torch_dist

```bash
source thirdparty/slime/scripts/models/<MODEL_CONFIG>.sh && \
uv run python thirdparty/slime/tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint <HF_CHECKPOINT_PATH>/ \
    --save <OUTPUT_TORCH_DIST_PATH>/
```

Example (Qwen3-4B-Thinking-2507):
```bash
source thirdparty/slime/scripts/models/qwen3-4B-Instruct-2507.sh && \
uv run python thirdparty/slime/tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /data/base_models/Qwen/Qwen3-4B-Thinking-2507/ \
    --save /data/base_models/Qwen/Qwen3-4B-Thinking-2507_torch_dist/
```

## Run GRPO training (tool_call_agent)

```bash
./tool_call_agent/run_grpo_async.sh <model_config> <run_name>
```

- `model_config`: name of a file under `tool_call_agent/models/` (without `.sh`), e.g. `qwen3-30B-A3B`, `qwen3-4B`
- `run_name`: wandb group / checkpoint subdirectory name

## Run GRPO training (web_agent)

```bash
./web_agent/run_grpo_async.sh <model_config> <run_name>
```

## Evaluate tool_call_agent checkpoints

```bash
python tool_call_agent/evaluate.py \
  --checkpoint /data/checkpoints/mcp/<run>-hf/ \
  --data-batches /data/mcp-data/data_batches \
  --batch 3 \
  --task-file test_tasks.json \
  --servers /data/mcp-data/mcp_servers_joined.json \
  --output /data/checkpoints/mcp/eval_results \
  --tool-mode dag \
  --no-pin-node \
  --passes 16 \
  --only-missing
```

## Convert training checkpoints to HF format

```bash
bash scripts/convert_all_checkpoints.sh \
  --input-dir /data/checkpoints/mcp/<run> \
  --output-dir /data/checkpoints/mcp/<run>-hf \
  --origin-hf-dir /data/base_models/<HF_MODEL_NAME>
```

# Repository Structure

## Top-level packages (all installed via setuptools find, see pyproject.toml)

- `shared/` — Common utilities used by all agents: async rollout, http, sglang registry, sample helpers, timers
- `tool_call_agent/` — MCP tool-calling agent: generate loop, tool parsing, tool simulation, evaluation
- `web_agent/` — Browser-based web agent: generate loop, browser env, tool parsing
- `tau2_bench/` — tau2-bench (retail/telecom) agent integration: generate loop, model/task configs, per-node env prep
- `scripts/` — Cluster ops: sglang server launch, RM router, registry CLI, file broadcast, checkpoint conversion

## Thirdparty (git subtrees, do NOT edit directly)

- `thirdparty/slime/` — RL training framework (Megatron-based). Model configs live in `thirdparty/slime/scripts/models/`
- `thirdparty/rl_web_agent/` — Web agent environment and evaluation
- `thirdparty/bfcl/` — Berkeley Function Calling Leaderboard eval
- `thirdparty/tau2-bench/` — tau2-bench task domains and evaluation harness

## Key paths on disk

- `/data/base_models/` — HF model weights and converted torch_dist checkpoints
- `/data/checkpoints/` — Training checkpoints (`mcp/` for tool_call_agent, `rl_web_agent/` for web_agent)
- `/data/mcp-data/` — MCP training data (data_batches, mcp_servers_joined.json)

# Architecture Notes

## Training script split

`run_grpo_async.sh` sources a model config from `models/<name>.sh` which sets: `MODEL_ARGS`, `CKPT_ARGS`, `PERF_ARGS`, `SGLANG_ARGS`, `MISC_ARGS`, cluster size vars (`ACTOR_NUM_NODES`, `ROLLOUT_NUM_GPUS`). The main script sets task/dataset config: `ROLLOUT_ARGS`, `GRPO_ARGS`, `OPTIMIZER_ARGS`, `WANDB_ARGS`, `CUSTOM_ARGS`.

## Environment variables that control behavior

| Variable | Where | Purpose |
|---|---|---|
| `TOOL_CALL_PARSER` | runtime env / code defaults | sglang tool call parser name (e.g. `qwen25`, `llama`) |
| `JUDGE_MODEL` | runtime env / evaluate.py | Path to judge model for reward scoring |
| `TRAJECTORY_PATH` | runtime env | Directory with trajectory data for tool simulation |
| `SERVERS_PATH` | runtime env | Path to `mcp_servers_joined.json` |
| `WANDB_API_KEY` | container runtime env | wandb login — not baked into the Docker image; pass via `docker run -e` |
| `HF_TOKEN` | container runtime env | Hugging Face downloads — not baked into the Docker image; pass via `docker run -e` |
| `DATA_BUCKET` / `MODEL_BUCKET` | entrypoint.sh | S3 buckets for data / model downloads (placeholder defaults: `your-s3-bucket`) |
| `BEDROCK_EVALUATOR_MODEL` | web_agent `conf/base.yaml` | Bedrock evaluator LLM id for web-agent reward (OmegaConf `${oc.env:...}`) |
| `AWS_ACCOUNT_ID` | web_agent `conf/base.yaml` | Account id interpolated into the Bedrock inference-profile ARN |
| `API_GATEWAY_URL` | web_agent `utils.sh` | SIGv4 proxy endpoint for the browser environment (required, no default) |
| `USER_SIM_MODEL` / `AZURE_USER_SIM_MODEL` / `BEDROCK_USER_SIM_MODEL` | tau2_bench task configs / runtime env | tau2-bench user simulator: local sglang via `rm_router`, Azure OpenAI, or AWS Bedrock — exactly one family must be set; Azure also requires `AZURE_API_KEY` + `AZURE_API_BASE` |
| `MODEL_S3_ROOT` / `DATA_S3_ROOT` | tau2_bench/prepare_model_env.sh | S3 roots for agent/user-sim weights and task data (required, no defaults) |

## Ray job submission

- Do NOT use `--working-dir` in `ray job submit` — the repo is already on shared storage
- Use `python3 ./thirdparty/slime/train_async.py` (relative path, not `/workdir/...`)

# Code Style

- **Fail fast**: use `data["key"]` not `data.get("key")` unless the field is explicitly optional
- Linting: `ruff` (config in pyproject.toml, excludes `thirdparty/`)
- Pre-commit: ruff format + ruff lint + standard hooks (trailing whitespace, etc.)
- Python 3.12, line length 120

# Dependency Management

- **Always use `uv add` to add dependencies** (never `uv pip install` or hand-editing `pyproject.toml` alone). This guarantees the environment is fully reproducible via `uv sync`.
  - Local wheels: drop the file under `wheels/` and run `uv add ./wheels/<name>.whl`
  - PyPI packages: `uv add <package>` (or `uv add <package>==<version>`)
- After any dependency change, commit both `pyproject.toml` and `uv.lock`.
