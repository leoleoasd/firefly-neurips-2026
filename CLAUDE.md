# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Distributed RL (GRPO) training for LLM **agents**, built on the **slime** framework (Megatron training + SGLang inference, orchestrated with **Ray**). Three agent task families share common infrastructure:

- `tool_call_agent/` — MCP tool-calling agent (multi-turn, RAG-simulated tool responses, LLM-as-judge reward)
- `web_agent/` — browser-based web agent (`thirdparty/rl_web_agent` environment)
- `tau2_bench/` — tau2-bench (retail/telecom) agent integration
- `shared/` — infrastructure used by all three (rollout, registry, data source, parsers, timers)
- `scripts/` — cluster ops CLIs (SGLang launch, RM router, registry, checkpoint conversion, file broadcast)

`thirdparty/` (`slime/`, `rl_web_agent/`, `bfcl/`, `tau2-bench/`) are **git subtrees — do not edit directly**.

## Big-picture architecture

The training loop lives in `thirdparty/slime/train_async.py`. Each iteration: **rollout** (run the LLM against an environment to produce trajectories) → **reward** (RM server or environment score) → **train** (Megatron weight update) → **weight sync** back to the SGLang inference engines.

Things that require reading several files to understand:

- **Rollout entry points are pluggable.** A run's `CUSTOM_ARGS` points slime at a `generate_rollout` function. The fully-async path (`shared/fully_async_rollout.py`) decouples rollout from training: a background daemon thread (`AsyncRolloutWorker`) continuously runs `generate_and_rm_group` and pushes completed groups onto a queue; `generate_rollout_async` consumes until `rollout_batch_size` groups pass the dynamic filter. Per-agent `generate.py` implements the actual multi-turn loop (generate → parse tool calls → execute/simulate tools → append → repeat).

- **Token bookkeeping uses pending/commit.** `shared/sample_helpers.py` accumulates tokens in pending buffers and commits them per assistant turn via `add_assistant_message()`, which is what produces correct loss masks and logprobs for RL across multi-turn conversations. Touch this carefully — off-by-one in masking silently corrupts training.

- **Service discovery is Ray-actor based.** `shared/sglang_registry.py` is a detached Ray actor mapping keys (`rm_worker`, `rm_router`, …) to URLs. RM servers (`scripts/sglang_job.py`) self-register; `scripts/rm_router.py` load-balances across them. Don't hardcode server URLs — go through the registry.

- **Tool responses are simulated, not live.** `tool_call_agent/tool_call_simulator.py` replays historical tool calls (exact-match cache → `rapidfuzz` fuzzy match → Anthropic-Claude synthesis for novel calls), distributed across nodes via `tool_workers.py`. This avoids live API calls during training.

- **Tool-call parsing is consolidated** in `shared/tool_call_parser.py` (used by all three agents). It runs SGLang's `FunctionCallParser` locally (no server round-trip). The parser name defaults to `$TOOL_CALL_PARSER` (`qwen25`).

## Common commands

```bash
# GRPO training (async). model_config = a file under <agent>/models/ without .sh
./tool_call_agent/run_grpo_async.sh <model_config> <run_name>
./web_agent/run_grpo_async.sh <model_config> <run_name>

# Convert HF checkpoint -> Megatron torch_dist (source the model config first)
source thirdparty/slime/scripts/models/<MODEL_CONFIG>.sh && \
  uv run python thirdparty/slime/tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" --hf-checkpoint <HF_PATH>/ --save <OUT_torch_dist>/

# Convert training checkpoints -> HF (use this batch script, NOT the per-iter slime tool)
bash scripts/convert_all_checkpoints.sh \
  --input-dir /data/checkpoints/mcp/<run> \
  --output-dir /data/checkpoints/mcp/<run>-hf \
  --origin-hf-dir /data/base_models/<HF_MODEL_NAME>

# Evaluate tool_call_agent checkpoints
python tool_call_agent/evaluate.py --checkpoint /data/checkpoints/mcp/<run>-hf/ \
  --data-batches /data/mcp-data/data_batches --batch 3 --task-file test_tasks.json \
  --servers /data/mcp-data/mcp_servers_joined.json --tool-mode dag --passes 16 --only-missing

# Lint / format (ruff, excludes thirdparty/)
uv run ruff check . && uv run ruff format .
```

### Run-script config split

`run_grpo_async.sh` sources `models/<name>.sh`, which sets model/cluster vars: `MODEL_ARGS`, `CKPT_ARGS`, `PERF_ARGS`, `SGLANG_ARGS`, `MISC_ARGS`, `ACTOR_NUM_NODES`, `ROLLOUT_NUM_GPUS`. The main script sets task/data vars: `ROLLOUT_ARGS`, `GRPO_ARGS`, `OPTIMIZER_ARGS`, `WANDB_ARGS`, `CUSTOM_ARGS`.

### Key runtime env vars

| Variable | Purpose |
|---|---|
| `TOOL_CALL_PARSER` | SGLang tool-call parser name (`qwen25`, `llama`, …) |
| `JUDGE_MODEL` | Path to judge model for reward scoring |
| `TRAJECTORY_PATH` | Trajectory data dir for tool simulation |
| `SERVERS_PATH` | Path to `mcp_servers_joined.json` |
| `WANDB_API_KEY` / `HF_TOKEN` | Container-runtime credentials — removed from the Docker image; pass via `docker run -e` |
| `DATA_BUCKET` / `MODEL_BUCKET` | S3 buckets used by `entrypoint.sh` (placeholder defaults: `your-s3-bucket`) |
| `BEDROCK_EVALUATOR_MODEL` | Bedrock evaluator LLM for web_agent rewards (`web_agent/conf/base.yaml`) |
| `AWS_ACCOUNT_ID` | Account id interpolated into the web_agent Bedrock inference-profile ARN |
| `USER_SIM_MODEL` | tau2_bench user simulator: local sglang model served via `rm_router` |
| `AZURE_USER_SIM_MODEL` | tau2_bench user simulator via Azure OpenAI (also requires `AZURE_API_KEY` + `AZURE_API_BASE`) |
| `BEDROCK_USER_SIM_MODEL` | tau2_bench user simulator via AWS Bedrock |
| `MODEL_S3_ROOT` / `DATA_S3_ROOT` | S3 roots for tau2_bench/prepare_model_env.sh (required, no defaults) |

### Key disk paths

`/data/base_models/` (HF weights + torch_dist), `/data/checkpoints/` (`mcp/` = tool_call_agent, `rl_web_agent/` = web_agent), `/data/mcp-data/` (training data).

## Running a tool_call_agent GRPO experiment end-to-end (30B-A3B)

`entrypoint.sh` is the canonical reference (cluster bootstrap + data/model prep + reward infra + training). On an already-bootstrapped cluster with `/data` populated, a run is **reward infra first, then training** (reward calls resolve the judge through the Ray registry, so the RM must be up before rollout reaches scoring).

**0. Prereqs (usually already on `/data`):** training data `/data/mcp-data/data_batches/3/training_data.jsonl` (built by `convert_mcp_to_training.py`), `mcp_servers_joined.json`, HF base `/data/base_models/Qwen/Qwen3-30B-A3B-Thinking-2507/`, and torch_dist ref-load `/data/base_models/megatron/qwen3-30B-A3B_torch_dist/`.

**1. RM router** (key `rm_router`, 0 GPU): `python scripts/rm_router.py` — self-submits as a detached `ray job ... --no-wait` and returns immediately. Wait until it appears in the registry before starting workers.

**2. Judge / reward servers** (key `rm_worker`, auto-added to the router):
```bash
ray job submit --address=auto --no-wait -- python scripts/sglang_job.py \
  --num-gpus 1 --num-nodes 16 \
  --model /data/base_models/Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --context-length 131072 --reasoning-parser deepseek-r1 --tool-call-parser qwen
```
In `sglang_job.py`, `--num-nodes N` = N independent replicas (each registers as `rm_worker`), `--num-gpus G` = GPUs/replica (= TP if >1). 30B-A3B fits on 1 H200 (143 GB). **Loading 16×30B from shared storage is slow (~15 min)** due to bandwidth contention — `0/16 rm_worker` for ~10 min is normal, not a hang; check `ray job logs <id>` for `Multi-thread loading shards: N/16`.

**Judge model name is ignored.** `run_grpo_async.sh` hardcodes `JUDGE_MODEL` (e.g. `/data/base_models/Qwen/Qwen3-30B-A3B`, which need not exist); the `model` field in the judge request is not matched against `served-model-name` — the router just forwards to whatever model the `rm_worker` serves. So **don't symlink / fix the path** — just serve the model you want as judge (here: Thinking-2507 with `reasoning-parser deepseek-r1`).

**3. GPU budget** (these numbers are specific to **this 30B-A3B config on an 8×8 = 64 H200 cluster**; other models / tasks / cluster sizes change all of them — recompute from the config's `ACTOR_NUM_NODES`, `ACTOR_NUM_GPUS_PER_NODE`, `ROLLOUT_NUM_GPUS`, and the judge's per-replica TP): training job = actor (`ACTOR_NUM_NODES`×8 = 16) + rollout (`ROLLOUT_NUM_GPUS` = 32) = **48**, leaving **16** for the judge (16×1-GPU). rm_router/registry are CPU-only. A bigger judge (e.g. a 235B RM) needs multi-GPU TP per replica and fewer replicas, so re-split accordingly.

**4. Launch training:** `./tool_call_agent/run_grpo_async.sh <model_config> <run_name>`. It streams logs and blocks — run detached (`nohup ... &`) for real runs. Checkpoints → `/data/checkpoints/mcp/qwen3-30B-A3B-<run_name>/`, wandb group = `<run_name>`.

**Log-prob / routing-replay modes** are set by the model config (`models/<name>.sh`), which is **sourced before** the runtime-env JSON is built, so exporting `SLIME_ROLLOUT_FULL_*` in your shell is overridden — pick or create a config variant instead. The two toggles are independent (`shared/sample_helpers.py`):
- `qwen3-30B-A3B.sh` — both incremental.
- `qwen3-30B-A3B-full-logprobs.sh` — `SLIME_ROLLOUT_FULL_LOGPROBS=1` (full log-probs).
- `qwen3-30B-A3B-full-routing.sh` — `SLIME_ROLLOUT_FULL_ROUTING=1` (incremental log-probs + full routing-replay experts).

**5. Monitor:** `ray job status <id>` / `ray job logs <id>`. Healthy markers: `model.py:811 - step N: {train/loss,...}` and `rollout.py - perf N: {rollout/...}`. Init is slow (~10–15 min: megatron init + 32 rollout engines loading weights from shared storage) before the first rollout. Benign noise: `Failed to parse JSON part: ...` — the policy occasionally emits malformed tool-call JSON, handled as a format error, not a crash. `ray job status` on a *terminal* job prints `Job 'X' succeeded` (not the `Status for job ... RUNNING` line) — grep for both.

### Evaluating a finished run

1. **Convert all iters → HF** (`evaluate.py` only reads HF format): `bash scripts/convert_all_checkpoints.sh --input-dir /data/checkpoints/mcp/<run> --output-dir /data/checkpoints/mcp/<run>-hf --origin-hf-dir /data/base_models/Qwen/Qwen3-30B-A3B-Thinking-2507`. Sequential, ~5 min/ckpt, skips already-converted.
2. **Reuse the training judge infra** — the `rm_router` + `rm_worker`s are still registered, so eval needs no new RM. `evaluate.py` launches its own policy SGLang server per checkpoint (`--num-gpus`/`--tp`) on free GPUs and routes judge + tool-sim through the registry.
3. **Match the prior eval hyperparams** (recorded in each `eval_results/eval_*.json` `metadata`; cross-check `--task-file` against the doc since it is *not* recorded). The canonical 30B-A3B config: `--data-batches /data/mcp-data/data_batches --batch 3 --task-file test_tasks.json --tool-mode dag --tool-call-parser qwen --max-rounds 10 --num-tasks 100 --passes 16 --parallel 400 --seed 42 --num-gpus 1`, `JUDGE_MODEL=/data/base_models/Qwen/Qwen3-30B-A3B-Thinking-2507`. Point `--checkpoint` at the `-hf/` parent (globs all `iter_*`), `--output /data/checkpoints/mcp/eval_results --only-missing`. Results: `pass_at_k` per checkpoint; aggregate the latest file per iter.

**Plotting:** `notebooks/eval_plots.ipynb` builds pass@k-vs-step curves from `eval_results/`. Add a run by appending `model_prefix -> (display_name, base_ckpt_name, color)` to `TRAINED_MODELS` (base eval `Qwen__Qwen3-30B-A3B-Thinking-2507` gives the step-0 point).

**Result (`fullroute-insclp`, incremental log-probs + full routing-replay):** final iter_299 pass@1 **44.8%** / pass@16 **60.0%**, beating both the full-log-probs (`full-test`: 38.7% / 55%) and the coupled-incremental (`insc-test`) baselines at matched iters — i.e. decoupling routing to full-update was a net win.

## Conventions (enforced)

- **Fail fast — no defensive coding.** Use `data["key"]` not `data.get("key")`, `obj.attr` not `getattr`. Missing required fields should raise immediately. Only use `.get()`/`getattr` for fields the user has explicitly said are optional.
- **Dependencies: always `uv add`** (never `uv pip install` or hand-editing `pyproject.toml`). Local wheels → drop in `wheels/` then `uv add ./wheels/<name>.whl`. Commit both `pyproject.toml` and `uv.lock`.
- Python 3.12, line length 120. Pre-commit runs ruff format + lint.
- All comments, docs, and code must be in English.

## Ray job submission gotchas

- Do **not** pass `--working-dir` to `ray job submit` — the repo is on shared storage already.
- Invoke the trainer by relative path: `python3 ./thirdparty/slime/train_async.py`.
