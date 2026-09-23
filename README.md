# slime-uv

Distributed RL (GRPO) training for LLM **agents** — MCP tool-calling, WebArena browser, and tau2-bench customer-service agents — built on the [slime](https://github.com/THUDM/slime) framework (Megatron training + SGLang inference + Ray orchestration). The whole environment is locked with uv: `pyproject.toml` + `uv.lock` replace slime's pip/docker-patch build, and Megatron-LM and sglang are maintained as fork branches with slime's patches pre-applied, referenced as uv git sources (see `docs/environment-setup.md`).

## Architecture

Training runs fully async: rollout workers continuously generate agent trajectories while the Megatron trainer consumes completed sample groups, decoupled via `shared/fully_async_rollout.py`. Rewards, tool simulation, and evaluation are served by a dedicated SGLang "RM" (reward-model / auxiliary-model) server fleet behind a router, discovered through a shared Ray-actor registry.

```text
Megatron trainer (actor nodes)            thirdparty/slime/train_async.py
   │  weight sync ▼                 ▲ GRPO update ◄── sample groups + rewards
SGLang rollout engines ──► async rollout workers (Ray actors; generate fns)
                               │ agent loop: tool_call_agent · web_agent · tau2_bench
                               ▼ judge / simulator / user-sim requests
SGLang RM fleet: rm_router ── rm_worker × N   (discovered via sglang_registry, ns "sglang")
   serves: LLM judge · tool-call simulator · eval judge · tau2 user simulator
```

- **`tool_call_agent/`** — MCP tool-calling agent. Multi-turn rollouts (max 10 turns); tool responses are replayed by a RAG-based simulator (exact-match cache → fuzzy match via rapidfuzz → LLM-synthesized outputs); binary reward via exact match or LLM-as-judge served by the RM fleet. Entry: `tool_call_agent.generate.generate`.
- **`web_agent/`** — WebArena browser agent. Ray `BrowserWorker` actors drive incus containers running the WebArena sites; reward comes from the environment score. Entry: `web_agent.generate.generate`.
- **`tau2_bench/`** — tau2-bench (retail/telecom customer-service) integration with an LLM user simulator; captures MoE routing replay data per turn. Entry: `generate_with_tau2_gym_moe.generate`.
- **`shared/`** — common infrastructure: fully-async rollout (`AsyncRolloutManager` feeders + `AsyncRolloutWorkerActor` workers, step-lag enforcement via `MAX_STEP_LAG`, dynamic zero-std filtering), `RolloutDataSourceWithExclusion` (replay buffer + permanent sample exclusion), SGLang service registry (detached Ray actor), sample/token helpers with pending+commit loss masking, NCCL-based file broadcast/all-gather, Ray semaphores, timers and wandb rollout logging.
- **`scripts/`** — cluster ops: launch SGLang RM servers (`sglang_job.py`), RM router (`rm_router.py`), registry CLI (`sglang_registry_cli.py`), RM monitor (`monitor_rm.py`), file broadcast, checkpoint conversion, job stoppers.

Training features: fully-async rollout decoupled from training; GRPO with asymmetric clip (`--eps-clip-high`) and low-var KL; TIS (truncated importance sampling); MoE routing replay (`--use-routing-replay`); dynamic sampling filters (`check_reward_nonzero_std`); pass@k evaluation.

## Repository layout

```text
├── shared/                    # rollout, data source, registry, timers, file ops (all agents)
├── tool_call_agent/           # MCP tool-calling agent: generate loop, tool sim, eval
├── web_agent/                 # WebArena browser agent: browser env, generate loop
├── tau2_bench/                # tau2-bench agent: env workers, user sim, MoE capture
├── scripts/                   # cluster ops (see SCRIPTS_REFERENCE.md)
├── thirdparty/                # git subtrees: slime, rl_web_agent, bfcl, tau2-bench
├── patches/                   # slime_local.patch — local diff on vendored slime
├── docs/                      # environment-setup.md, slime-local-modifications.md
├── wheels/                    # vendored wheels referenced as uv path sources
├── Dockerfile, docker-bake.hcl  # container image build
├── launch_ray.sh              # Ray cluster bootstrap (AWS Batch or K8s)
├── entrypoint.sh              # annotated end-to-end cluster workflow example
└── pyproject.toml, uv.lock    # reproducible uv environment (Python 3.12)
```

`thirdparty/` repos are vendored as git subtrees. slime is installed editable (path dependency), so local slime patches live in `patches/slime_local.patch` and are documented in `docs/slime-local-modifications.md`; treat the other subtrees as upstream.

## Installation

Python 3.12. Heavy CUDA dependencies (torch cu129, flash-attn, apex, transformer_engine, sglang kernels) only resolve/build on Linux with NVIDIA GPUs.

### Docker (recommended for clusters)

The `Dockerfile` builds on a slime/sglang CUDA base image that you must build yourself — the `REPO`/`BASE_TAG` ARGs point at your registry (see `docs/environment-setup.md` for how to build that base):

```bash
docker buildx build \
  --build-arg REPO=<your-registry>/slime-base --build-arg BASE_TAG=base \
  -t slime-uv .
# or via bake (target slime_rl; note: bake config pushes the image):
REGISTRY=<your-registry>/slime-base docker buildx bake
```

Secrets are not baked into the image; provide `WANDB_API_KEY` and `HF_TOKEN` at container runtime (e.g. `docker run -e WANDB_API_KEY=... -e HF_TOKEN=...`).

### Local development

`uv sync` creates `.venv` from `uv.lock`. For cluster code you'll realistically do this on a Linux GPU box or inside the container; `entrypoint.sh` / `launch_ray.sh` assume the venv at `.venv/`.

## Cluster setup quickstart

1. **Start the Ray cluster** on every node. `launch_ray.sh` detects AWS Batch (`AWS_BATCH_JOB_*` env vars) or Kubernetes (`K8S_RANK`, `K8S_WORLD_SIZE`, `K8S_MASTER_ADDR`, `K8S_MASTER_PORT`), starts `ray start --head` on the head and joins workers:

   ```bash
   bash launch_ray.sh
   ```

2. **Download / convert the model** (once; then broadcast to nodes as needed):

   ```bash
   bash scripts/download_convert_model.sh \
     --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
     --config qwen3-30B-A3B
   ```

3. **Start the RM router** on the head node (auto-submits itself as a Ray job, registers as `rm_router`):

   ```bash
   python scripts/rm_router.py
   ```

4. **Launch the SGLang RM fleet** (one SGLang server per actor, each registering as `rm_worker`):

   ```bash
   ray job submit --address=auto -- \
     python scripts/sglang_job.py --num-gpus 1 --num-nodes 16 \
       --model /data/base_models/Qwen/Qwen3-30B-A3B-Thinking-2507 \
       --context-length 131072 --reasoning-parser deepseek-r1 --tool-call-parser qwen
   ```

5. Watch the fleet with `python scripts/monitor_rm.py`.

`entrypoint.sh` is an annotated, end-to-end example of this workflow (including `s5cmd` data downloads and training-data conversion). Full per-script reference: `SCRIPTS_REFERENCE.md`.

## Training

All launch scripts only submit a Ray job for `thirdparty/slime/train_async.py`; they do not start/stop the cluster. Cluster path conventions: models in `/data/base_models/`, checkpoints in `/data/checkpoints/`, MCP data in `/data/mcp-data/`.

### tool_call_agent

```bash
./tool_call_agent/run_grpo_async.sh <model_config> <run_name>
# model_config: file under tool_call_agent/models/ (without .sh), e.g. qwen3-30B-A3B, qwen3-4B
# run_name:     wandb group / checkpoint subdirectory name
```

Training data defaults to `/data/mcp-data/data_batches/3/training_data.jsonl`; regenerate it from MCP tasks with `python tool_call_agent/convert_mcp_to_training.py` (see `tool_call_agent/DATA_FORMAT.md`). Requires `JUDGE_MODEL`, `TRAJECTORY_PATH`, `SERVERS_PATH` and a running RM fleet for the judge/simulator.

### web_agent

```bash
./web_agent/run_grpo_async.sh <model_config> <run_name>
# model_config: file under web_agent/models/ (without .sh), e.g. qwen3-4B, qwen3-30B-A3B
```

Training data ships in-repo (`web_agent/data/train_shopping_shopping_admin_gitlab.jsonl`); convert new WebArena tasks with `web_agent/convert_webarena_to_training.py`. Requires the incus/proxy browser environment (`INCUS_SERVER_URL`, `PROXY_SERVER`, ...).

### tau2_bench

```bash
./tau2_bench/run_grpo_async.sh <model_config> <task_config> <run_name>
# model_config: tau2_bench/models/, e.g. qwen3-30B-A3B
# task_config:  tau2_bench/tasks/,  e.g. retail | telecom | retail_azure
```

Task configs set the domain, reward shaping, and user simulator; model configs set architecture, checkpoint paths, and cluster sizing. Requires a user simulator (see env table) and the RM route registered as `rm_router`. Prepare model/data staging with `tau2_bench/prepare_model_env.sh` (`MODEL_S3_ROOT`, `DATA_S3_ROOT`); details in `tau2_bench/README.md`.

## Evaluation

### checkpoints → HF format

```bash
bash scripts/convert_all_checkpoints.sh \
  --input-dir /data/checkpoints/mcp/<run> \
  --output-dir /data/checkpoints/mcp/<run>-hf \
  --origin-hf-dir /data/base_models/<HF_MODEL_NAME>
```

### pass@k evaluation (tool_call_agent)

Spins up an SGLang eval server per checkpoint and scores with exact match or the LLM judge:

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

### BFCL evaluation

```bash
python tool_call_agent/bfcl_evaluate.py \
  --checkpoint /data/checkpoints/mcp/<run>-hf/ \
  --bfcl-model Qwen/Qwen3-30B-A3B-Instruct-2507-FC \
  --test-category all \
  --output /data/checkpoints/mcp/bfcl_eval_results
```

## Credentials / runtime environment

These are intentionally **not** committed to the repository or baked into the image; provide them via the runtime env (container flags or the `run_grpo_async.sh` runtime-env JSON).

| Variable | Used by | Purpose |
|---|---|---|
| `WANDB_API_KEY` | all runs | wandb auth (login via env; removed from Dockerfile) |
| `HF_TOKEN` | image build / runs | Hugging Face downloads |
| `AWS_PROFILE` | optional | standard AWS credential chain |
| `JUDGE_MODEL` | tool_call_agent | judge model path for reward + eval |
| `TRAJECTORY_PATH` | tool_call_agent | trajectory data for the tool simulator |
| `SERVERS_PATH` | tool_call_agent | path to `mcp_servers_joined.json` |
| `TOOL_CALL_PARSER` | all agents | sglang tool call parser (default `qwen25`) |
| `BEDROCK_EVALUATOR_MODEL`, `AWS_ACCOUNT_ID` | web_agent | evaluator LLM + Bedrock ARN construction (OmegaConf env interpolation) |
| `INCUS_SERVER_URL` | web_agent | container orchestrator for browser envs |
| `PROXY_SERVER`, `PROXY_ENABLED`, `BROWSER_HEADLESS` | web_agent | browser env proxy / display settings |
| `USER_SIM_MODEL` | tau2_bench | user simulator served by the RM fleet (or one of the alternatives below; **at least one is required**) |
| `AZURE_USER_SIM_MODEL` + `AZURE_API_BASE`, `AZURE_API_KEY` | tau2_bench | Azure OpenAI user simulator |
| `BEDROCK_USER_SIM_MODEL` | tau2_bench | Bedrock user simulator |
| `DATA_BUCKET`, `MODEL_BUCKET` | entrypoint.sh | example S3 buckets for data/model staging |
| `MODEL_S3_ROOT`, `DATA_S3_ROOT` | tau2_bench | staging roots for `prepare_model_env.sh` |

## Development

- **Lint/format**: `ruff` (line length 120) + `ruff-format` via pre-commit (`pre-commit install`; `thirdparty/` and `patches/` are excluded).
- **Style**: fail fast — use `data["key"]`, not `data.get("key")`, unless the field is explicitly optional.
- **Dependencies**: always `uv add <package>` (never hand-edit `pyproject.toml` alone or `uv pip install`); local wheels go in `wheels/` (`uv add ./wheels/<name>.whl`). Commit both `pyproject.toml` and `uv.lock`.
- **slime updates**: pull upstream with `git subtree pull`; keep local diffs in `patches/slime_local.patch` and documented in `docs/slime-local-modifications.md`.
- **`CODEBASE_DOCS.md`**: file-by-file reference for `shared/`, `scripts/`, and `web_agent/`.

## License

MIT — see `LICENSE` (Copyright (c) 2026 Yuxuan Lu).

---

Further reading: `CODEBASE_DOCS.md` (module reference), `SCRIPTS_REFERENCE.md` (CLI reference), `docs/` (environment and slime-patch details).
