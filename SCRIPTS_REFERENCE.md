# Scripts Reference

## Core Infrastructure

### `launch_ray.sh`
Bootstraps a Ray cluster on AWS Batch multi-node jobs or Kubernetes. The platform is auto-detected; anything else exits with an error.

- **AWS Batch**: reads `AWS_BATCH_JOB_NODE_INDEX`, `AWS_BATCH_JOB_MAIN_NODE_INDEX`, `AWS_BATCH_JOB_NUM_NODES`; head IP from `AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS`
- **Kubernetes**: reads `K8S_RANK`, `K8S_WORLD_SIZE`, `K8S_MASTER_ADDR`, `K8S_MASTER_PORT`; rank 0 is the head
- Activates the repo's `.venv` (dependencies are installed into the image at build time; no `uv sync` at runtime)
- On the **head node**: runs `ray start --head`, then waits (Python loop) until all expected nodes are alive, then exits.
- On **worker nodes**: runs `ray start --address=<HEAD_IP>:<RAY_PORT>`, then blocks with `tail -f /dev/null` (unless `INTERACTIVE_DEBUG=1` or sshd is running).
- Env vars: `RAY_PORT` (default 6379), `DASHBOARD_PORT` (default 8265), `RAY_TEMP_DIR` (default `/tmp/instance_storage/ray_tmp`, AWS Batch only), `NUM_GPUS` (auto-detected), `NUM_CPUS` (auto-detected).

### `entrypoint.sh`
Example orchestration script showing a full workflow. S3 locations come from env vars: `DATA_BUCKET` and `MODEL_BUCKET` (both default to the placeholder `your-s3-bucket` — set them to real buckets).

1. `bash launch_ray.sh` - start Ray cluster
2. `s5cmd cp` - download data batches, MCP servers config, and base model from S3
3. `bash scripts/download_convert_model.sh --model ... --config ...` - download & convert HF model to torch_dist
4. `python tool_call_agent/convert_mcp_to_training.py` - convert MCP tasks to training data
5. `python scripts/rm_router.py` - start the reward model router
6. `ray job submit ... python scripts/sglang_job.py ...` - launch SGLang reward model server(s)

### `utils.sh`
Two inline Python snippets for quick Ray registry inspection:

- **Snippet 1**: Dump the sglang_registry contents (read-only)
- **Snippet 2**: Dump, clear, then dump the sglang_registry

---

## Scripts (`scripts/`)

### `sglang_job.py`
**Launch SGLang reward model server(s) as Ray actors.**

Submitted as a Ray job. Starts N independent `RewardSGLangActor` instances, each running an SGLang server on a different node. After starting, each actor registers itself in the sglang_registry as `rm_worker` and adds itself to the `rm_router` if one exists.

```
ray job submit --address=auto --working-dir . -- \
  python scripts/sglang_job.py \
    --num-gpus <float>           # GPUs per actor (default: 1)
    --num-nodes <int>            # Number of independent instances (default: 1)
    --registry-name <str>        # Ray actor name (default: sglang_registry)
    --node-ip <str>              # Pin all actors to this node IP
    --model <path>               # Model path (SGLang arg)
    --tp <int>                   # Tensor parallelism (SGLang arg)
    --context-length <int>       # Context length (SGLang arg)
    --reasoning-parser <str>     # e.g. deepseek-r1 (SGLang arg)
    --tool-call-parser <str>     # e.g. qwen (SGLang arg)
    + all other SGLang ServerArgs
```

### `rm_router.py`
**Launch an SGLang router for reward model servers.**

When run directly (`python scripts/rm_router.py`), it auto-submits itself as a Ray job with `--no-wait`. Inside the Ray job, it:
1. Starts `sglang_router.launch_router` as a subprocess (policy: random)
2. Registers itself in sglang_registry as `rm_router`
3. Adds any existing `rm_worker` entries to the router
4. Blocks until the subprocess exits

No CLI args (auto-submits as Ray job).

### `sglang_registry_cli.py`
**CLI to operate the shared SGLang registry (Ray detached actor).**

```
python scripts/sglang_registry_cli.py [--registry-name NAME] <command>

Commands:
  dump                          # Print all entries as JSON
  add <key> <url>               # Register a URL under a key
  remove <key> [--url <url>]    # Remove all URLs for key, or a specific URL
  clear                         # Clear all entries
```

### `clear_sglang_registry.py`
**Clear everything in the SGLang registry.** No arguments. Connects to Ray, clears the registry, prints before/after state.

### `monitor_rm.py`
**Live monitor for reward model servers.**

Queries the `rm_router`'s `/workers` and `/get_loads` endpoints. Falls back to direct worker queries if no router found. Displays running requests, queued requests, tokens, throughput per worker in a continuously refreshing table.

```
python scripts/monitor_rm.py [--interval/-i SECONDS] [--debug/-d]
```

- `--interval` / `-i`: Refresh interval in seconds (default: 2.0)
- `--debug` / `-d`: Print raw API responses on first run

### `all_gather_files.py`
**All-gather files across all GPU nodes via NCCL.**

Each node contributes files from its local `src_dir`. At the end, every node has all files in `dst_dir`. Uses chunked, double-buffered GPU-to-GPU transfers via `torch.distributed`.

```
ray job submit --address=auto --working-dir . -- \
  python -m scripts.all_gather_files <src_dir> \
    [--dst <dir>]                # Destination (defaults to src_dir)
    [--chunk-size <MB>]          # Chunk size in MB (default: 1024)
    [--num-buffers <int>]        # Buffer pool size (default: 10)
    [--bench-mode sender|receiver]
```

### `broadcast_files.py`
**Broadcast files/directories from driver node to all GPU nodes via NCCL.**

Same transfer mechanism as `all_gather_files.py`, but one-to-all (driver broadcasts).

```
ray job submit --address=auto --working-dir . -- \
  python -m scripts.broadcast_files <src> \
    [--dst <dir>]                # Destination (defaults to src path)
    [--chunk-size <MB>]          # Chunk size in MB (default: 1024)
    [--num-buffers <int>]        # Buffer pool size (default: 10)
    [--bench-mode sender|receiver]
```

### `run_on_each_node.py`
**Run a command once on each Ray node and wait for completion.**

Creates a Ray actor per alive node (pinned via `NodeAffinitySchedulingStrategy`), runs the given command on each, and waits for all to finish.

```
ray job submit --address=auto --working-dir . -- \
  python scripts/run_on_each_node.py [--no-gpu] [--env KEY=VALUE ...] <command...>
```

- `--no-gpu`: Request only 1 CPU per node (default: 8 GPUs)
- `--env KEY=VALUE`: Extra env vars passed to the command (repeatable)

### `stop_all_ray_jobs.py`
**Stop all running Ray jobs.** No arguments. Lists and stops every job with status RUNNING.

### `stop_train_job.py`
**Stop the running `train_async.py` Ray job.** No arguments. Finds and stops only jobs whose entrypoint contains `train_async.py`.

### `download_convert_model.sh`
**Download a HuggingFace model and convert it to torch_dist format for slime training.**

```
bash scripts/download_convert_model.sh \
  -m|--model <HF_MODEL>         # e.g. Qwen/Qwen3-30B-A3B-Thinking-2507
  -c|--config <CONFIG_NAME>     # Model config from thirdparty/slime/scripts/models/ (e.g. qwen3-30B-A3B)
  [-o|--output <dir>]           # Output dir (default: /tmp/instance_storage/<config>_torch_dist)
  [-d|--download <dir>]         # Download dir (default: /tmp/instance_storage/<config>)
  [-h|--help]                   # Show help + available configs
```

Steps: `hf download` -> source model config -> `python tools/convert_hf_to_torch_dist.py`

Env vars: `HF_MODEL_NAME`, `MODEL_CONFIG`, `SAVE_PATH`, `DOWNLOAD_PATH` (alternatives to CLI args).

### `convert_all_checkpoints.sh`
**Convert all `iter_*` torch distributed checkpoints to HuggingFace format.**

```
bash scripts/convert_all_checkpoints.sh \
  --input-dir <dir>              # Training run dir containing iter_* subdirectories
  --output-dir <dir>             # Output dir for HF checkpoints
  --origin-hf-dir <dir>         # Path to original HF model (optional if --model-name in extra-args)
  [--extra-args "..."]           # Extra args for convert_torch_dist_to_hf.py
```

Finds all `iter_*` directories, converts each via `thirdparty/slime/tools/convert_torch_dist_to_hf.py`.

---

## Training Launch Scripts

### `tool_call_agent/run_grpo_async.sh`
**Run fully async GRPO training for tool-call agent.** Submits a Ray job running `thirdparty/slime/train_async.py`.

- Usage: `./run_grpo_async.sh <model_config> <run_name>` — `model_config` is a file under `tool_call_agent/models/` (without `.sh`), `run_name` is the wandb group / checkpoint subdirectory name
- Cluster size comes from the model config (qwen3-30B-A3B: 2 actor nodes x 8 GPUs, 32 rollout GPUs)
- Async-specific: `shared.fully_async_rollout.generate_rollout_fully_async`, `shared.data_source.RolloutDataSourceWithExclusion`
- 300 rollouts, rollout batch 16, 8 samples/prompt, over-sampling batch 128, temp 1.0
- Routing replay + TIS (train-infer-switch) enabled by the qwen3-30B-A3B model config
- Runtime env vars: `JUDGE_MODEL`, `TRAJECTORY_PATH`, `SERVERS_PATH`, `EXCLUDE_ON_DROP_REASONS=zero_std_1.0`

### `web_agent/run_grpo_async.sh`
**Run fully async GRPO training for web agent with browser environment.** Submits a Ray job running `thirdparty/slime/train_async.py`.

- Usage: `./run_grpo_async.sh <model_config> <run_name>` — `model_config` is a file under `web_agent/models/` (without `.sh`: `qwen3-4B`, `qwen3-30B-A3B`, `llama3.1-8B`); cluster size comes from the model config
- Training data: `web_agent/data/train_shopping_shopping_admin_gitlab.jsonl`
- 200 rollouts, rollout batch 16, 12 samples/prompt, over-sampling batch 64
- Custom generate: `web_agent.generate.generate`
- Browser env vars: `INCUS_SERVER_URL` (default `http://127.0.0.1:8001`), `PROXY_SERVER` (default `http://localhost:8080`), `PROXY_ENABLED`, `BROWSER_HEADLESS`, `MAX_CONCURRENT_CONTAINER_LAUNCHES` (80), `MAX_CONCURRENT_CONTAINERS_RUNNING` (1024), `AWS_PROFILE` (default `default`)

### `web_agent/utils.sh`
One-liner to start the SIGv4 proxy client for web agent browser environment. `API_GATEWAY_URL` is required (no default):
```
AWS_REGION="${AWS_REGION:-us-east-1}" API_GATEWAY_URL="<your proxy endpoint>" ./target/release/proxy_client_sigv4
```

### `tau2_bench/run_grpo_async.sh`
**Run fully async GRPO training for tau2-bench (retail/telecom).** Submits a Ray job running `thirdparty/slime/train_async.py`.

- Usage: `./run_grpo_async.sh <model_config> <task_config> <run_name>`
- Sources `tau2_bench/tasks/<task_config>.sh` first (domain, reward shaping, rollout/eval/GRPO args, user simulator), then `tau2_bench/models/<model_config>.sh` (architecture, ckpt paths, perf/sglang/MoE args, cluster size)
- Task configs: `retail`, `retail_azure`, `telecom`, `telecom_smoke`
- User simulator: one of `USER_SIM_MODEL` (local sglang served via `rm_router`), `AZURE_USER_SIM_MODEL` (Azure OpenAI — also requires `AZURE_API_KEY` + `AZURE_API_BASE`), or `BEDROCK_USER_SIM_MODEL` (AWS Bedrock); all `TAU2_*` / user-sim vars are forwarded to rollout workers via the Ray runtime env
- Prereqs: Ray cluster up, `rm_router` running, and (unless using Azure/Bedrock) a user-sim sglang server registered under `rm_router` — see `tau2_bench/README.md`

---

## Typical Workflow

1. **Start Ray cluster**: `bash launch_ray.sh` (run on all nodes via AWS Batch or Kubernetes)
2. **Download data**: `s5cmd cp` from S3
3. **Download & convert model**: `bash scripts/download_convert_model.sh -m <model> -c <config>` (use `run_on_each_node.py` to run on all nodes)
4. **Prepare training data**: `python tool_call_agent/convert_mcp_to_training.py`
5. **Start RM router**: `python scripts/rm_router.py`
6. **Start RM servers**: `ray job submit ... python scripts/sglang_job.py ...`
7. **Monitor RM**: `python scripts/monitor_rm.py`
8. **Launch training**: `bash tool_call_agent/run_grpo_async.sh`, `bash web_agent/run_grpo_async.sh`, or `bash tau2_bench/run_grpo_async.sh` (all fully async)
9. **Convert checkpoints**: `bash scripts/convert_all_checkpoints.sh ...`
10. **Stop training**: `python scripts/stop_train_job.py` or `python scripts/stop_all_ray_jobs.py`
