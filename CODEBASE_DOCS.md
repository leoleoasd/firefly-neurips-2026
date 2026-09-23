# Codebase Documentation: shared/, scripts/, tool_call_agent/, web_agent/, tau2_bench/

This document describes every file and every function/class in the `shared/`, `scripts/`, `tool_call_agent/`, `web_agent/`, and `tau2_bench/` directories. This codebase is built on top of the **slime** RL training framework and uses **Ray** for distributed orchestration and **SGLang** for LLM inference.

---

## Architecture Overview

The system implements distributed RLHF/GRPO training for LLM agents (tool-calling agents, web agents, tau2-bench customer-service agents). The high-level flow is:

1. **Training loop** (`thirdparty/slime/train_async.py`) orchestrates rollout and training steps.
2. **Rollout** (`shared/fully_async_rollout.py`) feeds prompt groups to a fleet of `AsyncRolloutWorkerActor` Ray actors, each running many async tasks that call slime's `generate_and_rm_group` → the agent-specific `generate()` function.
3. **Reward** is computed per-agent (LLM judge for tool_call_agent, environment scores for web_agent / tau2_bench, plus format/length penalties).
4. **Training** updates model weights, then syncs them to SGLang inference engines (this is why generate loops retry on sglang `abort` finish reasons — the server briefly aborts requests while reloading weights).

- `shared/` contains reusable infrastructure (async rollout orchestration, data source, service registry, sample building, timers, counters, file transfer).
- `scripts/` contains CLI utilities for cluster operations.
- `tool_call_agent/` contains the MCP tool-calling agent: rollout loop, RAG-based tool simulation, evaluation.
- `web_agent/` contains the web browser agent rollout logic.
- `tau2_bench/` contains the tau2-bench gym-integration agent (MoE token-in/token-out variant).

---

## shared/

### shared/__init__.py
Empty. Makes `shared` a Python package.

---

### shared/fully_async_rollout.py
Fully asynchronous rollout: CPU-heavy rollout work (tokenization, JSON serialization, tool execution) runs in a fleet of dedicated Ray actor processes, decoupled from training. Serves `tool_call_agent`, `web_agent`, and `tau2_bench`.

**Architecture** (see module docstring, `shared/fully_async_rollout.py:1`):
- `AsyncRolloutManager` lives in the RolloutManager process; a **feeder thread** pumps sample groups from `data_buffer` into a bounded input **Ray Queue**.
- N `AsyncRolloutWorkerActor` Ray actors (separate processes) each run `concurrency` async tasks that pull groups from the input queue, call slime's `generate_and_rm_group`, and accumulate results locally.
- `generate_rollout_async` (the collector, running in the RolloutManager process) drains completed groups from all workers until a training batch is assembled.

Results deliberately do **not** flow through a Ray Queue: `ray.util.queue.Queue`'s `qsize()`/`get_nowait()` desync under this workload (see `scripts/repro_ray_queue_bug.py`), so workers keep a lock-protected local list that the collector polls via `drain_completed()`.

#### Global State
- **`_global_worker`**: Singleton `AsyncRolloutManager` instance.
- **`_worker_lock`**: Threading lock for manager creation.
- **`_semaphore_initialized`**: Whether global semaphores have been initialized.
- **`current_rollout_step`**: Module-level mirror of the current rollout step (also held in the `_RolloutStepHolder` actor).
- **`_ROLLOUT_STEP_ACTOR_NAME = "rollout_step_holder"`**

#### `class _RolloutStepHolder` (Ray remote actor, `num_cpus=0`)
Tiny detached Ray actor holding the current rollout step, readable from any process (`shared/fully_async_rollout.py:59`).

- **`set(self, step: int)`**: Stores the step.
- **`get(self) -> int`**: Returns the step.

#### `_get_rollout_step_holder() -> ActorHandle`
Returns the named actor (`ray.get_actor`), creating it detached on `ValueError` (`shared/fully_async_rollout.py:73`).

#### `class AsyncRolloutWorkerActor` (Ray remote actor, `num_cpus=1`)
Async rollout worker in a dedicated process with its own event loop (`shared/fully_async_rollout.py:86`). Each constructor call:
- Creates its own httpx client (`slime.utils.http_utils.init_http_client`) and `GenerateState`.
- Initializes wandb/weave via `_init_wandb` (secondary/shared run, `primary=False`).
- Reads `MAX_STEP_LAG` env var (default `2`).
- Installs temporary diagnostic instrumentation: `faulthandler`, a SIGTERM handler that dumps stacks, and an `atexit` log (root-causing silent worker death).

- **`_init_wandb(args)`** (static): Initializes tracking + weave if `args.use_wandb`.
- **`_step_lag_monitor()`** (async): Every 1s, polls the step holder and cancels any in-flight group whose `begin_rollout_step` is more than `MAX_STEP_LAG` behind, via its registered `anyio.CancelScope`.
- **`run()`** (async, fire-and-forget `.remote()` entry): Calls `enable_global_counter()`, starts the step-lag monitor, then spawns `self.concurrency` `_process_loop(task_id)` tasks. Each loop pulls `(group_id, group, sampling_params)` from the input Ray Queue (`None` = poison pill), stamps `metadata["begin_rollout_step"]` on each sample, runs `generate_and_rm_group` inside an `anyio.CancelScope`, and appends `(group_id, result)` to `_completed_groups`. If the scope was cancelled, every sample in the group is marked `ABORTED` with `abort_reason="step_lag"` and reward 0. Ends with DIAG logging if the gather ever returns (which orphans the actor's env-worker pool).
- **`drain_completed() -> list[tuple]`** (async): Atomically returns and clears the completed-group list (polled by the collector).

#### `get_global_worker(args, data_buffer) -> AsyncRolloutManager`
Gets or creates the global manager. On first call initializes the `container_launch` and `container_running` semaphores. Recreates the manager if its feeder thread died (`shared/fully_async_rollout.py:290`).

#### `stop_global_worker()`
Stops the manager and kills worker actors. Registered via `atexit` at module bottom.

#### `class AsyncRolloutManager`
Orchestrator living in the RolloutManager process (`shared/fully_async_rollout.py:317`).

- **`__init__(self, args, data_buffer, concurrency=10)`**: Worker count from `NUM_ASYNC_ROLLOUT_WORKERS` env var (default `16`); `concurrency_per_worker = max(1, args.over_sampling_batch_size // num_workers)`. Input `RayQueue(maxsize=args.over_sampling_batch_size)` (bounded → backpressure); `output_queue` is a legacy `RayQueue(maxsize=1_000_000)` no longer used for results. Both use `maxsize>0` so `put_async`/`get_async` route through the queue actor's event-loop thread (sync paths race the internal asyncio.Queue — the desync root cause). Workers are created with SPREAD scheduling (one per node where possible, co-locating with that node's env workers) and started via `worker.run.remote()` fire-and-forget.
- **`_feeder_thread_entry()`**: Thread entry; runs `_feeder_loop()` via `asyncio.run`.
- **`_feeder_loop()`** (async): Registers wandb metrics, calls `enable_global_counter()`, starts `log_counter_loop(interval=10.0, router_url=<sglang router>)`, then repeatedly pulls `data_buffer.get_samples(1)` (offloaded to a thread) and puts `(group_id_counter, group, sampling_params)` onto the input queue.
- **`start()`**: Starts the daemon feeder thread (idempotent).
- **`stop()`**: Sets `running=False`, joins the feeder thread, `ray.kill`s all workers.
- **`get_completed_groups(max_groups=0) -> list[tuple]`** (async): Fans out `drain_completed.remote()` to all workers and flattens results (per-worker failures are logged at debug level and skipped).
- **`get_queue_size() -> int`**: Deprecated; returns 0 (kept for log compatibility).

#### `generate_rollout_async(args, rollout_id, data_buffer) -> RolloutFnTrainOutput` (async)
Collector function (`shared/fully_async_rollout.py:462`). Publishes `rollout_id` to the step holder, then loops until `args.rollout_batch_size` groups are accepted:
1. Drains completed groups from all workers.
2. Drops aborted groups entirely, recording the first sample's `abort_reason` via `MetricGatherer.on_aborted`.
3. Applies the dynamic filter (loaded from `args.dynamic_sampling_filter_path` via `call_dynamic_filter`). Dropped groups are counted; if the drop reason is in the `EXCLUDE_ON_DROP_REASONS` env var set (comma-separated, e.g. `zero_std_1.0`), the group's `ORIGIN_SAMPLE_KEY` is added to `origin_key_to_drop`.
4. Prints a no-progress warning after 30s without new completions.
5. After the loop: if more than 10 origin keys are pending exclusion, all keys past the first 10 are permanently excluded via `data_buffer.exclude_samples(...)` (cheap guard against mass exclusion in a single rollout).
6. Sorts data by `group[0].index`, stamps `end_rollout_step` on every sample, computes per-sample step lags, and returns `RolloutFnTrainOutput` with metrics including `rollout/epoch_id` and `rollout/avg_step_lag` (metrics merge `MetricGatherer.collect()`).

#### `generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False) -> RolloutFnTrainOutput`
Synchronous entry point (`shared/fully_async_rollout.py:619`). Raises `ValueError` for `evaluation=True` (not supported). Runs `generate_rollout_async` via slime's `run()`.

---

### shared/data_source.py
Manages the dataset for rollout: reads prompts, supports replay buffers, and can permanently exclude samples.

#### Constants
- **`ORIGIN_SAMPLE_KEY = "origin_sample_key"`**: Metadata key tagging each sample with its stable index in `Dataset.origin_samples`.

#### `class RolloutDataSourceWithExclusion`
A self-contained data source that reads from a `Dataset`, supports a replay buffer, and can permanently exclude samples by their origin index.

- **`__init__(self, args)`**: Loads tokenizer, processor (skippable via `SLIME_DISABLE_PROCESSOR=1` for text-only models shipped with VLM configs), and dataset from `args.prompt_data`. Tags each origin sample with its stable index (stable across shuffles). Shuffles if `args.rollout_shuffle`. Initializes `epoch_id`, offsets, buffer (`self.buffer: list[list[Sample]]`), and `excluded_keys: set[int]`.
- **`get_samples(self, num_samples: int) -> list[list[Sample]]`**: Main data fetching method.
  1. Drains replay buffer first via `buffer_filter` (defaults to `_pop_first`, overridable via `args.buffer_filter_path`).
  2. Pulls remaining from the dataset, skipping excluded origin keys; wraps epochs automatically (`epoch_id += 1` + reshuffle). Logs a warning and stops if everything left is excluded.
  3. Expands each prompt into a group of `n_samples_per_prompt` deep copies with running `group_index`/`index` (for GRPO-style sampling).
- **`add_samples(self, samples: list[list[Sample]])`**: Appends sample groups to the replay buffer.
- **`save(self, rollout_id)`**: Saves state (offsets, epoch, exclusion set) to `rollout/global_dataset_state_dict_{rollout_id}.pt` under `args.save`.
- **`load(self, rollout_id=None)`**: Restores state from `args.load`, re-shuffles to the saved epoch, restores `excluded_keys`.
- **`exclude_samples(self, keys: list[int])`**: Permanently excludes origin-sample indices. Used by the async rollout when a dynamic filter decides the model already solves a task (e.g. `zero_std_1.0`).
- **`_get_samples_from_buffer(self, num_samples) -> list[list[Sample]]`**: Internal helper draining the buffer via the configured filter.
- **`get_buffer_length(self) -> int`**: Current buffer size.

#### `_pop_first(args, rollout_id, buffer, num_samples) -> list[list[Sample]]`
Default buffer filter: pops the first `num_samples` groups from the buffer (FIFO).

---

### shared/sglang_registry.py
A Ray-based service registry for discovering SGLang server endpoints at runtime, with a background health-eviction loop.

#### Constants
- **`MAX_FAILURES = 3`**: Consecutive `/health` failures before a URL is evicted.
- **`HEALTH_CHECK_INTERVAL = 5`**: Seconds between health check rounds.
- **`HEALTH_CHECK_TIMEOUT = 5`**: Per-request `/health` timeout in seconds.

#### `class SGLangRegistry` (Ray remote actor)
A detached Ray actor storing a `dict[str, list[str]]` mapping keys (like `"rm_worker"`, `"rm_router"`) to URL lists, plus a `_failure_counts` map keyed `(key, url)`.

- **`__init__(self)`**: Initializes the store, failure counters, and the health-check task handle.
- **`add(self, key: str, url: str)`**: Registers a URL (deduplicated); resets its failure count.
- **`get_one(self, key: str) -> str | None`**: Random URL for the key (basic load balancing), or `None`.
- **`get_all(self, key: str) -> list[str]`**: All URLs registered under a key.
- **`dump(self)`**: Deep copy of the whole registry.
- **`remove(self, key: str, url: str) -> bool`**: Removes one URL; deletes the key if emptied; clears the failure count.
- **`remove_key(self, key: str) -> list[str]`**: Removes all URLs for a key; returns the removed list.
- **`clear(self)`**: Clears all entries and failure counts.
- **`start_health_check(self)`**: Starts the background asyncio health loop (idempotent).
- **`stop_health_check(self)`**: Cancels the background health loop.
- **`_health_check_loop(self)`** (async): Runs `_check_all()` every `HEALTH_CHECK_INTERVAL` seconds; exceptions are logged, not fatal.
- **`_check_all(self)`** (async): Snapshots all `(key, url)` pairs and health-checks them concurrently with one `aiohttp` session.
- **`_check_one(self, session, key, url)`** (async): `GET {url}/health`. On success resets the failure count; on failure increments it, and on the 3rd consecutive strike calls `remove(key, url)` (3-strikes eviction).

#### `get_or_create_registry(name: str = "sglang_registry") -> ActorHandle`
Atomically gets or creates the named detached actor (`lifetime="detached"`, `namespace="sglang"`, `get_if_exists=True`) and fires `start_health_check.remote()` so eviction is always running.

---

### shared/sample_helpers.py
Helper functions for building multi-turn samples with proper token management, loss masks, log probabilities, and MoE routing replay. Implements a "pending + commit" pattern.

**Design**: All functions except `add_assistant_message` accumulate into `pending_*` fields in `sample.metadata`. `add_assistant_message` "commits" pending + assistant tokens to `sample.tokens`, so `sample.tokens` always ends after an assistant message. After commit, a chat-template scaffolding newline is appended to pending (`loss_mask_value=0`) when the template expects one.

**Update modes** (independent env-var toggles):
- **`USE_FULL_LOGPROBS`** (`SLIME_ROLLOUT_FULL_LOGPROBS`): when set, `add_assistant_message` overwrites the whole response span's `rollout_log_probs` from sglang's prefill-recomputed `input_token_logprobs` instead of appending per-turn generation log-probs.
- **`USE_FULL_ROUTING`** (`SLIME_ROLLOUT_FULL_ROUTING`): when set, `update_rollout_routed_experts` overwrites with this turn's full-sequence routing instead of appending only the new tail.

#### Internal Caches
- **`_prefix_length_cache`**: Chat-template prefix length per tokenizer.
- **`_generation_prompt_cache`**: Generation prompt text (e.g., `"<|im_start|>assistant\n"`).
- **`_newline_token_cache`**: Token id of `"\n"` per tokenizer.
- **`_has_post_eot_newline_cache`**: Whether the template puts `"\n"` after the eot token between messages.

#### `_get_prefix_length(tokenizer) -> int` / `_get_generation_prompt(tokenizer) -> str` / `_get_newline_token(tokenizer) -> int` / `_has_post_eot_newline(tokenizer) -> bool`
Cached template introspection helpers.

#### `_ensure_pending_fields(sample)` / `_ensure_token_length_fields(sample)`
Ensure `pending_tokens`, `pending_response`, `pending_loss_mask`, `pending_log_probs` / the four `*_token_length` counters exist in metadata.

#### `_add_to_pending(sample, tokens, text, loss_mask_value, log_probs=None)`
Low-level helper appending to pending fields (log-prob placeholders default to `0.0`).

#### `add_text(sample, text, state, loss_mask_value=1)`
Tokenizes `text` and appends to pending.

#### `add_message(sample, message, state, loss_mask_value=1)`
Applies the chat template to `{role, content}` and appends only the formatted message part (prefix stripped).

#### `add_user_message(sample, content, state, loss_mask_value=0)`
Adds a user message (masked out by default); tracks `user_token_length`; appends to `metadata["messages"]`.

#### `add_tool_response(sample, content, state, loss_mask_value=0)`
Adds a `role="tool"` message (masked out by default); tracks `tool_response_token_length`.

#### `add_system_message(sample, content, state, loss_mask_value=0)`
Adds a `role="system"` message (e.g. format-error feedback in web_agent).

#### `add_generation_prompt(sample, state, loss_mask_value=0)`
Adds the generation prompt to pending; tracked under `assistant_token_length`.

#### `_overwrite_full_rollout_log_probs(sample, input_log_probs, assistant_log_probs)`
Full-update mode helper: rebuilds the response span's log-probs as prefix (this turn's `input_token_logprobs`, sent with `logprob_start_len=0`) ++ this turn's generated logprobs, sliced to the last `response_length` entries and zeroed where `loss_mask == 0`. Falls back to incremental append when prefix log-probs are missing or short (e.g. full prefix-cache hit).

#### `add_assistant_message(sample, token_ids, state, loss_mask_value=1, log_probs=None, input_log_probs=None)`
**Commits** pending + assistant tokens to `sample.tokens` (the only function that modifies `sample.tokens`). Updates `response`, `response_length`, `loss_mask`, `rollout_log_probs` (incremental or full per `USE_FULL_LOGPROBS`), `assistant_token_length`, appends the assistant message to `metadata["messages"]` (eot token stripped), clears pending fields, then adds the post-eot scaffolding `"\n"` to pending with `loss_mask_value=0` when the chat template requires it (masked so it isn't trained on and doesn't corrupt TIS importance ratios).

#### `get_pending_token_count(sample) -> int`
Number of uncommitted pending tokens.

#### `update_rollout_routed_experts(sample, new_experts)`
Updates `sample.rollout_routed_experts` honoring the incremental/full mode. `new_experts` covers the entire request each turn (`(seqlen-1, num_layers, topk)`); incremental mode keeps earlier turns' actual routing and concatenates only the new tail.

---

### shared/fileops.py
High-performance file broadcast and all-gather across Ray cluster nodes using NCCL over GPU memory. Pipelined chunked transfers overlap disk I/O with network.

#### Low-level I/O
- **`_libc_pread_all(fd, ptr, length, offset)`**: libc `pread()` loop until all bytes read (raw C pointers; bypasses Python buffer allocation).
- **`_libc_pwrite_all(fd, ptr, length, offset)`**: libc `pwrite()` loop until all bytes written.

#### Data Classes
- **`BroadcastConfig`**: `chunk_size` (default 1GB), `num_buffers` (default 10), `device` (defaults to current CUDA device).
- **`FileEntry`**: `rel_path`, `size`, `num_chunks`; `chunk_ranges(chunk_size)` computes `(offset, length)` pairs.
- **`BroadcastManifest`**: List of `FileEntry` + `chunk_size` + `src_is_file`; JSON-serializable (`to_json`/`from_json`).
- **`ChunkDescriptor`**: One chunk in the flattened pipeline: `file_idx`, `chunk_idx`, `num_chunks`, `rel_path`, `offset`, `length`.
- **`AllGatherTransfer`**: One broadcast in an all-gather plan: `src_rank` + files.
- **`AllGatherPlan`**: Sequence of `AllGatherTransfer`s + `chunk_size`; JSON-serializable.

#### `class FileTransferWorker` (Ray remote, `num_gpus=1`)
One worker (one torch.distributed rank) per node.

- **`init_process_group(rank, world_size, master_addr, master_port)`**: NCCL init + CUDA context warmup.
- **`teardown()`**: Destroys the process group.
- **`get_node_ip() -> str`**: This node's IP.
- **`_build_manifest(src_path, chunk_size) -> BroadcastManifest`**: Scans a file or directory (recursive, sorted) into a manifest.
- **`_broadcast_manifest(manifest, src_rank, device) -> BroadcastManifest`**: Broadcasts manifest size then payload from `src_rank`.
- **`_broadcast_file_chunked(...)`**: Legacy stub — raises `NotImplementedError`; use `_broadcast_all_chunks`.
- **`_flatten_chunks(manifest) -> list[ChunkDescriptor]`**: Flattens `(file, chunk)` pairs into one ordered list (empty files skipped — handled separately).
- **`CHUNK_HEADER_SIZE = 8`**: Per-chunk header prepended to every payload: `[0:4] chunk_id (int32 LE)`, `[4:8] chunk_len (int32 LE)`. Every NCCL call transfers the fixed size `HEADER + max_chunk` so ranks stay in sync regardless of send order.
- **`_broadcast_all_chunks(manifest, src_root, dst_root, src_rank, config, pbar, bench_mode)`**: Core pipeline with `num_buffers` worker threads, each owning a pinned CPU buffer + GPU buffer + CUDA stream + events:
  - **Sender**: workers `pread` chunks into pinned memory + H2D; main thread pops ready buffers and NCCL-broadcasts them (unordered, header carries the chunk id).
  - **Receiver**: pre-allocates each output file as `<name>.incomplete` (`ftruncate` to final size), main thread NCCL-receives into free slots and parses the 8-byte header, workers D2H + `pwrite`, then all files are atomically renamed `.incomplete` → final.
  - `bench_mode="sender"` skips disk reads; `"receiver"` skips D2H/writes (pure NCCL throughput measuring).
- **`broadcast(src_path, dst_dir, src_rank, chunk_size, num_buffers, bench_mode)`**: Public API. Phases: build manifest → broadcast manifest → create empty files → chunked pipeline → barrier. `dst_dir` defaults to the source path (same path on all nodes).
- **`all_gather(src_dir, dst_dir, chunk_size, num_buffers, bench_mode)`**: Each rank contributes files from `src_dir`; ends with all files in `dst_dir` (defaults to `src_dir`). Phases: local manifest → gather manifests to rank 0 → rank 0 builds plan (dedup: first rank wins) and broadcasts it → execute one broadcast per contributing rank.
- **`_gather_manifests(local_manifest, device)`**: All-gathers manifests (padded payloads) to rank 0; `None` on other ranks.
- **`_build_and_broadcast_plan(all_manifests, chunk_size, device) -> AllGatherPlan`**: Rank 0 builds + broadcasts the plan.

#### `class FileTransferGroup` (Ray remote)
Coordinator managing one `FileTransferWorker` per GPU node.

- **`setup(master_port, worker_options, src_node_ip)`**: Discovers alive GPU nodes, optionally puts `src_node_ip` at rank 0, spawns workers pinned via `NodeAffinitySchedulingStrategy`, initializes torch.distributed in parallel.
- **`broadcast(...)`** / **`all_gather(...)`**: Runs the op on all workers in parallel, returns when all finish.
- **`teardown()`**: Destroys process groups and `ray.kill`s all workers.

#### `broadcast_files(src_path, dst_dir, chunk_size, num_buffers, master_port, worker_options, bench_mode) -> FileTransferGroup`
One-shot convenience: creates the group with the driver node as rank 0, broadcasts, returns the group for reuse (caller tears it down).

#### `all_gather_files(src_dir, dst_dir, chunk_size, num_buffers, master_port, worker_options, bench_mode) -> FileTransferGroup`
One-shot convenience for all-gather; returns the group.

---

### shared/ray_semaphore.py
Distributed semaphore system using Ray actors to limit concurrent operations across the cluster. Cancellation-safe via per-acquisition tickets.

#### `class GlobalSemaphore` (Ray remote actor)
Distributed counting semaphore with async support.

- **`__init__(self, name: str, max_concurrent: int)`**: Initializes counter, waiter queue, `_granted_tickets` set, and an asyncio lock.
- **`acquire(self) -> str`** (async): Acquires a slot, blocking on an asyncio Future if at capacity. Returns a unique **ticket** string that must be passed to `release`.
- **`release(self, ticket: str | None = None)`** (async): Releases the slot held by `ticket`. `None`/unknown tickets are silently ignored (idempotent under client-side cancellation races — prevents slot leaks from phantom releases). Wakes the next waiter if any.
- **`get_status(self) -> dict`**: `{name, max_concurrent, current_count, waiters, granted}`.

#### Constants
- **`SEMAPHORE_CONFIGS`**: Maps semaphore names to `(env_var, default)`:
  - `"container_launch"` → `("MAX_CONCURRENT_CONTAINER_LAUNCHES", 8)`
  - `"container_running"` → `("MAX_CONCURRENT_CONTAINERS_RUNNING", 32)`

#### `_get_actor_name(semaphore_name: str) -> str`
Returns `f"global_semaphore_{semaphore_name}"`.

#### `initialize_semaphore(name, max_concurrent=None) -> ActorHandle`
Creates (or gets) the named detached semaphore actor; limit from env var or `SEMAPHORE_CONFIGS` default (fallback 8).

#### `get_semaphore_actor(name: str) -> ActorHandle`
Gets the actor, initializing it if missing.

#### `class SemaphoreContext`
Cancellation-safe async context manager.

- **`__aenter__`**: Shields the remote acquire; if the caller is cancelled mid-acquire, `_schedule_orphan_release` detaches a task that releases the ticket the actor eventually grants (no leaked slots).
- **`__aexit__`**: Shielded release with the stored ticket.

#### `acquire_semaphore(name="container_launch") -> SemaphoreContext`
Returns the async context manager for the named semaphore.

#### `get_semaphore_status(name) -> dict` (async)
Returns the semaphore's current status.

#### `initialize_global_semaphore(max_concurrent=None)`
Backward-compatible wrapper initializing `"container_launch"`.

---

### shared/rollout_timer.py
Per-rollout timing indexed by `(name, sample_id)` using `contextvars` for asyncio-safe sample tracking.

#### Context Variables
- **`_current_sample_id: ContextVar[int | None]`**

#### `set_sample_id(sample_id: int)` / `current_sample_id() -> int | None`
Sets / gets the sample id for the current async context.

#### `get_sample_timers(sample_id=None) -> dict[str, float]`
Returns `{name: accumulated_elapsed}` for one sample (current context's sample if omitted). Generate functions stash this into `sample.metadata["timing"]`.

#### `class RolloutTimer` (Singleton via `SingletonMeta`)
Stores `{name: {sample_id: accumulated_elapsed}}`.

- **`add(self, name, sample_id, elapsed_time)`**: Accumulates elapsed time.
- **`reset(self, name=None)`**: Resets all timers or one named timer.
- **`log_dict(self) -> dict[str, float]`**: `{name_mean, name_max}` aggregated across sample ids.
- **`total_context(self, name)`**: Context manager measuring wall-clock time and adding it under the current context's sample id (raises `RuntimeError` if none set). Sequential reuse accumulates.

#### `rollout_timer_total(name: str)`
Convenience wrapper around `RolloutTimer().total_context(name)`.

---

### shared/rollout_log.py
Custom rollout logging reporting timers, token length, turn-count, and tool-call statistics to wandb.

#### Constants
- **`TOKEN_LENGTH_FIELDS`**: `["prompt_token_length", "user_token_length", "assistant_token_length", "tool_response_token_length"]`

#### `_ensure_wandb_metrics()`
Registers custom wandb metric definitions once: `token_length/*`, `turns/*`, `rollout/dynamic_filter/*`, `rollout/aborted/*`, `tool_calls/*` use `rollout/step` as x-axis; `counter/*` uses `counter/_elapsed_s`.

#### `_compute_token_length_metrics(samples) -> dict[str, float]`
`max`/`mean`/`q25`/`q75` per token-length field, plus `total_token_length_*` (sum of fields).

#### `_compute_tool_call_metrics(samples) -> dict[str, float]`
Counts `metadata["tool_calls"]` entries by `match_type` (`total`, `exact`, `generated`, `no_data`).

#### `_compute_turn_metrics(samples) -> dict[str, float]`
`num_turns_mean/q25/q75/max` across samples.

#### `log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool`
Called by the slime framework after each rollout:
1. Aggregates per-sample timing from `sample.metadata["timing"]` (written by each worker process's `generate()`) into `perf/*_mean` / `perf/*_max`.
2. Computes token length (`token_length/`), turn (`turns/`), and tool call (`tool_calls/`) metrics.
3. Logs to wandb via `logging_utils.log` with `rollout/step = rollout_id`.
4. Returns `False` to also trigger default logging.

---

### shared/metric_gatherer.py
Per-rollout counter for dynamic-filter drops and aborts (used by `generate_rollout_async`).

#### `class MetricGatherer`
Tracks drop counts/ratios and abort counts/ratios.

- **`on_generated()`**: Counts a completed (non-aborted) group reaching the filter stage.
- **`on_aborted(reason="unknown")`**: Counts an aborted group by reason.
- **`on_dynamic_filter_drop(reason)`**: Counts a filter drop by reason (empty reasons ignored).
- **`collect() -> dict`**: Emits `rollout/dynamic_filter/drop_<reason>` counts, `rollout/dynamic_filter/drop_ratio[_<reason>]` (drop ratio = dropped / (generated + aborted)), `rollout/total_completed`, `rollout/total_non_aborted`, `rollout/aborted_count`, `rollout/aborted_ratio`, and `rollout/aborted/<reason>` counts.

---

### shared/global_counter.py
Cluster-wide atomic counters (a detached Ray actor named `tau2_global_counter`) for tracking in-flight request states across the pipeline, plus a loop that scrapes SGLang router load metrics into wandb.

#### `class GlobalCounter` (Ray remote actor, `num_cpus=0`)
- **`inc(key) -> dict`**, **`dec(key) -> dict`**: Increment/decrement a counter; return the full map.
- **`get_all() -> dict[str, int]`**: Snapshot of all counts.

#### `enable_global_counter()`
Must be called explicitly during training init (in each worker actor's event loop); the counter is disabled by default so eval code paths don't accidentally create the actor.

#### `get_global_counter() -> ActorHandle`
Get-or-create the detached named actor (double-checked locking).

#### `class counter_scope`
Async context manager: increments `key` on enter, decrements on exit. Fire-and-forget (no await of the RPC); a no-op unless `enable_global_counter()` was called.

#### `_query_router_stats(client, router_url) -> dict` (async)
Queries one router's `GET /workers`, then each worker's `GET /get_load` and `GET /get_server_info`, returning `total_running`, `total_queued`, `total_tokens`, `total_throughput`, `num_workers`, `num_healthy` (per-worker failures swallowed; works for both the agent rollout router and the rm_router).

#### `log_counter_loop(interval=10.0, router_url=None)` (async)
Forever loop (started as a task by the rollout manager's feeder): every `interval` seconds, reads all counters, optionally merges the rollout router's stats (`router_*` keys), lazily discovers the rm_router from the sglang registry once and merges its stats (`rm_router_*` keys), logs a line, and sends `counter/*` to wandb (x-axis `counter/_elapsed_s`, wall-clock seconds since loop start).

---

### shared/http_utils.py
Instrumented HTTP POST used by rollout code; drop-in replacement for `slime.utils.http_utils.post` at the call site.

#### `post(url, payload, max_retries=60) -> Any` (async)
Uses slime's process-wide `_http_client` (raises `RuntimeError` if `init_http_client(args)` wasn't called). Splits each request into three `counter_scope` phases so time inside `assistant_generate` can be attributed:
- `assistant_generate_encode`: orjson-encode the payload.
- `assistant_generate_inflight`: HTTP round trip (what the router "sees").
- `assistant_generate_decode`: orjson-decode the response, offloaded to a worker thread (large `return_routed_experts` base64 blobs); falls back to raw text on decode failure.

Retries up to `max_retries` with 1s backoff, logging the response body for `HTTPStatusError`.

---

### shared/tool_call_parser.py
Shared local tool-call parser wrapping sglang's `FunctionCallParser` (no server round-trip). Used by `tool_call_agent`, `web_agent` (with `strip_thinking=True`), and `tau2_bench`.

#### Constants
- **`DEFAULT_TOOL_CALL_PARSER`**: From `TOOL_CALL_PARSER` env var, default `"qwen25"`.

#### `_strip_thinking(text) -> tuple[str, str]`
Splits `<think>...</think>` from the response, returning `(thinking, rest)`.

#### `parse_tools(response, tools, parser=DEFAULT_TOOL_CALL_PARSER, strip_thinking=False) -> dict`
Builds sglang `Tool`/`Function` objects and runs `FunctionCallParser.parse_non_stream` locally. With `strip_thinking=True`, thinking is stripped before parsing (safe for all parsers) and prepended back onto `normal_text`. Returns `{normal_text, calls}` where `calls` is a list of `model_dump()` dicts.

#### `class OpenAIToolCall` / `class OpenAIAssistantMessage` (dataclasses)
OpenAI-format tool call (`id`, `type="function"`, `function={name, arguments}`) and assistant message (`role`, `content`, `tool_calls`).

#### `class OpenAICompatibleToolCallAdapter`
- **`__init__(self, tools_info, parser_type, strip_thinking=False)`**
- **`parse_response_to_openai_format(self, response) -> dict`**: Returns `{openai_message, parsed_result, success}`; on error `{success: False, error: str}`.
- **`_convert_to_openai_message(self, normal_text, calls) -> OpenAIAssistantMessage`**: Converts parsed calls (`id=f"call_{i}_{name}"`).
- **`get_openai_tools_format(self) -> list[dict]`**: Tools in OpenAI format.

#### `create_openai_adapter(tools_info, parser_type, strip_thinking=False) -> OpenAICompatibleToolCallAdapter`
Factory function.

---

### shared/utils.py
Small shared utilities.

#### `get_random_free_port() -> int`
Binds port 0 with `SO_REUSEADDR` and returns the OS-assigned port (thread/process-safe at allocation time, small TOCTOU window).

---

## scripts/

### scripts/__init__.py
Empty. Makes `scripts` a Python package.

---

### scripts/sglang_job.py
Launches one or more SGLang reward model servers as Ray actors. Registers them in the SGLang registry and adds them to an RM router if one exists.

#### Constants
- **`WORKER_REGISTRY_KEY = "rm_worker"`**
- **`ROUTER_REGISTRY_KEY = "rm_router"`**

#### `_wait_server_healthy(base_url, api_key, is_process_alive)`
Polls `/health_generate` until the server is ready. Raises `RuntimeError` if the process dies.

#### `launch_server_process(server_args) -> multiprocessing.Process`
Starts the SGLang server via `multiprocessing` spawn → `sglang.srt.entrypoints.http_server.launch_server`. Waits for health.

#### `class RewardSGLangActor` (Ray remote)
Manages one SGLang reward model server.

- **`__init__(self, args, registry_name)`**: Stores args, process handle, URL.
- **`start(self) -> str`**: Builds `ServerArgs.from_cli_args`, forces `host=0.0.0.0` on a random free port, launches, returns `http://<node_ip>:<port>`.
- **`register(self)`**: Registers self as `rm_worker`; if an `rm_router` exists in the registry, adds self via `POST /workers`.
- **`_add_to_router(self, router_url)`**: HTTP POST `{url}` to the router.
- **`wait_forever(self)`**: Blocks; raises if the server process dies.

#### `parse_args()`
Parses launcher args (`--num-gpus` (float, default 1), `--num-nodes` (independent instances, default 1), `--registry-name`, `--node-ip` (pin actors via a `node:<ip>` custom resource)) plus all SGLang `ServerArgs`. If `--tp` was left at 1 and `num_gpus > 1`, defaults `tensor_parallel_size` to `num_gpus`.

#### `main()`
Creates `num_nodes` actors, starts and registers them all in parallel, explicitly starts the registry health check (`registry.start_health_check`), then blocks on `wait_forever` for all actors (dies when any server dies).

---

### scripts/rm_router.py
Launches a SGLang router for load-balancing reward model servers. Runs as a Ray job (dies when the job stops).

#### Constants
- **`ROUTER_REGISTRY_KEY = "rm_router"`**

#### `_get_free_port() -> int`
Free port with `SO_REUSEADDR`.

#### `wait_for_router(url, timeout=30) -> bool`
Polls `/health` until the router is ready.

#### `_run_as_ray_job()`
`exec`s into `ray job submit --address=auto --no-wait -- python scripts/rm_router.py --_ray-job`.

#### `_run_router()`
Inside the Ray job: starts `sglang_router.launch_router` as a subprocess (`--policy random`, random ports incl. prometheus port), waits for health, registers itself as `rm_router`, health-checks each existing `rm_worker` and adds healthy ones via `POST /workers`, then blocks on the subprocess.

#### Entry point
If `--_ray-job` in argv: `_run_router()`; otherwise `_run_as_ray_job()`.

---

### scripts/monitor_rm.py
Real-time terminal dashboard for reward model servers; queries the router or falls back to direct worker queries.

#### `get_router_url() -> str | None` / `get_worker_urls() -> list[str]`
Reads `rm_router` / `rm_worker` from the sglang registry (namespace `sglang`).

#### `get_router_metrics(router_url, debug=False) -> list[dict]`
Fetches the worker list from `GET /workers`, then per worker `GET /get_load` (handles both list- and dict-shaped responses) and `GET /get_server_info` (`internal_states[0].last_gen_throughput`). Returns per-worker `{url, healthy, running, queued, tokens, throughput}`.

#### `get_direct_metrics(worker_urls, debug=False) -> list[dict]`
Fallback: `GET /get_load` per worker.

#### `monitor(interval=2.0, debug=False)`
Clears the screen and prints a table (Worker / Running / Queued / Tokens / Tput / Tput-Per-Req + totals, or a simplified direct-query table), refreshing every `interval` seconds. `--debug` prints raw API responses on the first run.

#### `main()`
Parses `--interval` / `--debug`, runs until Ctrl+C.

---

### scripts/sglang_registry_cli.py
CLI for operating the shared SGLang registry. Global flag `--registry-name` (default `sglang_registry`).

#### Subcommands
- **`dump`**: Prints all registry entries as JSON (`cmd_dump`).
- **`add <key> <url>`**: Registers a URL (`cmd_add`).
- **`remove <key> [--url <url>]`**: Removes one URL or the whole key; exits 1 if not found (`cmd_remove`).
- **`clear`**: Clears all entries, reporting counts (`cmd_clear`).

#### `main()`
Parses args, `ray.init(address="auto")`, dispatches via `get_or_create_registry`, shuts Ray down afterwards.

---

### scripts/clear_sglang_registry.py
One-shot script clearing all registry entries, printing before/after state and warning if non-empty afterwards.

#### `main()`
`ray.init` → dump → `clear` → dump → `ray.shutdown`.

---

### scripts/broadcast_files.py
CLI wrapper for `shared.fileops.broadcast_files`. Broadcasts files/directories from the driver node to all GPU nodes via NCCL, then tears the group down.

#### `main()`
Args: `src` (positional), `--dst` (defaults to same path as src), `--chunk-size` in MB (**default 1024** — the `--help` text still says 256, but the argparse default is 1024), `--num-buffers` (default 10), `--bench-mode` (`sender`/`receiver`).

Usage: `ray job submit --address=auto --working-dir . -- python -m scripts.broadcast_files /path/to/files/`

---

### scripts/all_gather_files.py
CLI wrapper for `shared.fileops.all_gather_files`. All-gathers files across all GPU nodes via NCCL, then tears down.

#### `main()`
Same args as `broadcast_files.py` (`--chunk-size` default 1024 MB).

Usage: `ray job submit --address=auto --working-dir . -- python -m scripts.all_gather_files /path/to/shards/`

---

### scripts/run_on_each_node.py
Runs an arbitrary command once on each alive Ray node and waits for completion.

#### `class PerNodeCommandRunner` (Ray remote)
- **`run(self, cmd, extra_env)`**: Runs the command with inherited stdout/stderr and the extra env; returns `"ok"` or an error message.

#### `main()`
Parses `--no-gpu` (request 1 CPU instead of the default 1 CPU + 8 GPUs per node) and repeatable `--env KEY=VALUE`; remaining args are the command. Creates one actor per alive node (pinned via `NodeAffinitySchedulingStrategy`), runs the command on all in parallel via `ray.get`.

---

### scripts/stop_train_job.py
Stops running Ray jobs whose entrypoint contains `train_async.py`, via `JobSubmissionClient(address="auto")`. Prints and stops each match.

---

### scripts/stop_all_ray_jobs.py
Stops all RUNNING Ray jobs (with a submission id) via `JobSubmissionClient`, without stopping the cluster.

---

### scripts/repro_ray_queue_bug.py
Standalone reproduction of the `ray.util.queue` desync that motivated the `drain_completed()` design in `shared/fully_async_rollout.py`: with `maxsize=0`, `put_async`/`get_nowait`/`qsize` become sync methods running on thread-pool threads of an async actor, concurrently mutating a non-thread-safe `asyncio.Queue` until `qsize()` counts items that `get`/`get_nowait` can never retrieve.

#### `_make_group(toklen=4000, n_samples=8)`
Builds a realistic ~0.5MB group (token/logprob lists + nested metadata), like a tau2 group of 8 trajectories.

#### `class Producer` (Ray remote, `num_cpus=0.1`)
Mimics an `AsyncRolloutWorkerActor`: `produce(q, n, big_item, slow)` puts `n` groups via `put_async` (with optional 50ms trickle).

#### `run_case(maxsize, num_producers, items_each, big_item, drain_mode, label, slow=False)` (async)
Runs one 60-second scenario, counting `put`/`got`/`leaked`/`stuck_obs` (observations of `qsize>0` but nothing drainable) and prints a `BUG`/`ok` verdict.

#### `main()` (async)
Runs the case matrix: tau2-faithful (`maxsize=0`, 16 producers, big groups, slow trickle) drained via `get_nowait` vs `get_async`; fast/high-volume; small-item control; and the working input-queue pattern (`maxsize=1_000_000` + `get_async`).

Run: `uv run python scripts/repro_ray_queue_bug.py` against a live cluster.

---

### scripts/download_convert_model.sh
Downloads an HF model and converts it to slime's `torch_dist` format. Intended to run inside the container (`/workdir`); sources `/workdir/.venv/bin/activate`.

- Args: `-m/--model` (HF name, or `HF_MODEL_NAME`), `-c/--config` (model config name from `thirdparty/slime/scripts/models/`, or `MODEL_CONFIG`), `-o/--output` (`SAVE_PATH`, default `/tmp/instance_storage/<config>_torch_dist`), `-d/--download` (`DOWNLOAD_PATH`, default `/tmp/instance_storage/<config>`).
- Validates the config file exists, runs `hf download`, then `python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint ... --save ...` with `MODEL_ARGS` sourced from the config.
- Typically driven cluster-wide via `scripts/run_on_each_node.py`.

---

### scripts/convert_all_checkpoints.sh
Converts all `iter_*` torch_dist checkpoints under a training run directory to HF format using `thirdparty/slime/tools/convert_torch_dist_to_hf.py`.

- Args: `--input-dir` and `--output-dir` (required), `--origin-hf-dir` (original HF model, for config/tokenizer), `--extra-args` (passed through, e.g. `"--chunk-size 5368709120 --vocab-size 152064"`).
- Iterates `iter_*` dirs in sorted order; skips checkpoints whose output already has `config.json`; runs the converter with `--force`; prints a summary and exits 1 if any conversion failed.

---

## tool_call_agent/

MCP tool-calling agent: multi-turn rollout loop, RAG-based tool simulation, distributed tool execution, and evaluation.

### tool_call_agent/__init__.py
Empty. Makes `tool_call_agent` a Python package.

---

### tool_call_agent/generate.py
Core rollout logic for the tool-calling agent: multi-turn loop where the LLM calls tools, gets simulated responses, and iterates. See `tool_call_agent/DATA_FORMAT.md` for the sample format.

#### Constants
- **`MAX_TURNS = 10`**: Max agent turns per episode.
- **`MAX_TOKENS = 1024 * 64`**: Whole-session token cap (committed + pending + this turn's `max_new_tokens`).
- **`DEFAULT_SYSTEM_PROMPT`**: Used when `metadata["system_prompt"]` is absent (one tool at a time, JSON answer).

#### `_init_weave(args) -> bool`
Initializes Weave tracing if wandb is enabled (idempotent).

#### `build_prompt(metadata, tokenizer) -> tuple[str, list[dict]]`
Builds the initial prompt from metadata: required `task_description`; optional `answer_schema` (placeholder template appended to the user message) and `system_prompt`. Applies the chat template with `metadata["tools"]`.

#### `agent_turn(turn, url, sample, sampling_params, args, state, tool_adapter) -> dict` (async, weave traced)
Executes one turn (`tool_call_agent/generate.py:144`):
1. Token-budget check against `MAX_TOKENS` (includes pending tokens) — sets `TRUNCATED` (or `ABORTED`/`prompt_too_long` if nothing generated yet) and returns `finish_reason="length"`.
2. `add_generation_prompt`, builds `input_ids = sample.tokens + pending`.
3. Payload: `input_ids`, `sampling_params`, `return_logprob=True`; adds `logprob_start_len=0` in full-logprobs mode and `return_routed_experts=True` if `args.use_rollout_routing_replay`.
4. **Abort-retry loop**: POSTs to sglang `/generate` (inside `counter_scope("assistant_generate")` and `rollout_timer_total("assistant_turn")`); if `finish_reason == "abort"` (server reloading weights), sleeps 1s and retries indefinitely.
5. Extracts generated tokens + logprobs (`output_token_logprobs`), commits via `add_assistant_message` (incremental/full logprobs per `SLIME_ROLLOUT_FULL_LOGPROBS`; `input_log_probs` passed in full mode).
6. MoE routing replay: decodes base64 `meta_info["routed_experts"]` → int32 array reshaped `(len(sample.tokens)-1, args.num_layers, args.moe_router_topk)`, then `update_rollout_routed_experts`.
7. Parses tool calls via the adapter and executes each one via `perform_tool_call` (timed `tool_turn`, `counter_scope("tool_call")`), appending `{id, round_index, name, arguments, result, match_type}` to `metadata["tool_calls"]`.
Returns `{finish_reason, output_text, tool_results}` — tool responses are added by the caller.

#### `sample_rollout(sample_index, prompt, url, sample, sampling_params, args, state, tool_adapter) -> Sample` (async, weave traced)
Loops up to `MAX_TURNS`: feeds previous `tool_results` back via `add_tool_response`, calls `agent_turn`, then checks terminal conditions — `abort` → `ABORTED` (`model_abort`); `length` → `TRUNCATED`; no tool calls → `COMPLETED`; loop exhaustion → `TRUNCATED`. Stores `metadata["num_turns"]`.

#### `generate(args, sample, sampling_params) -> Sample` (async, weave traced)
**Main entry point** called by slime (`tool_call_agent/generate.py:402`):
1. `set_sample_id`, `counter_scope("rollout")`, timer `generate`; builds `GenerateState` and the sglang router `/generate` URL.
2. Builds the prompt, initializes `metadata["messages"]`, `metadata["tool_calls"]`, token state, and token-length counters.
3. Runs `sample_rollout` under a 3600s `asyncio.timeout`.
4. Computes reward via `reward_func` (`counter_scope("reward_judge")`).
5. Asserts routing-replay shape when enabled; stashes `get_sample_timers()` into `metadata["timing"]`.
Exception handling: `TimeoutError` → `ABORTED`/`timeout`, reward 0; any other exception → `ABORTED`/`exception:<Type>`, reward 0 (and slices over-long routed experts defensively).

#### Reward: exact match → LLM-as-a-judge
- **`_rm_urls`, `_judge_model`**: Cached judge endpoints/model (resolved once).
- **`JUDGE_SYSTEM_PROMPT` / `JUDGE_PROMPT_TEMPLATE`**: Judge prompt evaluating both answer correctness and tool-call correctness; expects JSON `{correct, answer_correct, tools_correct, reasoning}`.
- **`_extract_json(text) -> dict | None`** (weave traced): Cuts everything before the last `</think>`, strips ```` ```json ```` fences, `json.loads`.
- **`_format_gt_trajectory(trajectory)` / `_format_model_trajectory(tool_calls)`**: Formats trajectories for the judge prompt (inputs truncated to 200 chars).
- **`_get_judge_endpoint(sample_index) -> tuple[str, str]`** (async): Prefers the `rm_router` URL(s) from the sglang registry, falls back to `rm_worker` list; distributes across multiple URLs by `sample_index % len`. Judge model from the **required** `JUDGE_MODEL` env var.
- **`_llm_judge(task_description, expected_answer, model_answer, ground_truth_trajectory, model_tool_calls, sample_index) -> dict`** (async): POSTs `{base_url}/v1/chat/completions` with `temperature=0.0`, `max_tokens=8192`; parses the JSON verdict (`correct=False` with a parse-failure reasoning if unparseable).
- **`_exact_match(expected, actual) -> bool`**: All expected keys present and equal after `strip().lower()`.
- **`reward_func(args, sample, **kwargs) -> float`** (async, weave traced): `0.0` for aborted samples. Extracts the model's final answer from the last assistant message, exact match → `1.0`; otherwise LLM judge (`metadata["expected_answer"]`, `ground_truth_trajectory`, `tool_calls`) → `1.0`/`0.0`; judge exceptions → `0.0`.

---

### tool_call_agent/tool_call_simulator.py
RAG-based tool call simulation over historical trajectory DAGs: exact hash match → fuzzy top-5 similar calls → LLM generation.

#### `class ToolInfo` (dataclass)
Tool metadata from `mcp_servers_joined.json`: `tool_id` (`<server>::<name>`), `tool_name`, `server_name`, `description`, `input_schema`; `to_prompt_string()` for LLM prompts.

#### `class ToolCall` (dataclass)
Historical call: `tool_id`, `tool_name`, `tool_input`, `tool_output`, `is_error`, plus computed `input_hash` (SHA-256 of key-sorted JSON) and `batch_id`/`task_id` provenance. `to_example_string()` formats an LLM example.

#### `class SimulationResult` (dataclass)
`tool_id`, `tool_input`, `tool_output`, `is_error`, `match_type` (`"exact"` / `"generated"` / `"no_data"`), `confidence`, `similar_calls`.

#### `class ToolCallSimulator`
- **`__init__(llm_client=None, model_id=...)`**: `tool_calls` (tool_id → calls), `exact_match_index` (tool_id → input_hash → calls), `tool_info` (tool_id → ToolInfo).
- **`create(data_batches_path, servers_path, llm_client, model_id) -> ToolCallSimulator`** (async classmethod): Loads tool info then trajectories.
- **`load_tool_info(servers_path)`** (async): Loads `qualifiedName` + tools from `mcp_servers_joined.json`.
- **`load_trajectories(data_batches_path)`** (async): Accepts a single batch dir (containing `dag_trajectories.json`) or a parent of batch dirs; loads every DAG node from every trajectory (`request_id` → `task_id`, dir name → `batch_id`), then builds `exact_match_index`.
- **`_load_trajectory_file(filepath) -> int`** (async): Loads one file; returns the call count.
- **`call_tool(tool_id, tool_input, batch_id="", task_id="") -> SimulationResult`** (async):
  1. **Exact match** (same input hash): returns the cached output; prefers the candidate matching both `batch_id` and `task_id` (ground-truth trajectory).
  2. **Fuzzy match**: `_find_similar_calls(top_k=5)` — ground-truth candidates first, then similarity score descending. No candidates → error output, `match_type="no_data"`.
  3. **Generation**: `_generate_simulated_result` → `match_type="generated"`, confidence 0.7.
- **`_find_similar_calls(tool_id, tool_input, top_k, batch_id, task_id)`** (async): Scores all calls of the tool and sorts by `(-is_ground_truth, -score)`.
- **`_compute_similarity(input_a, input_b) -> float`**: Weighted score — 80% value similarity on matched query keys + 20% key coverage − 0.02 per extra candidate key; "config" keys (`format`, `language`, `limit`, `*_format`, `*_limit`, etc.) get weight 0.3.
- **`_value_similarity(val_a, val_b) -> float`**: Equality → 1.0; strings/lists via `rapidfuzz.fuzz.token_sort_ratio`; numbers via relative difference; dicts recurse.
- **`LONG_OUTPUT_THRESHOLD = 2000`**: If any example output is ≥ this many chars, use the constrained mode.
- **`_generate_simulated_result(tool_id, tool_input, similar_calls) -> tuple[str, bool]`** (async): Without an LLM client, returns the most similar call's output. Otherwise dispatches on the threshold.
- **`_generate_free(...)`** (async): For short outputs. Prompts the LLM with up to 5 examples (`[EX1]`…); the LLM answers `USE_EXAMPLE:<id>` or free-generates a new output (Anthropic-format request, `max_tokens=40960`, thinking budget 2048; wrapped in `counter_scope("tool_sim_llm")`). Unknown id → first example. Errors → fallback text prefixed `[LLM generation failed: ...]`.
- **`_generate_pick_or_error(...)`** (async): For long outputs. The LLM sees input-only summaries and MUST answer exactly `PICK:<id>` or `NO_MATCH` (`max_tokens=256`) — no synthesized long payloads; invalid ids treated as no match → structured error JSON.
- **`_no_match_error(tool_id, tool_input)`** (static), **`_extract_llm_text(result)`** (static, first non-empty text block), **`_looks_like_error(text)`** (static heuristic).
- **`get_stats() -> dict`**, **`list_tools() -> list[str]`**, **`get_tool_examples(tool_id, limit=5) -> list[dict]`**.
- **`main()`** (async): Demo/self-test (`python tool_call_agent/tool_call_simulator.py`).

---

### tool_call_agent/tool_workers.py
Distributed tool execution: one `ToolWorker` Ray actor per node running a `ToolCallSimulator`, keeping simulation off the rollout workers' event loops.

Required env vars (fail fast): `TRAJECTORY_PATH` (data batches dir; last component becomes `batch_id`), `SERVERS_PATH` (`mcp_servers_joined.json`), `JUDGE_MODEL`.

#### `_normalize_tool_name(tool_id) -> str`
`@org/server::tool` → `org_server__tool` (matches `convert_mcp_to_training.py` / evaluator convention).

#### `class ToolWorker` (Ray remote, `num_cpus=1`)
- **`__init__(node_id, node_index)`**: Reads env config.
- **`init()`** (async): Resolves the judge endpoint — `rm_router` (`api_base=<router>/v1`) preferred, else `rm_worker[node_index % len]` for a uniform spread; builds a `BatchInferenceClient` (`hosted_vllm/<JUDGE_MODEL>`, litellm mode), creates the `ToolCallSimulator`, builds the reverse name map, logs stats.
- **`_build_name_mapping()`**: normalized name → original `tool_id` (covers both registry tools and trajectory tools).
- **`execute(tool_name, tool_args, task_id="") -> tuple[str, str, str]`** (async): Maps the (possibly hallucinated) normalized name back to a `tool_id` (falls back to raw name → simulator returns `no_data`), calls the simulator with `batch_id`/`task_id`. **Output cap**: outputs longer than `20 * 2024 * 3` chars are replaced with the string `"error"` (match_type preserved). Returns `(output, node_id, match_type)`.

#### `class ToolWorkerPool`
One worker per alive node (node-affinity pinned), all initialized in parallel; `get_worker()` round-robins; `num_workers` property.

#### `get_tool_pool() -> ToolWorkerPool` / `reset_tool_pool()`
Global singleton accessors.

#### `perform_tool_call(tool_name, tool_args, task_id="") -> tuple[str, str]` (async, weave traced)
Public API: lazily initializes the pool, dispatches to the next worker, returns `(result, match_type)`.

---

### tool_call_agent/evaluate.py
Checkpoint evaluator: runs models on MCP tasks using simulated tool calls, computing pass@k.

#### Constants
- **`JUDGE_SYSTEM_PROMPT` / `JUDGE_PROMPT_TEMPLATE`**: Same judge prompt as training.

#### `debug_log(msg, data=None)`
`--verbose`-gated debug printer.

#### `_get_rm_router_url() -> str`
Prefers `rm_router` from the sglang registry; falls back to a random `rm_worker`; raises if neither exists.

#### `compute_pass_at_k(num_correct, n, k) -> float`
Unbiased Codex pass@k estimator: per task `1 - C(n-c, k) / C(n, k)`, averaged over tasks.

#### `class Evaluator`
Runs multi-round conversations with simulated tools (Anthropic message-API shape: `anthropic_version`, `system`, `tools` with `input_schema`, `max_tokens=40960`).

- **`__init__(simulator, eval_client, rm_router_url, judge_model, semaphore, max_rounds=10, use_llm_judge=True, tool_mode="simple", random_tool_count=20, num_passes=4, task_timeout=7200, thinking=False, thinking_budget=10000)`**
- **`_get_tools_for_task(task) -> list[dict]`**: Tool mode `simple` (ground truth only), `dag` (+ all `all_dag_tool_calls`), `random` (+ `random_tool_count` tools sampled from tools seen in historical calls).
- **`_make_tool_def(tool_id) -> dict | None`**: Anthropic-format tool def; sanitizes invalid schemas; skips names failing `^[a-zA-Z0-9_-]{1,128}$`.
- **`_normalize_tool_name(tool_id)`** / **`_denormalize_tool_name(normalized_name, task)`**: Name conversion (checks ground truth → DAG → simulator registry).
- **`evaluate_task(task) -> dict`** (async): Conversation loop up to `max_rounds` — invokes the eval client, executes `tool_use` blocks via `simulator.call_tool` (with `_batch_id`/`source_request_id` for ground-truth prioritization), feeds `tool_result` blocks back. On completion: `_exact_match` first, then `_llm_judge` fallback (records `judge_reasoning`).
- **`_extract_json(text)`**: Strips thinking, tries ```` ```json ```` fences, whole text, then scans for the last `{...}` object.
- **`_exact_match(expected, actual)`** (static): Same semantics as training's reward.
- **`_llm_judge(...)`** (async): Judge via the RM router `/v1/chat/completions`; errors/unparseable → `correct=False` with reasoning.
- **`_format_gt_trajectory(...)`** / **`_format_model_trajectory(...)`** (static).
- **`run(tasks, label="Evaluating") -> dict`** (async): Launches all `(pass, task)` pairs concurrently under the shared semaphore (+ jitter sleep, per-task `asyncio.wait_for`); aggregates `per_task_correct`, `pass_at_k` (k=1..num_passes), `per_pass_accuracy`, `tool_call_stats`, `all_pass_results`.

#### `load_tasks(data_batches_path, num_tasks, seed, batch=None, min_difficulty=None, task_file_name="valid_tasks.json") -> list[dict]`
Loads tasks from one batch or all batches (tags each with `_batch_id`), filters by difficulty, deterministically samples with an isolated `random.Random(seed)`.

#### `resolve_checkpoints(checkpoint_path) -> list[str]`
Model ID (non-path) → as-is; dir with `config.json` → single checkpoint; dir with `iter_*` subdirs → all of them (sorted); else `FileNotFoundError`.

#### `_make_checkpoint_name(checkpoint) -> str`
`.../batch_hf/iter_0000059/` → `batch_hf__iter_0000059`; model IDs sanitized.

#### `class EvalCheckpointActor` (Ray remote)
One actor per checkpoint (own process/event loop; queued by Ray if GPUs are scarce).

- **`run()`** (async): Dispatches to `_run_external_model` (model ID not on disk) or `_run_sglang_model`; retries from scratch up to `max_retries=3` on failure.
- **`_run_external_model(args)`** (async): Bedrock via `BatchInferenceClient(debug_mode=True)`; judge + simulator via the RM router.
- **`_run_sglang_model(args)`** (async): Launches a per-checkpoint `RewardSGLangActor` (`--tool-call-parser`, `--tp num_gpus`, optional node pinning), builds eval/judge clients (litellm `hosted_vllm/`), runs the `Evaluator`, saves results, shuts the server down in `finally` (retries up to 3 attempts).

#### `_print_checkpoint_summary(result)` / `_save_result(result, output_dir, args, rm_router_url, judge_model, use_llm_judge)`
Per-checkpoint pass@k printout; timestamped `eval_<name>_<ts>.json` with full run metadata.

#### `main()`
CLI: `--checkpoint` (required), `--num-gpus`, `--region`, `--tool-call-parser` (default `$TOOL_CALL_PARSER` or `qwen`), task selection (`--num-tasks`, `--seed`, `--data-batches`, `--batch`, `--task-file`, `--servers`, `--min-difficulty`), `--max-rounds` (10), `--task-timeout` (7200), `--thinking` / `--thinking-budget` (0 = adaptive), `--output`, `--parallel` (400), `--no-llm-judge`, `--tool-mode`, `--random-tool-count`, `--long-output-threshold`, `--passes` (4), `--pin-node` (default: off — Ray schedules freely), **`--only-missing`** (skips checkpoints with an existing `eval_<name>_*.json`). `JUDGE_MODEL` env var with a Qwen3-235B default. Resolves the RM router, loads tasks once, launches one `EvalCheckpointActor` per checkpoint, tolerates individual failures, prints a summary.

---

### tool_call_agent/bfcl_evaluate.py
BFCL (Berkeley Function Calling Leaderboard) wrapper: launches a per-checkpoint SGLang server, runs `bfcl generate` + `bfcl evaluate` subprocesses, collects scores.

#### `resolve_checkpoints(checkpoint_path) -> list[str]` / `_make_checkpoint_name(checkpoint) -> str`
Same pattern as `evaluate.py`.

#### `_make_bfcl_env(checkpoint_output_dir, server_url=None) -> dict`
Builds the subprocess env: `BFCL_PROJECT_ROOT=<checkpoint_output_dir>` (so `result/` and `score/` land there), strips `REMOTE_OPENAI_*` (forces the local-server code path), sets `LOCAL_SERVER_ENDPOINT`/`LOCAL_SERVER_PORT` from the server URL.

#### `_run_bfcl_generate(bfcl_model, checkpoint, server_url, test_categories, checkpoint_output_dir, temperature=0.001, num_threads=100, allow_overwrite=True)`
`bfcl generate --skip-server-setup --local-model-path <ckpt> --backend sglang`. Raises on non-zero exit.

#### `_run_bfcl_evaluate(bfcl_model, test_categories, checkpoint_output_dir)`
`bfcl evaluate`. Raises on non-zero exit.

#### `_read_overall_scores(score_dir) -> dict | None`
First row of `score/data_overall.csv`.

#### `class BFCLCheckpointActor` (Ray remote)
Per-checkpoint pipeline (same GPU-queueing pattern as `EvalCheckpointActor`): launches `RewardSGLangActor` (`--tool-call-parser`, `--reasoning-parser deepseek-r1`), runs generate → evaluate, always kills the server, writes `metadata.json`, returns `{checkpoint, checkpoint_name, scores}`.

#### `SUMMARY_KEYS`
`["Overall Acc", "Non-Live AST Acc", "Live Acc", "Multi Turn Acc", "Memory Acc", "Web Search Acc"]`.

#### `_print_checkpoint_summary(result)` / `main()`
CLI: `--checkpoint`, `--bfcl-model` (default `Qwen/Qwen3-30B-A3B-Instruct-2507-FC` from BFCL's model registry), `--num-gpus`, `--tool-call-parser`, `--no-pin-node`, `--test-category` (default `all`), `--temperature` (0.001), `--num-threads` (default 800), `--allow-overwrite` (default on), `--output`. Launches one actor per checkpoint in parallel; writes a combined `summary.json`.

---

### tool_call_agent/convert_mcp_to_training.py
Converts MCP task JSON to slime training JSONL (`{"index", "metadata"}`); tool definitions inlined in OpenAI function-calling format.

#### `ToolMode = Literal["simple", "random", "dag"]`
- `simple`: only ground-truth trajectory tools.
- `dag`: ground truth + all DAG tools (`all_dag_tool_calls`).
- `random`: ground truth + N random tools from the full registry.

#### `normalize_tool_name(tool_id) -> str` / `validate_tool_name(name) -> bool`
Same `{normalized_server}__{tool_name}` convention as the evaluator; OpenAI/Bedrock name regex.

#### `load_tool_registry(servers_path) -> dict[str, dict]`
`tool_id` → OpenAI-format tool def from `mcp_servers_joined.json` (sanitizes schemas, strips `$schema`).

#### `_synthesize_tool_def(tool_id, tool_input) -> dict`
Infers a minimal schema from a trajectory call's actual inputs when the tool is missing from the registry.

#### `extract_tools_for_task(task, tool_registry, tool_mode, random_tool_count=20, all_tool_ids=None) -> tuple[list[dict], int, int]`
Builds the per-task tool list per mode; returns `(tools, registry_hits, registry_misses)`.

#### `convert_task(task, index, tool_registry, tool_mode, random_tool_count, all_tool_ids) -> tuple[dict, int, int]`
Metadata: `task_description`, `tools`, `answer_schema`, `expected_answer`, `ground_truth_trajectory`, `difficulty`; optional `answer_template`, `task_id` (from `task_id` or `source_request_id`), `batch_id` (from `BATCH_ID` env var).

#### `main()`
CLI: `--tasks`, `--servers`, `--output`, `--tool-mode`, `--random-tool-count` (20), `--seed` (42), `--filter-success` (default) / `--no-filter-success`. Skips non-success tasks, tasks without trajectories, and tasks with no valid tools; writes JSONL + summary.

---

## web_agent/

Browser-based web agent rollout logic. (Tool-call parsing moved to `shared/tool_call_parser.py`, used here with `strip_thinking=True`.)

### web_agent/__init__.py
Empty. Makes `web_agent` a Python package.

---

### web_agent/generate.py
Multi-turn browser interaction: the LLM generates `step_browser` tool calls, observes results, iterates.

#### Constants
- **`MAX_TURNS = 200`**: Max agent turns per episode.
- **`MAX_TOKENS = 1024 * 128`**: Whole-session token cap (committed + pending + `max_new_tokens`).
- **`FORMAT_PENALTY = -0.05`**: Flat reward penalty applied if any format error (bad tool call, wrong/multiple tools, browser exception) occurred during the rollout.

#### `_get_system_prompt_template() -> str`
Loads and caches `web_agent/prompts/tool_cot.txt`.

#### `_init_weave(args) -> bool`
Initializes Weave if wandb is enabled.

#### `build_initial_prompt(metadata, initial_observation, tokenizer) -> tuple[str, list[dict]]`
System message = template formatted with `metadata["intent"]`; first user message = initial browser observation; chat template applied with the browser tool schema.

#### `agent_turn(turn, url, sample, sampling_params, args, state, tool_adapter) -> dict` (async, weave traced)
Checks the token budget (`TRUNCATED`, or `ABORTED` if nothing generated), adds the generation prompt, and POSTs to sglang `/generate` with `return_logprob` (+ `return_routed_experts` when routing replay is on) — **retrying in 1s intervals while `finish_reason == "abort"`** (weight sync). Commits via `add_assistant_message`, handles `routed_experts` (base64 → int32 `(len(tokens)-1, num_layers, topk)`). Then parses the response through the adapter and returns format errors instead of executing: unparseable response, zero tool calls, >1 parallel tool calls, wrong tool name (must be `step_browser`), unparseable arguments JSON, or missing `action` key. A valid call is tracked in `metadata["tool_calls"]`. Returns `{finish_reason, output_text, action, format_error}`.

#### `browser_init(worker, instance_id, task_config) -> str` (async, weave traced)
Acquires the `container_launch` semaphore (`acquire_semaphore()` default) around `worker.create_instance`; returns the initial observation string.

#### `browser_step(worker, instance_id, action) -> tuple[str, bool, float]` (async, weave traced)
Executes an action; returns `(formatted_observation, terminated, score)`.

#### `sample_rollout(sample_index, url, sample, sampling_params, args, state, tool_adapter, worker, instance_id) -> Sample` (async, weave traced)
Loops up to `MAX_TURNS`: `agent_turn` → terminal checks (`abort`/`length`) → format errors and browser exceptions set `had_format_error`, feed the error back via `add_system_message`, and **continue**; valid actions run `browser_step`, observations are added via `add_tool_response`, `terminated` → `COMPLETED` (+ `metadata["model_answer"]`); loop exhaustion → `TRUNCATED`. Stores `num_turns`, `format_penalty` (`FORMAT_PENALTY` if any error), `had_format_error`.

#### `generate(args, sample, sampling_params) -> Sample` (async, weave traced)
**Main entry point**:
1. Acquires the `container_running` semaphore for the whole session.
2. Gets a **node-local** browser worker (`pool.get_local_worker(sample.index)`, falling back to round-robin) and creates the instance (`container_launch` semaphore inside `browser_init`).
3. Builds the prompt, initializes tracking, creates the adapter with `strip_thinking=True`.
4. Runs `sample_rollout` under a 1500s `asyncio.timeout`.
5. Reward via `reward_func`; asserts routing-replay shape; stashes `metadata["timing"]`.
Timeout/exception → `ABORTED` (`timeout` / `exception:<Type>`) with the accumulated format penalty preserved as reward. `finally`: releases the browser instance if created.

#### `reward_func(args, sample, worker, instance_id) -> float` (async, weave traced)
`reward = task_score + format_penalty`; aborted runs get only the accumulated penalty (no browser score). `task_score` from `worker.get_score` (0.0 on error).

---

### web_agent/browser_env.py
Browser environment management for distributed web agent training (wraps `rl_web_agent.env.WebAgentEnv`).

#### Constants
- **`_SETUP_MAX_ATTEMPTS = 5`**, **`_SETUP_BASE_DELAY_SECONDS = 2.0`**, **`_SETUP_MAX_DELAY_SECONDS = 60.0`**: Setup-retry policy (exponential backoff with full jitter).
- **`BROWSER_TOOL_SCHEMA`**: OpenAI-format schema for `step_browser` with actions `click`, `type`, `hover`, `select`, `clear`, `key_press`, `goto_url`, `back`, `forward`, `refresh`, `new_tab`, `switch_tab`, `close_tab`, `terminate`; params include `target`, `text`, `enter`, `value`, `key`, `url`, `tab_id`, `answer`.

#### `get_browser_tool_schema() -> dict`
Returns the schema.

#### `format_llm_observation(observation: dict) -> str`
Formats a browser observation for the LLM: active tab URL/title, HTML, clickable elements, input elements (type/value/focused/read-only), select elements, and the tab list.

#### `load_env_config() -> Any`
Loads `web_agent/conf/base.yaml` (`config.environment`) with env-var overrides: `INCUS_SERVER_URL`, `PROXY_SERVER`, `PROXY_ENABLED`, `BROWSER_HEADLESS`.

#### `class BrowserWorker` (Ray remote, `num_cpus=1`)
One actor per node, managing many browser instances.

- **`create_instance(instance_id, task_config) -> str`** (async): Builds a **fresh `WebAgentEnv` per attempt** (new container uuid / proxy session each retry, letting the scheduler place us elsewhere) and calls `env.setup`; on failure closes the env and retries with exponential backoff + full jitter (up to 5 attempts). Only stores in `_instances` after success. Returns the formatted initial observation.
- **`step(instance_id, action) -> tuple[str, bool, float]`** (async): Executes an action (dict or JSON string); returns `(formatted_observation, terminated, score)`.
- **`get_observation(instance_id) -> dict`** (async), **`get_score(instance_id) -> float`** (async), **`get_model_answer(instance_id) -> str | None`** (async, answer from a `terminate` action).
- **`release_instance(instance_id)`** (async): Closes and removes; always pops from `_instances` even if close fails.

#### `class BrowserWorkerPool`
One worker per alive node (node-affinity pinned), with a `node_id → worker` map.

- **`initialize()`**, **`get_worker_for_sample(sample_index)`** (round-robin), **`get_local_worker(sample_index=0)`** (returns the worker on the *caller's* Ray node; warns and falls back to round-robin), **`num_workers`** property.

#### `get_browser_pool() -> BrowserWorkerPool` / `reset_browser_pool()`
Global singleton accessors.

---

### web_agent/convert_webarena_to_training.py
Converts WebArena task JSON files to slime training JSONL, preserving all original fields: `{"index": int, "metadata": {<original task config>}}`.

#### `load_task_file(task_path) -> dict` / `convert_task(task_config, index) -> dict`
Loads one task; wraps it in `{index, metadata}`.

#### `main()`
CLI: `--tasks-dir` (required), `--output` (required), `--sites` (include filter), `--single-site-only`, `--exclude-sites`. Writes JSONL and prints a summary with site distribution.

---

## tau2_bench/

tau2-bench gym-integration agent (MoE token-in/token-out variant). Not an installed package — `tau2_bench/` itself is on `sys.path` at runtime, so modules import each other flat (`from tau2_env_workers import ...`, `from agent_moe import ...`).

### tau2_bench/generate_with_tau2_gym_moe.py
slime-facing entry point.

#### `generate(args, sample, sampling_params) -> Sample` (async)
Asserts `not args.partial_rollout`; lazily creates/initializes the `Tau2EnvWorkerPool`, dispatches the episode to the next worker (`worker.run_episode.remote(sample, sampling_params)`), and returns the result. **Error-sample isolation**: any exception (pool init failure, lost actor, env crash) is logged and returned as a minimal aborted Sample via `_create_error_sample(task_index, error_msg, prompt_text)` so one bad episode never takes down training.

---

### tau2_bench/tau2_env_workers.py
Pool of Ray actors running tau2-bench episodes, eliminating the default asyncio-thread-pool bottleneck on blocking `env.step`/`env.reset` calls.

#### Constants
- **`DEFAULT_WORKERS_PER_NODE = 32`**, **`DEFAULT_THREAD_POOL_SIZE = 256`**
- **`_PAD_TOKEN_ID = 0`**

#### Per-process task registration
- **`_get_tau2_domain() -> str`**: `TAU2_DOMAIN` env var, default `"telecom"`.
- **`_register_tasks()`**: Registers `_get_custom_tasks` into the tau2 `registry` under the domain name (once per process).
- **`_get_custom_tasks(split=None) -> list[Task]`**: Loads tasks from `TAU2_DATA_DIR` (default: `tau2_bench/data/`); telecom loads train + test jsonl, other domains train only. Handles both JSON-string and pre-parsed `evaluation_criteria`. **Reward-basis override**: telecom tasks with an empty basis or one containing `ENV_ASSERTION` get `[RewardType.ENV_ASSERTION]`; all other tasks get `[RewardType.DB, RewardType.COMMUNICATE]`.
- **`_create_error_sample(task_index, error_msg, prompt="") -> Sample`**: Minimal `ABORTED` sample (one pad token, reward 0, `metadata={"error", "task_id"}`).

#### `class Tau2EnvWorker` (Ray remote, `num_cpus=1`)
One episode-runner actor.

- **`__init__(worker_id, node_id)`**: Stores ids and uninitialized state.
- **`init(args)`** (async): Sets a `ThreadPoolExecutor(256)` as the event loop's default executor (so the orchestrator's blocking user-model calls never starve); builds `GenerateState`; registers tasks; sets the sglang `/generate` URL, `TAU2_MAX_TURNS` (env, default 100), routing-replay flag; resolves the user sim; initializes wandb/weave (shared run, `primary=False`); `init_http_client`; joins the global counter (`active_workers`).
- **`_resolve_user_sim()`** (async): User-simulator resolution, in priority order — `USER_SIM_MODEL` (local model served through the rm_router: `hosted_vllm/<model>` against `<router>/v1`, temperature 0.7) → `AZURE_USER_SIM_MODEL` (`azure/<deployment>`, `reasoning_effort="high"`, plus explicit `AZURE_API_KEY`/`AZURE_API_BASE`/`AZURE_API_VERSION` if set) → `BEDROCK_USER_SIM_MODEL` (raw litellm string). **All three env vars are required to have at least one set** — otherwise raises `RuntimeError` listing them.
- **`_init_wandb()`**: Attaches to the shared wandb run as a secondary process so weave traces land on the training run.
- **`run_episode(sample, sampling_params) -> Sample`** (async): Guards against uninitialized use (returns an error sample), extracts task info, creates a **fresh `AgentGymEnv`** (`domain`, `task_id`, `solo_mode=False`, `user_llm`, `user_llm_args`), `set_sample_id`, runs `run_tau2_env_loop_async_moe`, stashes `metadata["_timers"]`, and returns the sample. Exceptions → `_create_error_sample`. Tracks `active_episodes` in the global counter (inc/dec in finally).

#### `class Tau2EnvWorkerPool`
Per-process singleton; `initialize(args, workers_per_node=None)`: count from `TAU2_WORKERS_PER_NODE` env (default 32); creates all workers **on the current node** (`NodeAffinitySchedulingStrategy`, so episode dispatch ships Samples node-locally; total env workers = `NUM_ASYNC_ROLLOUT_WORKERS × workers_per_node`); concurrent init under an asyncio lock. `get_worker()` round-robins.

#### `get_tau2_env_pool() -> Tau2EnvWorkerPool`
Thread-safe global singleton accessor.

---

### tau2_bench/agent_moe.py
Multi-turn agent loop against the tau2 gym env with token-in/token-out generation and routed-experts collection.

#### Constants
- **`MAX_SESSION_TOKENS`**: Whole-session context limit from `TAU2_MAX_SESSION_TOKENS` env (default `32 * 1024`).
- **`AGENT_INSTRUCTION` / `SYSTEM_PROMPT`**: `{policy}`-templated system prompt.

#### `system_prompt(domain_policy) -> str`
Formats the system prompt.

#### `tau2_observation_to_messages(observation, incremental=True) -> list[dict]` (weave traced)
Converts structured tau2 `Message` objects (from `env._agent.observation`) into chat-message dicts. With `incremental=True`, returns only messages after the last `AssistantMessage` (mirrors the gym's default observation reset).

#### `parsed_response_to_action_str(parsed) -> str`
Converts a parsed response into the tau2 `step()` action string; **only the first tool call is used** (multi-call responses are dropped to one).

#### `decode_routed_experts(routed_experts_data, num_layers, moe_router_topk) -> np.ndarray`
Accepts both base64-int32 (older) and list (newer) sglang formats; returns `(num_tokens, num_layers, topk)`.

#### `@dataclass TurnResult`
Per-turn record: `response_text`, `new_token_ids`, `new_log_probs`, `routed_experts`, `routed_experts_token_count`, `terminated`, `truncated`, `aborted`, `parse_failed`, `called_tool_signature`, `called_tool_name`, `reward`, `obs`, `step_info`.

#### `final_reward_with_penalties(base_reward, had_tool_parse_failure, assistant_token_length, consecutive_same_tool_count, had_successful_find_name_by_tool_call) -> tuple[float, dict]`
Env-var-controlled penalties on top of the base reward:
- `TAU2_TOOL_FORMAT_PENALTY` (default 0): added when any turn failed tool-call parsing.
- `TAU2_MAX_ASSISTANT_TOKENS` (default 0 = off): if exceeded, `max(0, tokens - max) * TAU2_ASSISTANT_LENGTH_PENALTY_PER_EXCESS_TOKEN` (default **-0.0001** per excess token).
- `TAU2_CONSECUTIVE_SAME_TOOL_PENALTY` (default 0): per consecutive identical call pair (same name + args).
- **Fixed -0.5** when no successful `find_user_id_*` tool call occurred (parameter name says `find_name_by`; the check is the `find_user_id_` prefix).
Returns `(reward, info)` with a breakdown in `info`.

#### `run_single_turn_async(env, url, sample, state, sampling_params, tools_info, policy, turn_idx, num_layers, moe_router_topk, return_routed_experts=False) -> TurnResult` (async, weave traced)
One turn:
1. `input_ids = sample.tokens + pending`; POSTs to sglang with `return_logprob` (+ `return_routed_experts`). An `abort` finish reason → `TurnResult(aborted=True)`.
2. Extracts tokens/logprobs; strips a trailing `<|im_end|>`; **strips everything before the last `</think>` and extracts `<message>...</message>` content** before parsing.
3. `parse_tools` failure → `aborted=True, parse_failed=True`; otherwise computes the canonical tool signature (`name:sorted-args-json`).
4. Executes `env.step(action_str)` via `run_in_executor` (default executor = the actor's thread pool), timed and counted in the global counter (`active_env_steps`).

#### `run_tau2_env_loop_async_moe(env, url, sampling_params, sample, state, args, max_steps=100, return_routed_experts=False) -> Sample` (async, weave traced)
Wraps the inner loop with the `active_env_loops` global counter.

#### `_run_tau2_env_loop_inner(...) -> Sample` (async)
1. `env.reset` (executor) → policy, tools, task id; builds the prompt from `system_prompt(policy)` + initial observation (incremental=False), tokenizes, and initializes the sample (pending fields, `metadata = dict(info)`).
2. Per turn: converts new observation messages to `add_tool_response`/`add_user_message` (masked), adds the generation prompt, then enforces **`MAX_SESSION_TOKENS`** — projected overflow → `TRUNCATED` (or `ABORTED`/`prompt_too_long` if nothing generated).
3. Runs `run_single_turn_async`; on abort commits any generated tokens (parse failures keep them), sets `ABORTED`, breaks. Tracks consecutive identical tool signatures and the `find_user_id_`-prefix flag.
4. Commits each turn via `add_assistant_message` (loss mask 1); break on `terminated`/`truncated`.
5. Final: `sample.reward, penalty_info = final_reward_with_penalties(...)`; `metadata["reward_penalty_info"]`.
6. Routing replay: uses the **last turn's** full-sequence `routed_experts`, **asserts `shape[0] == routed_experts_token_count - 1`**, and slices `tokens`/`loss_mask`/`rollout_log_probs` if tokens grew past what the routing covers.
