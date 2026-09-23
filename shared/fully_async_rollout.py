"""
Fully async rollout for tool_call_agent.

CPU-heavy work (JSON serialization, tokenizer ops, list copies) is distributed
across multiple Ray actor workers, each with its own process and event loop.

Architecture:
  AsyncRolloutManager process
    └── feeder thread: data_buffer.get_samples() → input Ray Queue

  AsyncRolloutWorkerActor x N  (separate processes)
    └── each runs `concurrency` async tasks pulling from input queue,
        processing via generate_and_rm_group, pushing to output Ray Queue

  generate_rollout_async (collector)
    └── drains output Ray Queue until target_data_size groups collected
"""

import asyncio
import atexit
import contextlib
import faulthandler
import logging
import os
import signal
import threading
import time
import traceback

import anyio
import ray
import weave
from ray.util.queue import Queue as RayQueue
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.misc import load_function
from slime.utils.types import Sample

from shared.data_source import ORIGIN_SAMPLE_KEY
from shared.global_counter import enable_global_counter, log_counter_loop
from shared.metric_gatherer import MetricGatherer
from shared.ray_semaphore import initialize_semaphore

# Global worker manager
_global_worker = None
_worker_lock = threading.Lock()

# Track if semaphore has been initialized
_semaphore_initialized = False

# Current rollout step (updated each time generate_rollout_async is called)
current_rollout_step: int = 0

_ROLLOUT_STEP_ACTOR_NAME = "rollout_step_holder"


@ray.remote(num_cpus=0)
class _RolloutStepHolder:
    """Tiny Ray actor that holds the current rollout step, readable from any process."""

    def __init__(self):
        self._step: int = 0

    def set(self, step: int) -> None:
        self._step = step

    def get(self) -> int:
        return self._step


def _get_rollout_step_holder() -> ray.actor.ActorHandle:
    try:
        return ray.get_actor(_ROLLOUT_STEP_ACTOR_NAME)
    except ValueError:
        return _RolloutStepHolder.options(name=_ROLLOUT_STEP_ACTOR_NAME, lifetime="detached").remote()


# ---------------------------------------------------------------------------
# Ray actor worker — each runs in its own process with its own event loop.
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=1)
class AsyncRolloutWorkerActor:
    """Async rollout worker running in a dedicated Ray actor process.

    Runs a continuous loop with ``concurrency`` async tasks, each pulling
    groups from the shared input Ray Queue, processing them via
    generate_and_rm_group, and pushing results to the output Ray Queue.

    A background monitor task polls the current rollout step and cancels
    any in-flight groups whose begin_rollout_step has fallen behind by
    more than MAX_STEP_LAG steps.  Cancelled groups are emitted with
    every sample marked ABORTED (abort_reason="step_lag").
    """

    def __init__(self, worker_id, args, input_queue, output_queue, concurrency):
        self.worker_id = worker_id
        self.args = args
        self.input_queue = input_queue
        self.output_queue = output_queue  # legacy; no longer used for results (see below)
        self.concurrency = concurrency
        self._step_holder = _get_rollout_step_holder()
        self._max_step_lag = int(os.environ.get("MAX_STEP_LAG", "2"))

        # Completed groups accumulate here and are drained by the collector via
        # drain_completed(). We deliberately do NOT use a Ray Queue for results:
        # ray.util.queue.Queue wraps an asyncio.Queue whose qsize()/get() desync
        # under this workload (qsize() counts items get()/get_nowait() can never
        # retrieve), so groups silently never get collected.
        self._completed_groups: list[tuple] = []
        self._completed_lock = asyncio.Lock()

        # Lock-protected registry of in-flight cancel scopes keyed by
        # (task_id, group_id).  Values are (begin_step, cancel_scope).
        self._inflight_lock = asyncio.Lock()
        self._inflight: dict[tuple[int, int], tuple[int, anyio.CancelScope]] = {}

        # Each actor process gets its own httpx client and tokenizer
        from slime.utils.http_utils import init_http_client

        init_http_client(args)
        self.state = GenerateState(args)

        # Initialize wandb/weave early so log_counter_loop can use wandb.run
        self._init_wandb(args)

        print(f"[AsyncRolloutWorkerActor {worker_id}] initialized, concurrency={concurrency}")

        # --- DIAGNOSTIC INSTRUMENTATION (temporary; root-causing silent worker death) ---
        # Native fatal signals (SIGSEGV/SIGFPE/SIGABRT/SIGBUS/SIGILL) -> C-level stack to stderr.
        with contextlib.suppress(Exception):
            faulthandler.enable(all_threads=True)

        # If something SIGTERMs us, log it + dump all thread stacks before dying.
        def _diag_sigterm(signum, _frame, wid=worker_id):
            print(f"[DIAG][Worker {wid}] received SIGTERM ({signum}); dumping stacks then exiting", flush=True)
            with contextlib.suppress(Exception):
                faulthandler.dump_traceback(all_threads=True)
            os._exit(143)

        with contextlib.suppress(Exception):
            signal.signal(signal.SIGTERM, _diag_sigterm)

        @atexit.register
        def _diag_atexit(wid=worker_id):
            print(f"[DIAG][Worker {wid}] atexit: actor process is exiting", flush=True)

        # --- END DIAGNOSTIC INSTRUMENTATION ---

    @staticmethod
    def _init_wandb(args):
        """Initialize wandb + weave in this actor process if enabled."""
        if not getattr(args, "use_wandb", False):
            return
        try:
            from slime.utils.logging_utils import init_tracking

            from shared.rollout_log import _ensure_wandb_metrics

            init_tracking(args, primary=False)
            _ensure_wandb_metrics()
            weave.init(args.wandb_project)
            logging.getLogger("weave.trace.weave_client").setLevel(logging.WARNING)
            print(f"[AsyncRolloutWorkerActor] wandb/weave initialized (project={args.wandb_project})")
        except Exception as e:
            print(f"[AsyncRolloutWorkerActor] wandb/weave init failed: {e}")

    # -- step-lag monitor -----------------------------------------------------

    async def _step_lag_monitor(self):
        """Periodically poll the rollout step and cancel stale in-flight groups."""
        while True:
            await asyncio.sleep(1.0)
            try:
                current_step = await self._step_holder.get.remote()
            except Exception:
                continue

            async with self._inflight_lock:
                for key, (begin_step, scope) in list(self._inflight.items()):
                    if current_step - begin_step > self._max_step_lag:
                        print(
                            f"[Worker {self.worker_id}] Cancelling group {key} "
                            f"(begin_step={begin_step}, current_step={current_step}, "
                            f"max_lag={self._max_step_lag})",
                            flush=True,
                        )
                        scope.cancel()

    # -- main loop ------------------------------------------------------------

    async def run(self):
        """Main entry point — called once via .remote(), runs forever."""
        # Enable global counter and pool stats in each actor's event loop
        enable_global_counter()

        # Start the step-lag monitor as a background task
        self._monitor_task = asyncio.create_task(self._step_lag_monitor())

        async def _process_loop(task_id):
            while True:
                try:
                    item = await self.input_queue.get_async(block=True, timeout=2.0)
                except Exception:
                    # Timeout or empty — loop back and retry
                    continue

                if item is None:
                    # Poison pill — graceful shutdown
                    break

                group_id, group, sampling_params = item
                try:
                    # Stamp current rollout step as begin_rollout_step on each sample
                    rollout_step = await self._step_holder.get.remote()
                    for sample in group:
                        sample.metadata["begin_rollout_step"] = rollout_step

                    with anyio.CancelScope() as scope:
                        async with self._inflight_lock:
                            self._inflight[(task_id, group_id)] = (rollout_step, scope)

                        result = await generate_and_rm_group(
                            self.args,
                            group,
                            sampling_params=sampling_params,
                            evaluation=False,
                        )

                    # Remove from inflight registry
                    async with self._inflight_lock:
                        self._inflight.pop((task_id, group_id), None)

                    if scope.cancel_called:
                        # Mark every sample in the group as aborted
                        for sample in group:
                            sample.status = Sample.Status.ABORTED
                            sample.metadata["abort_reason"] = "step_lag"
                            if sample.reward is None:
                                sample.reward = 0
                        result = group
                        print(
                            f"[Worker {self.worker_id}, task {task_id}] Group {group_id} aborted due to step lag",
                            flush=True,
                        )

                    async with self._completed_lock:
                        self._completed_groups.append((group_id, result))
                except Exception as e:
                    async with self._inflight_lock:
                        self._inflight.pop((task_id, group_id), None)
                    print(f"[Worker {self.worker_id}, task {task_id}] Group {group_id} failed: {e}")

        tasks = [asyncio.create_task(_process_loop(i)) for i in range(self.concurrency)]
        # DIAG: run() should never return. If gather() ever returns or raises, the actor
        # process terminates and orphans its Tau2EnvWorker pool ("owner crashed"). Capture
        # exactly which _process_loop task died and why (incl. BaseException like
        # CancelledError, which the inner `except Exception` does NOT catch).
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(
            f"[DIAG][Worker {self.worker_id}] run() gather RETURNED — actor will now exit. "
            f"This orphans the env-worker pool. Per-task outcomes:",
            flush=True,
        )
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                tb = "".join(traceback.format_exception(type(r), r, r.__traceback__))
                print(f"[DIAG][Worker {self.worker_id}] task {i} raised {type(r).__name__}: {r}\n{tb}", flush=True)
            else:
                print(f"[DIAG][Worker {self.worker_id}] task {i} returned {r!r}", flush=True)
        with contextlib.suppress(Exception):
            faulthandler.dump_traceback(all_threads=True)

    async def drain_completed(self) -> list[tuple]:
        """Return and clear all groups completed since the last call (collector polls this)."""
        async with self._completed_lock:
            out = self._completed_groups
            self._completed_groups = []
        return out


# ---------------------------------------------------------------------------
# Manager — lives in the RolloutManager process, starts multiple workers.
# ---------------------------------------------------------------------------


def get_global_worker(args, data_buffer):
    """Get or create global worker."""
    global _global_worker, _semaphore_initialized
    with _worker_lock:
        # Initialize global semaphores once
        if not _semaphore_initialized:
            print("Initializing global semaphores...")
            initialize_semaphore("container_launch")
            initialize_semaphore("container_running")
            _semaphore_initialized = True

        if _global_worker is None or not _global_worker.feeder_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutManager(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop global worker."""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutManager:
    """
    Orchestrator that feeds data to Ray actor workers via Ray Queues.

    - A feeder thread pulls groups from data_buffer (local object in
      RolloutManager) and puts them into the input Ray Queue.
    - N AsyncRolloutWorkerActor(s) each run ``concurrency_per_worker``
      async tasks that pull from the input queue, process, and push
      results to the output Ray Queue.
    - The collector (generate_rollout_async) drains the output queue.
    """

    def __init__(self, args, data_buffer, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer
        self.running = True
        self.feeder_thread = None
        self.state = GenerateState(args)

        num_workers = int(os.environ.get("NUM_ASYNC_ROLLOUT_WORKERS", "16"))
        concurrency_per_worker = max(1, args.over_sampling_batch_size // num_workers)

        # Ray queues — input is bounded to provide backpressure to the feeder.
        # IMPORTANT: both queues use maxsize>0 so put_async/get_async route through
        # the _QueueActor's *async* put/get (run on its event-loop thread). With
        # maxsize=0, put_async falls back to put_nowait and the collector uses
        # get_nowait — both *sync* methods that run in a thread-pool thread on the
        # async actor, concurrently mutating the non-thread-safe asyncio.Queue.
        # Under high concurrent puts that desyncs the queue so qsize() counts items
        # get_nowait() can never retrieve (groups silently never collected).
        self.input_queue = RayQueue(maxsize=args.over_sampling_batch_size)
        self.output_queue = RayQueue(maxsize=1_000_000)

        # Create and start worker actors — spread uniformly across Ray nodes
        # so each actor tends to run on the same node as one BrowserWorker,
        # enabling local (same-node) browser dispatch in the generate function.
        self.workers = []
        worker_cls = AsyncRolloutWorkerActor.options(
            scheduling_strategy="SPREAD",
        )
        for i in range(num_workers):
            worker = worker_cls.remote(
                i,
                args,
                self.input_queue,
                self.output_queue,
                concurrency_per_worker,
            )
            worker.run.remote()  # fire-and-forget — runs continuously
            self.workers.append(worker)

        self.num_workers = num_workers
        print(f"Created {num_workers} AsyncRolloutWorkerActor(s), concurrency={concurrency_per_worker} each")

    # -- feeder thread --------------------------------------------------------

    def _feeder_thread_entry(self):
        """Thread entry point — run the async feeder loop."""
        asyncio.run(self._feeder_loop())

    async def _feeder_loop(self):
        """Pull from data_buffer (local) and push to input Ray Queue."""
        # Define wandb metrics in the manager process (runs once)
        from shared.rollout_log import _ensure_wandb_metrics

        _ensure_wandb_metrics()

        # Start log_counter_loop in the manager's event loop
        enable_global_counter()
        router_url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
        self._counter_log_task = asyncio.create_task(log_counter_loop(interval=10.0, router_url=router_url))

        sampling_params = self.state.sampling_params.copy()
        group_id_counter = 0
        while self.running:
            try:
                # data_buffer is a local object — offload sync call
                samples = await asyncio.to_thread(self.data_buffer.get_samples, 1)
                if not samples:
                    await asyncio.sleep(0.1)
                    continue

                for group in samples:
                    await self.input_queue.put_async(
                        (group_id_counter, group, sampling_params),
                    )
                    group_id_counter += 1
                    break  # get_samples(1) returns at most 1 group

            except Exception as e:
                print(f"Feeder error: {e}")
                await asyncio.sleep(1)

        print("Feeder loop stopped")

    # -- public API -----------------------------------------------------------

    def start(self):
        """Start the feeder thread."""
        from shared.rollout_log import _ensure_wandb_metrics

        _ensure_wandb_metrics()

        if self.feeder_thread is None or not self.feeder_thread.is_alive():
            self.feeder_thread = threading.Thread(target=self._feeder_thread_entry, daemon=True)
            self.feeder_thread.start()
            print("Started feeder thread for async rollout workers")

    def stop(self):
        """Stop feeder thread and kill worker actors."""
        self.running = False
        if self.feeder_thread and self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=10)
        for worker in self.workers:
            with contextlib.suppress(Exception):
                ray.kill(worker)
        print("Stopped async rollout workers")

    async def get_completed_groups(self, max_groups: int = 0) -> list[tuple]:
        """Drain completed groups by polling each worker actor directly.

        Replaces the old output Ray Queue (whose asyncio.Queue desynced under load:
        qsize() counted items get()/get_nowait() could never retrieve). Each worker
        accumulates results locally; we poll them all and flatten. ``max_groups`` is
        accepted for API compatibility but we always drain everything available.
        """
        results = await asyncio.gather(*[w.drain_completed.remote() for w in self.workers], return_exceptions=True)
        completed: list[tuple] = []
        for r in results:
            if isinstance(r, list):
                completed.extend(r)
            elif isinstance(r, BaseException):
                logging.getLogger(__name__).debug("[get_completed_groups] worker drain failed: %s", r)
        return completed

    def get_queue_size(self) -> int:
        """Deprecated: results no longer use a Ray Queue. Kept for log compatibility."""
        return 0


# ---------------------------------------------------------------------------
# Collector — drains output queue until we have enough for a training batch.
# ---------------------------------------------------------------------------


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    """Collect completed rollout results from the global async worker."""
    global current_rollout_step
    current_rollout_step = rollout_id
    _get_rollout_step_holder().set.remote(rollout_id)

    assert args.rollout_global_dataset

    worker = get_global_worker(args, data_buffer)
    target_data_size = args.rollout_batch_size

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    exclude_reasons_str = os.environ.get("EXCLUDE_ON_DROP_REASONS", "")
    exclude_on_drop_reasons = {r.strip() for r in exclude_reasons_str.split(",") if r.strip()}
    if exclude_on_drop_reasons:
        print(f"Will exclude samples on drop reasons: {exclude_on_drop_reasons}")

    data = []
    completed_groups = {}
    do_print = True

    print(f"Starting async rollout generation for {target_data_size} groups")
    print(f"Global worker queue size: {worker.get_queue_size()}")

    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0
    origin_key_to_drop = set()

    while len(data) < target_data_size:
        completed = await worker.get_completed_groups()
        if completed:
            print(
                f"[DIAG-COLLECTOR] pulled {len(completed)} group(s); qsize_now={worker.get_queue_size()} "
                f"collected={len(data)}/{target_data_size}",
                flush=True,
            )

        made_progress = False
        for group_id, group in completed:
            completed_groups[group_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        processed_any = False
        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)

            try:
                _diag_statuses = [
                    getattr(getattr(s, "status", None), "name", str(getattr(s, "status", None))) for s in group
                ]
            except Exception as _diag_e:
                _diag_statuses = f"<not-iterable {type(group).__name__}: {_diag_e}>"
            print(
                f"[DIAG-COLLECTOR] processing group {group_id}: type={type(group).__name__} "
                f"n={len(group) if hasattr(group, '__len__') else '?'} statuses={_diag_statuses}",
                flush=True,
            )

            try:
                any_aborted = any(sample.status == Sample.Status.ABORTED for sample in group)
            except Exception:
                any_aborted = False

            if any_aborted:
                abort_reason = "unknown"
                for sample in group:
                    if sample.status == Sample.Status.ABORTED:
                        abort_reason = sample.metadata.get("abort_reason", "unknown")
                        break
                metric_gatherer.on_aborted(reason=abort_reason)
                print(f"[DIAG-COLLECTOR] group {group_id} -> ABORTED (reason={abort_reason})", flush=True)
                continue

            metric_gatherer.on_generated()

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                print(
                    f"Dynamic filter dropped group {group_id}: {dynamic_filter_output.reason}",
                    flush=True,
                )
                if dynamic_filter_output.reason in exclude_on_drop_reasons:
                    origin_key = group[0].metadata.get(ORIGIN_SAMPLE_KEY)
                    origin_key_to_drop.add(origin_key)
                continue

            data.append(group)
            processed_any = True
            print(
                f"[DIAG-COLLECTOR] group {group_id} -> APPENDED (collected={len(data)}/{target_data_size})", flush=True
            )

        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        if not processed_any:
            # Polling cadence: get_completed_groups now fans out to all worker
            # actors, so poll a bit less aggressively than the old single-queue drain.
            await asyncio.sleep(0.2)

    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")
    if len(origin_key_to_drop) > 10:
        for origin_key in list(origin_key_to_drop)[10:]:
            print(f"Excluding sample {origin_key} from data buffer forever")
            data_buffer.exclude_samples([origin_key])

    if data:
        print(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    data = sorted(data, key=lambda group: group[0].index)

    # Stamp end_rollout_step on every sample and compute step lags
    step_lags = []
    for group in data:
        for sample in group:
            sample.metadata["end_rollout_step"] = rollout_id
            begin = sample.metadata.get("begin_rollout_step", rollout_id)
            step_lags.append(rollout_id - begin)

    metrics = metric_gatherer.collect()
    metrics["rollout/epoch_id"] = data_buffer.epoch_id
    metrics["rollout/avg_step_lag"] = sum(step_lags) / len(step_lags) if step_lags else 0.0
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    """Entry point for fully async rollout."""
    if evaluation:
        raise ValueError("Evaluation mode not supported in fully async rollout")

    completed_samples = run(generate_rollout_async(args, rollout_id, data_buffer))
    return completed_samples


atexit.register(stop_global_worker)
