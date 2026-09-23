"""
Tau2 Environment Worker Pool for distributed episode execution.

Creates Ray actors to run tau2-bench environment episodes in parallel,
eliminating the default thread pool bottleneck in asyncio.run_in_executor calls.

Each Tau2EnvWorker actor:
- Runs in its own process with a large ThreadPoolExecutor (no bottleneck on
  env.step/env.reset which block a thread while the orchestrator calls the
  user model synchronously via litellm)
- Holds its own tokenizer / GenerateState (no shared state)
- Creates a fresh AgentGymEnv per episode

Architecture:
    generate()  --->  Tau2EnvWorkerPool.get_worker()  (round-robin)
                          |
                          v
                    Tau2EnvWorker.run_episode()   (Ray async actor)
                          |
                          +-- AgentGymEnv (env.reset / env.step)
                          +-- run_tau2_env_loop_async_moe  (agent model + env)
                          +-- ThreadPoolExecutor(256)  for blocking env calls
"""

import asyncio
import json
import logging
import os
import sys
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import loguru
import ray
from agent_moe import run_tau2_env_loop_async_moe
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import init_http_client
from slime.utils.types import Sample
from tau2.data_model.tasks import RewardType, Task
from tau2.gym.gym_agent import AgentGymEnv
from tau2.registry import registry

from shared.global_counter import get_global_counter
from shared.rollout_timer import get_sample_timers, set_sample_id
from shared.sglang_registry import get_or_create_registry

try:
    import weave

    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    weave = None

logger = logging.getLogger(__name__)

_PAD_TOKEN_ID = 0

# ── Per-process task registration ────────────────────────────────────────────

_tasks_registered = False
_custom_tasks_cache: dict[str, list[Task]] = {}
_tau2_domain: str = ""


def _get_tau2_domain() -> str:
    logger.info(f"TAU2_DOMAIN: {os.environ.get('TAU2_DOMAIN')}")
    return os.environ.get("TAU2_DOMAIN", "telecom")


def _register_tasks():
    global _tasks_registered, _tau2_domain
    if _tasks_registered:
        return
    _tau2_domain = _get_tau2_domain()
    logger.info(f"Registering tasks for domain: {_tau2_domain}")
    registry.register_tasks(_get_custom_tasks, name=_tau2_domain)
    logger.info(f"Tasks registered for domain: {_tau2_domain}")
    _tasks_registered = True


def _get_custom_tasks(split: str | None = None) -> list[Task]:
    domain = _tau2_domain or _get_tau2_domain()
    cache_key = f"{domain}_train+test"
    if cache_key in _custom_tasks_cache:
        return _custom_tasks_cache[cache_key]

    # Data dir: prefer TAU2_DATA_DIR env var, fall back to script-relative data/
    data_dir = os.environ.get(
        "TAU2_DATA_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    )

    tasks: list[Task] = []
    if domain == "telecom":
        data_paths = [
            os.path.join(data_dir, f"{domain}_train_tasks.jsonl"),
            os.path.join(data_dir, f"{domain}_test_tasks.jsonl"),
        ]
    else:
        data_paths = [os.path.join(data_dir, f"{domain}_train_tasks.jsonl")]

    for data_path in data_paths:
        if not os.path.exists(data_path):
            continue
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                task_dict = data["metadata"]["task"]
                # tau2-bench datasets have used both:
                # - evaluation_criteria as a JSON-encoded string
                # - evaluation_criteria as an already-parsed dict (newer)
                eval_criteria = task_dict["evaluation_criteria"]
                if isinstance(eval_criteria, str):
                    task_dict["evaluation_criteria"] = json.loads(eval_criteria)

                task = Task.model_validate(task_dict)
                if task.evaluation_criteria is not None and domain == "telecom":
                    if (
                        not task.evaluation_criteria.reward_basis
                        or RewardType.ENV_ASSERTION in task.evaluation_criteria.reward_basis
                    ):
                        task.evaluation_criteria.reward_basis = [RewardType.ENV_ASSERTION]
                else:
                    task.evaluation_criteria.reward_basis = [
                        RewardType.DB,
                        RewardType.COMMUNICATE,
                    ]
                tasks.append(task)

    _custom_tasks_cache[cache_key] = tasks
    return tasks


def _create_error_sample(task_index: int, error_msg: str, prompt: str = "") -> Sample:
    return Sample(
        index=task_index,
        prompt=prompt,
        tokens=[_PAD_TOKEN_ID],
        rollout_log_probs=[],
        response="",
        reward=0.0,
        loss_mask=[],
        status=Sample.Status.ABORTED,
        metadata={"error": error_msg, "task_id": task_index},
        response_length=0,
    )


# ── Ray Actor ────────────────────────────────────────────────────────────────

DEFAULT_WORKERS_PER_NODE = 32
DEFAULT_THREAD_POOL_SIZE = 256


@ray.remote(num_cpus=1)
class Tau2EnvWorker:
    """
    Ray actor for running tau2-bench episodes.

    Each worker runs in its own process with:
    - Its own tokenizer and GenerateState
    - A large ThreadPoolExecutor so env.step()/env.reset() never starve

    Call init() after construction to load tokenizer and resolve endpoints.
    """

    def __init__(self, worker_id: int, node_id: str):
        self.worker_id = worker_id
        self.node_id = node_id
        self._tag = f"[W{worker_id}@{node_id[:8]}]"
        self._state: GenerateState | None = None
        self._args: Any = None
        self._sglang_url: str = ""
        self._user_llm: str = ""
        self._user_llm_args: dict[str, Any] = {}
        self._return_routed_experts: bool = False
        self._max_steps: int = 100
        self._weave_initialized: bool = False
        self._active_episodes: int = 0
        self._total_episodes: int = 0
        self._counter = None
        self._initialized: bool = False
        logger.info(f"{self._tag} __init__ done")

    # ── Initialisation ───────────────────────────────────────────────────

    async def init(self, args: Any):
        """Initialize worker state. Must be called after construction."""
        # init logging basic config for this new process
        logging.basicConfig(
            level=logging.WARNING,
            format="[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger.info(f"{self._tag} init called")

        loguru.logger.remove()
        loguru.logger.add(sys.stderr, level="WARNING")

        loop = asyncio.get_event_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_SIZE))

        self._args = args
        self._state = GenerateState(args)

        _register_tasks()

        self._sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        self._return_routed_experts = getattr(args, "use_rollout_routing_replay", False)
        self._max_steps = int(os.environ.get("TAU2_MAX_TURNS", "100"))

        await self._resolve_user_sim()
        self._init_wandb()
        init_http_client(args)

        self._counter = get_global_counter()
        counts = ray.get(self._counter.inc.remote("active_workers"))
        self._initialized = True
        logger.info(
            f"{self._tag} init done: sglang_url={self._sglang_url}, user_llm={self._user_llm}, return_routed_experts={self._return_routed_experts}, max_steps={self._max_steps}, global_counts={counts}"
        )

    async def _resolve_user_sim(self):
        """Resolve user simulator endpoint from env vars / sglang registry.

        Priority:
        1. USER_SIM_MODEL  -> local sglang-served model via rm_router
        2. AZURE_USER_SIM_MODEL -> Azure OpenAI (e.g. "gpt-4o")
           Requires AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION env vars.
        3. BEDROCK_USER_SIM_MODEL -> AWS Bedrock (e.g. Claude Sonnet via an
           inference-profile id or a bedrock/converse/<model> litellm string).

        Raises RuntimeError if none of the three env vars is set.
        """
        logger.info(f"{self._tag} _resolve_user_sim called")
        if not (
            os.environ.get("USER_SIM_MODEL")
            or os.environ.get("AZURE_USER_SIM_MODEL")
            or os.environ.get("BEDROCK_USER_SIM_MODEL")
        ):
            raise RuntimeError(
                "No user simulator model configured. Set exactly one of: "
                "USER_SIM_MODEL (local sglang model path served via rm_router), "
                "AZURE_USER_SIM_MODEL (Azure OpenAI deployment name), "
                "BEDROCK_USER_SIM_MODEL (AWS Bedrock model / inference-profile id)."
            )

        if os.environ.get("USER_SIM_MODEL"):
            registry_actor = get_or_create_registry("sglang_registry")
            url = await registry_actor.get_one.remote("rm_router")
            if url is None:
                raise RuntimeError("No sglang rm server found in registry (key='rm_router').")
            user_sim_model = os.environ["USER_SIM_MODEL"]
            self._user_llm = f"hosted_vllm/{user_sim_model}"
            self._user_llm_args = {
                "temperature": 0.7,
                "api_base": f"{url}/v1",
            }
            logger.info(
                f"{self._tag} _resolve_user_sim done: sglang model={self._user_llm}, api_base={self._user_llm_args['api_base']}"
            )
        elif os.environ.get("AZURE_USER_SIM_MODEL"):
            # Azure OpenAI path: model string format is "azure/<deployment_name>"
            azure_deployment = os.environ["AZURE_USER_SIM_MODEL"]
            self._user_llm = f"azure/{azure_deployment}"
            self._user_llm_args = {
                "temperature": 0.7,
                "reasoning_effort": "high",
            }
            # litellm reads AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
            # from env vars automatically, but we can also pass them explicitly
            if os.environ.get("AZURE_API_KEY"):
                self._user_llm_args["api_key"] = os.environ["AZURE_API_KEY"]
            if os.environ.get("AZURE_API_BASE"):
                self._user_llm_args["api_base"] = os.environ["AZURE_API_BASE"]
            if os.environ.get("AZURE_API_VERSION"):
                self._user_llm_args["api_version"] = os.environ["AZURE_API_VERSION"]
            logger.info(
                f"{self._tag} _resolve_user_sim done: azure model={self._user_llm}, "
                f"api_base={self._user_llm_args.get('api_base', 'from env')}"
            )
        else:
            self._user_llm = os.environ["BEDROCK_USER_SIM_MODEL"]
            self._user_llm_args = {}
            logger.info(f"{self._tag} _resolve_user_sim done: bedrock model={self._user_llm}")

    def _init_wandb(self):
        """Initialise wandb + weave in this env-worker process.

        Mirrors AsyncRolloutWorkerActor._init_wandb: attaches to the SHARED
        wandb run as a secondary process (init_wandb_secondary uses
        mode="shared", x_primary=False, id=wandb_run_id — so all env workers
        join the one run by id, they do NOT each create a run; it no-ops if
        wandb_run_id is unset). Without init_tracking, weave.init() here would
        log traces against an orphan run instead of the training run.
        """
        if self._weave_initialized:
            return
        if not getattr(self._args, "use_wandb", False):
            return
        try:
            from slime.utils.logging_utils import init_tracking

            from shared.rollout_log import _ensure_wandb_metrics

            init_tracking(self._args, primary=False)
            _ensure_wandb_metrics()
            if WEAVE_AVAILABLE:
                weave.init(self._args.wandb_project)
                logging.getLogger("weave.trace.weave_client").setLevel(logging.WARNING)
            self._weave_initialized = True
            logger.info(f"{self._tag} wandb/weave initialized (project={self._args.wandb_project})")
        except Exception as e:
            logger.exception(f"{self._tag} wandb/weave init failed: {e}")

    # ── Episode execution ────────────────────────────────────────────────

    async def run_episode(self, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
        """
        Run one complete tau2-bench episode.

        Creates a fresh AgentGymEnv, runs the full multi-turn interaction loop
        via run_tau2_env_loop_async_moe, and returns the completed Sample.
        """
        task_index = 0
        prompt_text = ""

        if not self._initialized:
            error_msg = (
                f"{self._tag} run_episode called but init() was never completed. "
                "Check init() logs for errors (e.g. _resolve_user_sim failure, "
                "missing env vars like AZURE_USER_SIM_MODEL / USER_SIM_MODEL)."
            )
            logger.error(error_msg)
            return _create_error_sample(0, error_msg)

        try:
            if hasattr(sample, "metadata") and sample.metadata and "task" in sample.metadata:
                task_index = sample.metadata.get("index", 0)
            else:
                task_index = int(sample.prompt)
            prompt_text = str(sample.prompt) if hasattr(sample, "prompt") else ""
        except Exception:
            pass

        self._active_episodes += 1
        self._total_episodes += 1
        counts = ray.get(self._counter.inc.remote("active_episodes"))
        logger.info(
            f"{self._tag} run_episode called: task={task_index} active={self._active_episodes} total={self._total_episodes} global_counts={counts}"
        )

        try:
            if hasattr(sample, "metadata") and sample.metadata and "task" in sample.metadata:
                task_index = sample.metadata.get("index", 0)
                task_id = sample.metadata["task"]["id"]
            else:
                task_index = int(sample.prompt)
                task_id = None

            env = AgentGymEnv(
                domain=_tau2_domain or _get_tau2_domain(),
                task_id=task_id,
                solo_mode=False,
                user_llm=self._user_llm,
                **({"user_llm_args": self._user_llm_args} if self._user_llm_args else {}),
            )

            sample.index = task_index
            set_sample_id(task_index)

            result_sample = await run_tau2_env_loop_async_moe(
                env=env,
                url=self._sglang_url,
                sampling_params=sampling_params,
                sample=sample,
                state=self._state,
                args=self._args,
                max_steps=self._max_steps,
                return_routed_experts=self._return_routed_experts,
            )

            result_sample.metadata["_timers"] = get_sample_timers(task_index)

            logger.info(
                f"{self._tag} run_episode done: task={task_index} reward={result_sample.reward:.3f} status={result_sample.status.name} active={self._active_episodes - 1} total={self._total_episodes}"
            )
            return result_sample

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e!s}"
            logger.error(
                f"{self._tag} run_episode failed: task={task_index} error={error_msg}\n{traceback.format_exc()}"
            )
            return _create_error_sample(task_index, error_msg, prompt_text)

        finally:
            self._active_episodes -= 1
            counts = ray.get(self._counter.dec.remote("active_episodes"))
            logger.info(f"{self._tag} run_episode finally: task={task_index} global_counts={counts}")


# ── Worker Pool ──────────────────────────────────────────────────────────────


class Tau2EnvWorkerPool:
    """
    Pool of Tau2EnvWorkers distributed across Ray nodes.

    Creates ``workers_per_node`` workers on every alive node and provides
    round-robin access.
    """

    def __init__(self):
        self.workers: list[ray.actor.ActorHandle] = []
        self.node_ids: list[str] = []
        self._idx = 0
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(
        self,
        args: Any,
        workers_per_node: int | None = None,
    ):
        """Create and initialise workers on all alive Ray nodes.

        Blocks until every worker has loaded its tokenizer and resolved
        the user-sim endpoint.  Safe to call concurrently — only the first
        caller does the actual work; others wait on the lock and return.

        ``workers_per_node`` defaults to env ``TAU2_WORKERS_PER_NODE`` (else
        ``DEFAULT_WORKERS_PER_NODE``). NOTE: this pool is a *per-process*
        singleton, so the total Tau2EnvWorker count is
        ``num_async_rollout_workers x num_nodes x workers_per_node``. Each
        worker is ``@ray.remote(num_cpus=1)``; keep the product under the
        cluster CPU budget or actors pile up in PENDING_CREATION and episodes
        starve.
        """
        if workers_per_node is None:
            workers_per_node = int(os.environ.get("TAU2_WORKERS_PER_NODE") or DEFAULT_WORKERS_PER_NODE)
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return

            # Node-local pool: this pool lives in the AsyncRolloutWorkerActor
            # process; create all its Tau2EnvWorkers ON THE SAME NODE so episode
            # dispatch (run_episode.remote — which ships the Sample with token_ids
            # /log_probs/routed_experts) stays node-local instead of crossing the
            # network. Each rollout-worker process owns its own node-local pool, so
            # total env workers = num_rollout_workers x workers_per_node (NOT
            # x num_nodes — that per-process x all-nodes product is what previously
            # over-subscribed the cluster CPUs and starved scheduling).
            my_node_id = ray.get_runtime_context().get_node_id()
            print(
                f"[Tau2EnvWorkerPool] Initialising {workers_per_node} node-local workers on {my_node_id[:8]}",
                flush=True,
            )
            logger.info(f"Initialising {workers_per_node} node-local Tau2EnvWorkers on node {my_node_id}")

            init_refs = []
            for this_worker_id in range(workers_per_node):
                worker = Tau2EnvWorker.options(
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=my_node_id,
                        soft=False,
                    ),
                ).remote(this_worker_id, my_node_id)

                self.workers.append(worker)
                self.node_ids.append(my_node_id)
                init_refs.append(worker.init.remote(args))
            logger.info(f"Initialising node-local Tau2EnvWorkerPool; {len(self.workers)=} on {my_node_id[:8]}")
            await asyncio.gather(*init_refs)

            self._initialized = True

            node_counts = Counter(self.node_ids)
            print(
                f"[Tau2EnvWorkerPool] READY: {len(self.workers)} workers across {len(node_counts)} nodes",
                flush=True,
            )
            for nid, count in node_counts.items():
                print(f"  node {nid[:8]}...: {count} workers", flush=True)
            logger.info(f"Tau2EnvWorkerPool ready: {len(self.workers)} workers")

    def get_worker(self) -> ray.actor.ActorHandle:
        """Get next worker using round-robin."""
        if not self._initialized:
            raise RuntimeError("Tau2EnvWorkerPool not initialised. Call initialize(args) first.")
        worker = self.workers[self._idx]
        self._idx = (self._idx + 1) % len(self.workers)
        return worker


# ── Global singleton ─────────────────────────────────────────────────────────

_pool: Tau2EnvWorkerPool | None = None
_pool_lock = threading.Lock()


def get_tau2_env_pool() -> Tau2EnvWorkerPool:
    """Get (or create) the global Tau2EnvWorkerPool singleton."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = Tau2EnvWorkerPool()
    return _pool
