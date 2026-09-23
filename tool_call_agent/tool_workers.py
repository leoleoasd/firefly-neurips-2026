"""
Tool Worker Pool for distributed tool execution.

Each Ray node gets one ToolWorker actor to execute tool calls in parallel
without blocking the main generate event loop.

Uses ToolCallSimulator for RAG-based tool call simulation:
- Exact match: returns cached result if input matches historical data
- RAG generation: finds similar inputs and uses judge LLM to generate result

Required environment variables:
    TRAJECTORY_PATH  - path to data batches dir (e.g. /tmp/instance_storage/data_batches/0)
    SERVERS_PATH     - path to mcp_servers_joined.json
    JUDGE_MODEL      - model name for hosted_vllm (e.g. /tmp/instance_storage/qwen3-30B-A3B)
"""

import logging
import os
from pathlib import Path

import ray
import weave
from batch_inference import BatchInferenceClient

from shared.sglang_registry import get_or_create_registry
from tool_call_agent.tool_call_simulator import ToolCallSimulator

logger = logging.getLogger(__name__)


def _normalize_tool_name(tool_id: str) -> str:
    """Convert tool_id to normalized tool name.

    Matches format from convert_mcp_to_training.py and evaluator.py:
    - Format: {normalized_server}__{tool_name}
    - Server: normalized (@ removed, / and - replaced with _)
    - Tool name: kept as-is
    """
    if "::" in tool_id:
        server_part, tool_name = tool_id.split("::", 1)
    else:
        server_part = ""
        tool_name = tool_id

    normalized_server = server_part.replace("@", "").replace("/", "_").replace("-", "_")
    return f"{normalized_server}__{tool_name}"


@ray.remote(num_cpus=1)
class ToolWorker:
    """
    Ray actor for executing tool calls on a specific node.

    Each node has one ToolWorker to handle tool execution,
    keeping the heavy work off the main RolloutManager.
    Call init() after construction to load the ToolCallSimulator.
    """

    def __init__(self, node_id: str, node_index: int):
        self.node_id = node_id
        self.node_index = node_index
        self._simulator: ToolCallSimulator | None = None
        # Reverse mapping: normalized_name -> original tool_id
        self._name_to_tool_id: dict[str, str] = {}
        self._init_config()
        logger.info(f"ToolWorker created on node {node_id} (index={node_index})")

    def _init_config(self):
        """Read configuration from environment variables."""
        # Required env vars (fail fast if missing)
        self._trajectory_path = os.environ["TRAJECTORY_PATH"]
        self._servers_path = os.environ["SERVERS_PATH"]
        self._judge_model = os.environ["JUDGE_MODEL"]

        # Derive batch_id from trajectory path (e.g. /tmp/.../data_batches/0 -> "0")
        self._batch_id = Path(self._trajectory_path).name

        self._registry_name = "sglang_registry"

    async def init(self):
        """Initialize the ToolCallSimulator. Must be called after construction."""
        registry = get_or_create_registry(self._registry_name)

        # Try to get rm_router first (handles load balancing)
        router_urls = await registry.get_all.remote("rm_router")
        if router_urls:
            api_base = f"{router_urls[0]}/v1"
            logger.info(f"ToolWorker {self.node_id} (index={self.node_index}): using RM router (api_base={api_base})")
        else:
            # Fallback to direct rm_worker access with uniform distribution
            all_urls = await registry.get_all.remote("rm_worker")
            if not all_urls:
                raise RuntimeError(f"No RM router or workers found in registry (name={self._registry_name})")

            # Uniformly pick an RM based on node_index (guaranteed uniform distribution)
            api_base = all_urls[self.node_index % len(all_urls)]
            api_base = f"{api_base}/v1"

            logger.info(
                f"ToolWorker {self.node_id} (index={self.node_index}): using RM worker directly "
                f"(api_base={api_base}, RM {self.node_index % len(all_urls) + 1}/{len(all_urls)})"
            )

        llm_client = BatchInferenceClient()
        await llm_client.setup(
            model_id="not-used",
            litellm_mode=True,
            litellm_model=f"hosted_vllm/{self._judge_model}",
            litellm_api_base=api_base,
        )

        # Create and load the simulator
        self._simulator = await ToolCallSimulator.create(
            data_batches_path=self._trajectory_path,
            servers_path=self._servers_path,
            llm_client=llm_client,
        )

        # Build reverse mapping: normalized_name -> tool_id
        self._build_name_mapping()

        stats = self._simulator.get_stats()
        logger.info(
            f"ToolWorker {self.node_id}: simulator loaded "
            f"(tools={stats['total_tools']}, calls={stats['total_calls']}, "
            f"unique_inputs={stats['unique_inputs']})"
        )

    def _build_name_mapping(self):
        """Build mapping from normalized tool names to original tool_ids."""
        # Cover both tool_info (from servers JSON) and tool_calls (from trajectories)
        all_tool_ids = set(self._simulator.tool_info.keys()) | set(self._simulator.tool_calls.keys())
        for tool_id in all_tool_ids:
            normalized = _normalize_tool_name(tool_id)
            self._name_to_tool_id[normalized] = tool_id

    async def execute(self, tool_name: str, tool_args: dict, task_id: str = "") -> tuple[str, str, str]:
        """
        Execute a tool call using the ToolCallSimulator.

        Args:
            tool_name: Normalized tool name from model output
            tool_args: Arguments for the tool
            task_id: Task ID for ground truth prioritization

        Returns:
            Tuple of (tool output string, node_id, match_type)
        """
        # Map normalized name back to original tool_id.
        # .get() is intentional: model output may contain hallucinated tool names
        # that don't exist in the mapping; fall back to raw name so the simulator
        # returns a proper "no_data" result instead of crashing.
        tool_id = self._name_to_tool_id.get(tool_name, tool_name)

        result = await self._simulator.call_tool(tool_id, tool_args, batch_id=self._batch_id, task_id=task_id)
        if len(result.tool_output) > 20 * 2024 * 3:
            # tool response too long, return error
            return "error", self.node_id, result.match_type
        return result.tool_output, self.node_id, result.match_type


class ToolWorkerPool:
    """
    Pool of ToolWorkers distributed across Ray nodes.

    Creates one worker per node and provides round-robin access.
    """

    def __init__(self):
        self.workers: list[ray.actor.ActorHandle] = []
        self.node_ids: list[str] = []
        self._idx = 0
        self._initialized = False

    def initialize(self):
        """Initialize workers on all available nodes."""
        if self._initialized:
            return

        # Get all alive nodes
        nodes = [n for n in ray.nodes() if n.get("Alive")]
        if not nodes:
            raise RuntimeError("No alive Ray nodes found")

        logger.info(f"Initializing ToolWorkerPool with {len(nodes)} nodes")

        init_refs = []
        for node_index, node in enumerate(nodes):
            node_id = node["NodeID"]

            # Create worker with node affinity
            worker = ToolWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                )
            ).remote(node_id, node_index)

            self.workers.append(worker)
            self.node_ids.append(node_id)
            init_refs.append(worker.init.remote())

        # Wait for all workers to finish loading their simulators
        ray.get(init_refs)

        self._initialized = True
        logger.info(f"ToolWorkerPool initialized with {len(self.workers)} workers")

    def get_worker(self) -> ray.actor.ActorHandle:
        """Get next worker using round-robin."""
        if not self._initialized:
            self.initialize()

        worker = self.workers[self._idx]
        self._idx = (self._idx + 1) % len(self.workers)
        return worker

    @property
    def num_workers(self) -> int:
        return len(self.workers)


# Global singleton pool
_pool: ToolWorkerPool | None = None


def get_tool_pool() -> ToolWorkerPool:
    """Get the global ToolWorkerPool singleton."""
    global _pool
    if _pool is None:
        _pool = ToolWorkerPool()
        _pool.initialize()
    return _pool


def reset_tool_pool():
    """Reset the global pool (useful for testing)."""
    global _pool
    _pool = None


# ── Public API ───────────────────────────────────────────────────────────────


@weave.op()
async def perform_tool_call(tool_name: str, tool_args: dict, task_id: str = "") -> tuple[str, str]:
    """
    Execute a tool call by dispatching to a ToolWorker in the pool.

    Lazily initializes the pool on first call (one worker per Ray node).
    Uses round-robin to distribute load across workers.

    Args:
        tool_name: The name of the tool to call.
        tool_args: The arguments to pass to the tool.
        task_id: Task ID for ground truth prioritization.

    Returns:
        Tuple of (JSON string containing the tool result, match_type).
    """
    pool = get_tool_pool()
    worker = pool.get_worker()
    result_ref = worker.execute.remote(tool_name, tool_args, task_id)
    result, _node_id, match_type = await result_ref
    return result, match_type
