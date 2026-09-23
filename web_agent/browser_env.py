"""
Browser environment workers for web agent training.

Provides a pool of workers that manage browser environments for distributed
rollout. Each worker can handle multiple browser instances (one per sample).

Uses WebAgentEnv from rl_web_agent for browser interaction.
"""

import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import ray
from omegaconf import OmegaConf
from rl_web_agent.env import WebAgentEnv

logger = logging.getLogger(__name__)

# Enable debug logging for rl_web_agent
logging.getLogger("rl_web_agent").setLevel(logging.DEBUG)


# Retry policy for setup failures (InsufficientMemoryError, RuntimeError, etc.).
# On each retry we rebuild a fresh WebAgentEnv, which rolls a new container
# uuid (env.py:__init__) and proxy client_session_id — giving the scheduler
# a chance to place us on a different backend host with free memory. Backoff
# is exponential with full jitter, capped at _MAX_DELAY_SECONDS.
_SETUP_MAX_ATTEMPTS = 5
_SETUP_BASE_DELAY_SECONDS = 2.0
_SETUP_MAX_DELAY_SECONDS = 60.0


# Browser tool schema in OpenAI format
BROWSER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "step_browser",
        "description": "Execute an action in the web browser environment and get the resulting observation.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Type of action to perform",
                    "enum": [
                        "click",
                        "type",
                        "hover",
                        "select",
                        "clear",
                        "key_press",
                        "goto_url",
                        "back",
                        "forward",
                        "refresh",
                        "new_tab",
                        "switch_tab",
                        "close_tab",
                        "terminate",
                    ],
                },
                "target": {
                    "type": "string",
                    "description": "Semantic ID of the element to interact with",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type (for type action)",
                },
                "enter": {
                    "type": "boolean",
                    "description": "Whether to press Enter after typing",
                },
                "value": {
                    "type": "string",
                    "description": "Value to select (for select action)",
                },
                "key": {
                    "type": "string",
                    "description": "Key to press (for key_press action)",
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to",
                },
                "tab_id": {
                    "type": "integer",
                    "description": "Tab ID for tab operations",
                },
                "answer": {
                    "type": "string",
                    "description": "Final answer for terminate action",
                },
            },
            "required": ["action"],
        },
    },
}


def get_browser_tool_schema() -> dict:
    """Return the browser tool schema in OpenAI format."""
    return BROWSER_TOOL_SCHEMA


def format_llm_observation(observation: dict[str, Any]) -> str:
    """
    Format observation dict for LLM consumption.

    Matches rl_web_agent.utils.format_llm_observation exactly.
    """
    if observation is None:
        return "No observation data available"

    obs_parts = []

    # Basic page info from current tab
    active_tab = next((tab for tab in observation["tabs"] if tab["is_active"]), observation["tabs"][0])
    obs_parts.append(f"URL: {active_tab['url']}")
    obs_parts.append(f"Title: {active_tab['title']}")
    obs_parts.append("")

    # HTML
    obs_parts.append("HTML:")
    obs_parts.append(observation["html"])
    obs_parts.append("")

    # Clickable elements
    if observation.get("clickable_elements"):
        obs_parts.append(f"CLICKABLE ELEMENTS ({len(observation['clickable_elements'])})")
        obs_parts.append("-" * 40)
        for i, elem_id in enumerate(observation["clickable_elements"], 1):
            obs_parts.append(f"  {i:2d}. {elem_id}")
        obs_parts.append("")

    # Input elements
    if observation.get("input_elements"):
        obs_parts.append(f"INPUT ELEMENTS ({len(observation['input_elements'])})")
        obs_parts.append("-" * 40)
        for i, inp in enumerate(observation["input_elements"], 1):
            elem_id = inp["id"]
            elem_type = inp["type"]
            value = inp["value"]
            can_edit = inp["canEdit"]
            is_focused = inp["isFocused"]

            status_parts = []
            if is_focused:
                status_parts.append("focused")
            if not can_edit:
                status_parts.append("read-only")

            status = " " + " ".join(status_parts) if status_parts else ""

            obs_parts.append(f"  {i:2d}. {elem_id} [{elem_type}]{status}")
            if value:
                obs_parts.append(f"      Value: '{value}'")
        obs_parts.append("")

    # Select elements
    if observation.get("select_elements"):
        obs_parts.append(f"SELECT ELEMENTS ({len(observation['select_elements'])})")
        obs_parts.append("-" * 40)
        for i, select in enumerate(observation["select_elements"], 1):
            elem_id = select["id"]
            elem_value = select["value"]
            obs_parts.append(f"  {i:2d}. {elem_id} - {elem_value}")
        obs_parts.append("")

    # Tabs with URLs
    obs_parts.append(f"TABS ({len(observation['tabs'])})")
    obs_parts.append("-" * 40)
    for tab in observation["tabs"]:
        active = "ACTIVE" if tab["is_active"] else "inactive"
        tab_title = tab["title"]
        tab_url = tab["url"]
        tab_id = tab["id"]
        obs_parts.append(f"  {tab_id:2d}. {active} - {tab_title} - {tab_url}")
    obs_parts.append("")

    return "\n".join(obs_parts)


def load_env_config() -> Any:
    """
    Load the environment configuration.

    Uses base.yaml from web_agent/conf/ and allows overrides via
    environment variables.

    Returns:
        Environment config object with sites, accounts, evaluator_llm, etc.
    """
    # Load base config from our local copy
    config_path = Path(__file__).parent / "conf" / "base.yaml"
    config = OmegaConf.load(config_path)

    # Apply environment variable overrides
    if os.environ.get("INCUS_SERVER_URL"):
        config.environment.incus_server_url = os.environ["INCUS_SERVER_URL"]
    if os.environ.get("PROXY_SERVER"):
        config.environment.proxy.server = os.environ["PROXY_SERVER"]
    if os.environ.get("PROXY_ENABLED"):
        config.environment.proxy.enabled = os.environ["PROXY_ENABLED"].lower() == "true"
    if os.environ.get("BROWSER_HEADLESS"):
        config.environment.browser.launch_options.headless = os.environ["BROWSER_HEADLESS"].lower() == "true"

    return config.environment


@ray.remote(num_cpus=1)
class BrowserWorker:
    """
    Ray actor for managing browser environments on a specific node.

    Each worker can handle multiple browser instances (one per sample).
    Uses WebAgentEnv from rl_web_agent for browser interaction.
    """

    def __init__(self, node_id: str, node_index: int):
        self.node_id = node_id
        self.node_index = node_index
        self._instances: dict[str, WebAgentEnv] = {}
        self._env_config = None
        logger.info(f"BrowserWorker created on node {node_id} (index={node_index})")

    def _get_env_config(self):
        """Lazy load environment config."""
        if self._env_config is None:
            self._env_config = load_env_config()
        return self._env_config

    async def create_instance(self, instance_id: str, task_config: dict) -> str:
        """
        Create a browser environment instance for a sample.

        Args:
            instance_id: Unique identifier for this instance
            task_config: WebArena task configuration from metadata

        Returns:
            Initial observation as formatted string for LLM

        Retry behavior:
            If ``env.setup()`` fails for any reason (e.g.
            ``InsufficientMemoryError`` from the incus host, ``RuntimeError``
            from a container or browser crash, network timeouts, etc.), we
            tear down the partially-constructed env and build a completely new
            ``WebAgentEnv``.  Because ``WebAgentEnv.__init__`` generates a
            fresh ``uuid.uuid4()`` (and a fresh proxy ``client_session_id``)
            whenever the env config does not carry a ``uuid``, each retry gets
            a new container name and proxy session, giving the scheduler a
            chance to route us to a different backend host.
        """
        env_config = self._get_env_config()

        max_attempts = _SETUP_MAX_ATTEMPTS
        base_delay = _SETUP_BASE_DELAY_SECONDS
        max_delay = _SETUP_MAX_DELAY_SECONDS

        for attempt in range(1, max_attempts + 1):
            # Construct a fresh env per attempt so that each retry rolls a new
            # container uuid / proxy client_session_id.
            env = WebAgentEnv(env_config)

            # Setup environment with task config and get initial observation.
            # Only store in _instances after setup succeeds. If setup() fails
            # partway (e.g. containers launched but browser/login fails), we
            # must close the env here to release any partially-acquired
            # resources (containers, playwright refcount, browser process).
            try:
                obs_dict = await env.setup(task_config)
            except Exception as exc:
                await env.close()
                if attempt >= max_attempts:
                    logger.error(
                        f"create_instance {instance_id}: setup failed after "
                        f"{attempt} attempts ({type(exc).__name__}: {exc}); giving up"
                    )
                    raise
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                delay += random.uniform(0.0, delay)  # full jitter on top of backoff
                logger.warning(
                    f"create_instance {instance_id}: {type(exc).__name__} on "
                    f"attempt {attempt}/{max_attempts}; "
                    f"recreating env with new uuid after {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue

            self._instances[instance_id] = env
            formatted_obs = format_llm_observation(obs_dict)
            logger.info(
                f"Created browser instance {instance_id} on node {self.node_id} (attempt {attempt}/{max_attempts})"
            )
            return formatted_obs

        # Unreachable: loop either returns or raises.
        raise RuntimeError(f"create_instance {instance_id}: exited retry loop without result")

    async def step(self, instance_id: str, action: dict | str) -> tuple[str, bool, float]:
        """
        Execute an action in a browser instance.

        Args:
            instance_id: Instance identifier
            action: Action dict with 'action' key and parameters, or JSON string

        Returns:
            Tuple of (formatted_observation, terminated, score)
        """
        env = self._instances[instance_id]

        # Convert action to JSON string if it's a dict
        # WebAgentEnv expects JSON string like '{"action": "click", "target": "..."}'
        if isinstance(action, str):
            # If it's already a string, try to parse and re-serialize to ensure valid JSON
            try:
                parsed = json.loads(action)
                action_json = json.dumps(parsed)
            except json.JSONDecodeError:
                action_json = action
        else:
            action_json = json.dumps(action)

        logger.debug(f"Browser step action_json: {action_json}")
        obs_dict = await env.step(action_json)

        # Format observation for LLM
        formatted_obs = format_llm_observation(obs_dict)

        return formatted_obs, obs_dict["terminated"], obs_dict["score"]

    async def get_observation(self, instance_id: str) -> dict:
        """Get current observation dict for an instance."""
        env = self._instances[instance_id]
        return await env.observation()

    async def get_score(self, instance_id: str) -> float:
        """Get the final score for an instance."""
        env = self._instances[instance_id]
        obs = await env.observation()
        return obs["score"]

    async def get_model_answer(self, instance_id: str) -> str | None:
        """Get the model's answer from terminate action."""
        env = self._instances[instance_id]
        return env.model_answer

    async def release_instance(self, instance_id: str):
        """Release a browser instance. Always removes from _instances."""
        if instance_id in self._instances:
            env = self._instances.pop(instance_id)
            try:
                await env.close()
            except Exception as e:
                logger.error(f"Error closing browser instance {instance_id} on node {self.node_id}: {e}")
            logger.info(f"Released browser instance {instance_id} on node {self.node_id}")


class BrowserWorkerPool:
    """
    Pool of BrowserWorkers distributed across Ray nodes.

    Creates one worker per node and provides affinity-based access.
    """

    def __init__(self):
        self.workers: list[ray.actor.ActorHandle] = []
        self.node_ids: list[str] = []
        # Map node_id -> worker for O(1) local lookup
        self._node_to_worker: dict[str, ray.actor.ActorHandle] = {}
        self._initialized = False

    def initialize(self):
        """Initialize workers on all available nodes."""
        if self._initialized:
            return

        # Get all alive nodes
        nodes = [n for n in ray.nodes() if n.get("Alive")]
        if not nodes:
            raise RuntimeError("No alive Ray nodes found")

        logger.info(f"Initializing BrowserWorkerPool with {len(nodes)} nodes")

        for node_index, node in enumerate(nodes):
            node_id = node["NodeID"]

            # Create worker with node affinity
            worker = BrowserWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                )
            ).remote(node_id, node_index)

            self.workers.append(worker)
            self.node_ids.append(node_id)
            self._node_to_worker[node_id] = worker

        self._initialized = True
        logger.info(f"BrowserWorkerPool initialized with {len(self.workers)} workers")

    def get_worker_for_sample(self, sample_index: int) -> ray.actor.ActorHandle:
        """Get worker for a sample using round-robin distribution."""
        if not self._initialized:
            self.initialize()

        return self.workers[sample_index % len(self.workers)]

    def get_local_worker(self, sample_index: int = 0) -> ray.actor.ActorHandle:
        """Return the BrowserWorker on the current Ray node.

        Falls back to round-robin by ``sample_index`` if no worker is
        registered for this node (e.g. the caller isn't running on a
        node where we spawned a BrowserWorker).
        """
        if not self._initialized:
            self.initialize()

        node_id = ray.get_runtime_context().get_node_id()
        worker = self._node_to_worker.get(node_id)
        if worker is not None:
            return worker

        logger.warning(
            "No local BrowserWorker on node %s; falling back to round-robin (sample_index=%d)",
            node_id,
            sample_index,
        )
        return self.workers[sample_index % len(self.workers)]

    @property
    def num_workers(self) -> int:
        return len(self.workers)


# Global singleton pool
_pool: BrowserWorkerPool | None = None


def get_browser_pool() -> BrowserWorkerPool:
    """Get the global BrowserWorkerPool singleton."""
    global _pool
    if _pool is None:
        _pool = BrowserWorkerPool()
        _pool.initialize()
    return _pool


def reset_browser_pool():
    """Reset the global pool (useful for testing)."""
    global _pool
    _pool = None
