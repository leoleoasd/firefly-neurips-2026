"""
Tau2-Bench Integration for Slime Training - MoE Version

This module provides the main interface for training MoE agents in tau2-bench environments.
Key differences from generate_with_tau2_gym.py:
- Uses token-in/token-out mode (input_ids instead of text)
- Collects routed_experts from each turn for routing replay
- Aggregates routed_experts across multi-turn interactions

Episode execution is distributed across Ray actors (Tau2EnvWorker) to avoid the
default asyncio ThreadPoolExecutor bottleneck that limits concurrent user-model
requests.  See tau2_env_workers.py for details.
"""

import logging
import traceback
from typing import Any

from slime.utils.types import Sample
from tau2_env_workers import _create_error_sample, get_tau2_env_pool

logger = logging.getLogger(__name__)


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for tau2-bench (MoE version).

    Dispatches the episode to a Tau2EnvWorker Ray actor for true parallelism.
    Each worker has its own process, event loop, and large thread pool, so
    env.step() / env.reset() calls (which block a thread while the orchestrator
    calls the user model) never starve.
    """
    assert not args.partial_rollout, "Partial rollout is not supported for tau2-bench."

    # Extract task info early for error handling
    task_index = 0
    prompt_text = ""
    try:
        if hasattr(sample, "metadata") and sample.metadata and "task" in sample.metadata:
            task_index = sample.metadata.get("index", 0)
        else:
            task_index = int(sample.prompt)
        prompt_text = str(sample.prompt) if hasattr(sample, "prompt") else ""
    except Exception:
        pass

    try:
        # Lazily initialise the pool on first call (blocks once for tokenizer
        # loading + endpoint resolution, then all subsequent calls are async).
        pool = get_tau2_env_pool()
        await pool.initialize(args)

        worker = pool.get_worker()
        result = await worker.run_episode.remote(sample, sampling_params)
        return result

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e!s}"
        logger.error(f"Task {task_index} failed with exception: {error_msg}\nTraceback: {traceback.format_exc()}")
        return _create_error_sample(task_index, error_msg, prompt_text)
