"""
Per-rollout timer: like slime's Timer, but indexed by (name, sample_id).
log_dict() returns both mean and max time across sample IDs for each name.

Uses contextvars to track the current sample_id, so callers don't need to pass it explicitly.
"""

import logging
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from time import time

from slime.utils.misc import SingletonMeta

__all__ = [
    "RolloutTimer",
    "current_sample_id",
    "get_sample_timers",
    "rollout_timer_total",
    "set_sample_id",
]

logger = logging.getLogger(__name__)

# ContextVar for tracking current sample_id (works correctly with asyncio)
_current_sample_id: ContextVar[int | None] = ContextVar("current_sample_id", default=None)


def set_sample_id(sample_id: int):
    """Set the current sample_id in the context. Returns a token for resetting."""
    return _current_sample_id.set(sample_id)


def current_sample_id() -> int | None:
    """Get the current sample_id from context."""
    return _current_sample_id.get()


def get_sample_timers(sample_id: int | None = None) -> dict[str, float]:
    """Get all timer values for a specific sample_id.

    Args:
        sample_id: Sample ID. If None, uses the current sample_id from contextvars.

    Returns:
        Dict mapping timer name to accumulated elapsed time for this sample.
    """
    if sample_id is None:
        sample_id = _current_sample_id.get()
    if sample_id is None:
        return {}

    result = {}
    timer = RolloutTimer()
    for name, sample_map in timer.timers.items():
        if sample_id in sample_map:
            result[name] = sample_map[sample_id]
    return result


class RolloutTimer(metaclass=SingletonMeta):
    def __init__(self):
        # {name: {sample_id: accumulated_elapsed}}
        self.timers: dict[str, dict[int, float]] = defaultdict(dict)

    def add(self, name: str, sample_id: int, elapsed_time: float):
        prev = self.timers[name].get(sample_id, 0)
        new_total = prev + elapsed_time
        self.timers[name][sample_id] = new_total
        logger.debug(
            f"RolloutTimer.add: name={name}, sample_id={sample_id}, "
            f"elapsed_time={elapsed_time:.6f}s, prev_total={prev:.6f}s, new_total={new_total:.6f}s"
        )

    def reset(self, name: str | None = None):
        if name is None:
            timer_count = sum(len(sample_map) for sample_map in self.timers.values())
            logger.debug(f"RolloutTimer.reset: resetting all timers (total timer entries: {timer_count})")
            self.timers = defaultdict(dict)
        elif name in self.timers:
            sample_count = len(self.timers[name])
            logger.debug(f"RolloutTimer.reset: resetting timer name={name} (sample_count={sample_count})")
            del self.timers[name]
        else:
            logger.debug(f"RolloutTimer.reset: timer name={name} not found, nothing to reset")

    def log_dict(self) -> dict[str, float]:
        """Return {name_mean: mean_elapsed, name_max: max_elapsed} for each timer name."""
        result = {}
        for name, sample_map in self.timers.items():
            values = list(sample_map.values())
            if values:
                mean_val = sum(values) / len(values)
                max_val = max(values)
                result[f"{name}_mean"] = mean_val
                result[f"{name}_max"] = max_val
                logger.debug(
                    f"RolloutTimer.log_dict: name={name}, sample_count={len(values)}, "
                    f"mean={mean_val:.6f}s, max={max_val:.6f}s"
                )
        logger.debug(f"RolloutTimer.log_dict: returning {len(result)} metrics: {list(result.keys())}")
        return result

    @contextmanager
    def total_context(self, name: str):
        """Context manager that can be used multiple times sequentially (not concurrently).

        Tracks total accumulated time across all sequential uses of the same (name, sample_id).
        Each use measures its own elapsed time and adds it to the accumulated total.

        sample_id is read from contextvars (must be set via set_sample_id() before use).
        """
        sample_id = _current_sample_id.get()
        if sample_id is None:
            raise RuntimeError("No sample_id set in context. Use set_sample_id() before timing.")

        logger.debug(f"RolloutTimer.total_context: entering context for name={name}, sample_id={sample_id}")
        start = time()
        try:
            yield
        finally:
            elapsed_time = time() - start
            logger.debug(
                f"RolloutTimer.total_context: elapsed_time={elapsed_time:.6f}s for name={name}, sample_id={sample_id}"
            )
            self.add(name, sample_id, elapsed_time)
            logger.debug(f"RolloutTimer.total_context: exiting context for name={name}, sample_id={sample_id}")


def rollout_timer_total(name: str):
    """Context manager for timing a block, can be used multiple times sequentially.

    Tracks total accumulated time across all sequential uses of the same (name, sample_id).
    Each use measures its elapsed time and adds it to the accumulated total.

    sample_id is read from contextvars (must be set via set_sample_id() before use).

    Usage:
        set_sample_id(3)
        with rollout_timer_total("generate"):
            ...
    """
    return RolloutTimer().total_context(name)
