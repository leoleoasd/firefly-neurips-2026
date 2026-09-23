"""Custom rollout logging that includes per-sample timing metrics."""

import logging
from collections import defaultdict

import numpy as np
from slime.utils import logging_utils

import wandb

logger = logging.getLogger(__name__)

_wandb_metrics_defined = False


def _ensure_wandb_metrics():
    """Register custom metric prefixes with wandb so they use rollout/step as x-axis."""
    global _wandb_metrics_defined
    if _wandb_metrics_defined:
        return
    _wandb_metrics_defined = True
    try:
        wandb.define_metric("token_length/*", step_metric="rollout/step")
        wandb.define_metric("turns/*", step_metric="rollout/step")
        wandb.define_metric("rollout/dynamic_filter/*", step_metric="rollout/step")
        wandb.define_metric("rollout/aborted/*", step_metric="rollout/step")
        wandb.define_metric("tool_calls/*", step_metric="rollout/step")
        wandb.define_metric("counter/*", step_metric="counter/_elapsed_s")
        print("Wandb metrics defined")
        print("=" * 100)
    except Exception as e:
        logger.exception(f"Error defining wandb metrics: {e}")
        pass


# Token length fields to track
TOKEN_LENGTH_FIELDS = [
    "prompt_token_length",
    "user_token_length",
    "assistant_token_length",
    "tool_response_token_length",
]


def _compute_token_length_metrics(samples) -> dict[str, float]:
    """Compute max, mean, q25, q75 for each token length field across samples.

    Args:
        samples: List of samples from rollout

    Returns:
        Dict with metrics like "prompt_token_length_max", "prompt_token_length_mean", etc.
    """
    metrics = {}

    for field in TOKEN_LENGTH_FIELDS:
        values = []
        for sample in samples:
            if field in sample.metadata:
                values.append(sample.metadata[field])

        if values:
            arr = np.array(values)
            metrics[f"{field}_max"] = float(np.max(arr))
            metrics[f"{field}_mean"] = float(np.mean(arr))
            metrics[f"{field}_q25"] = float(np.percentile(arr, 25))
            metrics[f"{field}_q75"] = float(np.percentile(arr, 75))

    # Compute total token length (sum of all fields)
    total_values = []
    for sample in samples:
        total = 0
        for field in TOKEN_LENGTH_FIELDS:
            if field in sample.metadata:
                total += sample.metadata[field]
        total_values.append(total)

    if total_values:
        arr = np.array(total_values)
        metrics["total_token_length_max"] = float(np.max(arr))
        metrics["total_token_length_mean"] = float(np.mean(arr))
        metrics["total_token_length_q25"] = float(np.percentile(arr, 25))
        metrics["total_token_length_q75"] = float(np.percentile(arr, 75))

    return metrics


def _compute_tool_call_metrics(samples) -> dict[str, float]:
    """Compute tool call counts by match_type across all samples.

    Args:
        samples: List of samples from rollout

    Returns:
        Dict with metrics like "total", "exact", "generated", "no_data".
    """
    total = 0
    counts = {"exact": 0, "generated": 0, "no_data": 0}

    for sample in samples:
        if "tool_calls" not in sample.metadata:
            continue
        for tc in sample.metadata["tool_calls"]:
            total += 1
            if "match_type" in tc:
                match_type = tc["match_type"]
                if match_type in counts:
                    counts[match_type] += 1

    if total == 0:
        return {}

    metrics: dict[str, float] = {"total": float(total)}
    for match_type, count in counts.items():
        metrics[match_type] = float(count)
    return metrics


def _compute_turn_metrics(samples) -> dict[str, float]:
    """Compute mean, q25, q75, max for number of turns across samples.

    Args:
        samples: List of samples from rollout

    Returns:
        Dict with metrics like "num_turns_mean", "num_turns_q25", etc.
    """
    values = []
    for sample in samples:
        if "num_turns" in sample.metadata:
            values.append(sample.metadata["num_turns"])

    if not values:
        return {}

    arr = np.array(values)
    return {
        "num_turns_mean": float(np.mean(arr)),
        "num_turns_q25": float(np.percentile(arr, 25)),
        "num_turns_q75": float(np.percentile(arr, 75)),
        "num_turns_max": float(np.max(arr)),
    }


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Custom rollout logging that includes per-sample timing metrics.

    Args:
        rollout_id: Current rollout iteration ID
        args: Training arguments
        samples: List of samples from rollout
        rollout_extra_metrics: Additional metrics from rollout
        rollout_time: Total time for rollout

    Returns:
        False to continue with default logging
    """
    # Aggregate per-sample timing from sample.metadata["timing"]
    # (written by generate() in each worker process)
    timing_accum: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        timing = sample.metadata.get("timing")
        if timing:
            for name, elapsed in timing.items():
                timing_accum[name].append(elapsed)

    timer_dict = {}
    for name, values in timing_accum.items():
        arr = np.array(values)
        timer_dict[f"{name}_mean"] = float(np.mean(arr))
        timer_dict[f"{name}_max"] = float(np.max(arr))
    logger.info(f"Timing metrics for rollout {rollout_id}: {timer_dict}")

    # Compute token length metrics
    token_length_metrics = _compute_token_length_metrics(samples)
    logger.info(f"Token length metrics for rollout {rollout_id}: {token_length_metrics}")

    # Compute turn metrics
    turn_metrics = _compute_turn_metrics(samples)
    logger.info(f"Turn metrics for rollout {rollout_id}: {turn_metrics}")

    # Compute tool call metrics
    tool_call_metrics = _compute_tool_call_metrics(samples)
    logger.info(f"Tool call metrics for rollout {rollout_id}: {tool_call_metrics}")

    # Build log dict
    log_dict = {}

    if timer_dict:
        # Log timer metrics with perf/ prefix
        for k, v in timer_dict.items():
            log_dict[f"perf/{k}"] = v

    if token_length_metrics:
        # Log token length metrics with token_length/ prefix
        for k, v in token_length_metrics.items():
            log_dict[f"token_length/{k}"] = v

    if turn_metrics:
        # Log turn metrics with turns/ prefix
        for k, v in turn_metrics.items():
            log_dict[f"turns/{k}"] = v

    if tool_call_metrics:
        # Log tool call metrics with tool_calls/ prefix
        for k, v in tool_call_metrics.items():
            log_dict[f"tool_calls/{k}"] = v
    print("log_dict", log_dict)
    _ensure_wandb_metrics()
    log_dict["rollout/step"] = rollout_id
    logging_utils.log(args, log_dict, step_key="rollout/step")

    # Return False to also run default logging
    return False
