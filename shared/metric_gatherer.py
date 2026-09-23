"""Extended MetricGatherer that also tracks drop ratio and aborted ratio."""

from collections import defaultdict


class MetricGatherer:
    """Collects dynamic filter metrics including drop counts, drop ratios, and aborted ratio.

    Drop ratio = dropped instances / total generated instances (kept + dropped + aborted).
    Aborted ratio = aborted instances / total generated instances.
    """

    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._abort_reason_count = defaultdict(lambda: 0)
        self._total_generated = 0
        self._total_aborted = 0

    def on_generated(self):
        """Call for every completed (non-aborted) instance that reaches the filter stage."""
        self._total_generated += 1

    def on_aborted(self, reason: str = "unknown"):
        """Call for every instance that was aborted."""
        self._total_aborted += 1
        self._abort_reason_count[reason] += 1

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }

        total = self._total_generated + self._total_aborted

        if total > 0:
            total_dropped = sum(self._dynamic_filter_drop_reason_count.values())
            metrics["rollout/dynamic_filter/drop_ratio"] = total_dropped / total
            for reason, count in self._dynamic_filter_drop_reason_count.items():
                metrics[f"rollout/dynamic_filter/drop_ratio_{reason}"] = count / total
            metrics["rollout/total_completed"] = total
            metrics["rollout/total_non_aborted"] = self._total_generated
            metrics["rollout/aborted_count"] = self._total_aborted
            metrics["rollout/aborted_ratio"] = self._total_aborted / total
            for reason, count in self._abort_reason_count.items():
                metrics[f"rollout/aborted/{reason}"] = count

        return metrics
