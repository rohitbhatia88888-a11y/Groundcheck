"""Ops metrics: latency percentiles and per-query cost. Kept separate from
retrieval/generation/behaviour metrics by design (see CLAUDE.md) — these
measure the system's operating characteristics, not its answer quality.
"""

from __future__ import annotations


def percentile(values: list[float], p: float) -> float:
    """The value at percentile `p` (0..1) of `values`, nearest-rank method.
    0.0 on an empty list rather than raising — a run with zero timed queries
    has nothing to report, not an error."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(int(p * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


def mean_cost_usd(costs: list[float | None]) -> float | None:
    """Average of the non-None costs, or None if none of them reported a
    cost at all (some OpenRouter routes don't) — never silently reported as
    $0, which would understate real spend."""
    known = [c for c in costs if c is not None]
    if not known:
        return None
    return sum(known) / len(known)
