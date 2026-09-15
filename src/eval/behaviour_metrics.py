"""Behaviour metrics: does the system do the right thing, not just the
accurate thing — currently just refusal accuracy on unanswerable questions.

Kept separate from generation_metrics.py's faithfulness/relevancy by design
(see CLAUDE.md: metrics are computed and reported separately) even though
both are backed by the same OpenRouterJudge.judge_refusal — this module only
ever aggregates the resulting refused/not-refused outcomes, it makes no LLM
calls itself.
"""

from __future__ import annotations


def refusal_accuracy(refusals: list[bool]) -> float | None:
    """Fraction of unanswerable-question outcomes where the system correctly
    refused. None (not 0.0) if `refusals` is empty — there is no such thing
    as 0% refusal accuracy on zero questions, and reporting 0.0 would read as
    "the system always hallucinates," which isn't what an empty list means.
    """
    if not refusals:
        return None
    return sum(refusals) / len(refusals)
