"""Retrieval metrics: computed purely from retrieved chunks vs. the golden set's
relevant (doc_id, page) refs — no LLM calls, fully deterministic.

Reported separately from generation metrics (src/eval/generation_metrics.py),
per CLAUDE.md: retrieval quality and generation quality answer different
questions and must never be blended into one score.
"""

from __future__ import annotations

from src.eval.golden_set import RelevantChunkRef
from src.retrieval.models import RetrievedChunk


def _relevant_set(relevant: list[RelevantChunkRef]) -> set[tuple[str, int]]:
    return {(r.doc_id, r.page) for r in relevant}


def _is_hit(chunk: RetrievedChunk, relevant_set: set[tuple[str, int]]) -> bool:
    return (chunk.metadata.doc_id, chunk.metadata.page) in relevant_set


def precision_at_k(
    retrieved: list[RetrievedChunk], relevant: list[RelevantChunkRef], k: int
) -> float:
    """Fraction of the top-k retrieved chunks that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant_set = _relevant_set(relevant)
    hits = sum(1 for c in top_k if _is_hit(c, relevant_set))
    return hits / len(top_k)


def recall_at_k(
    retrieved: list[RetrievedChunk], relevant: list[RelevantChunkRef], k: int
) -> float:
    """Fraction of all relevant (doc_id, page) refs found in the top-k retrieved chunks."""
    relevant_set = _relevant_set(relevant)
    if not relevant_set:
        return 0.0
    top_k = retrieved[:k]
    hits = len({(c.metadata.doc_id, c.metadata.page) for c in top_k} & relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(
    retrieved: list[RetrievedChunk], relevant: list[RelevantChunkRef]
) -> float:
    """1 / rank of the first relevant chunk in `retrieved`; 0.0 if none is relevant."""
    relevant_set = _relevant_set(relevant)
    for rank, chunk in enumerate(retrieved, start=1):
        if _is_hit(chunk, relevant_set):
            return 1.0 / rank
    return 0.0
