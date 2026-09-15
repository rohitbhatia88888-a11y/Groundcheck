"""No-op Reranker: passes retrieval order through unchanged, truncated to top_k.

Placeholder that satisfies the Reranker protocol with zero extra dependencies.
Swap in a real reranker (cross-encoder, Cohere, ...) via config once the eval
harness can measure whether it actually improves retrieval metrics.
"""

from __future__ import annotations

from src.retrieval.models import RetrievedChunk


class IdentityReranker:
    """Reranker that does nothing but truncate to top_k."""

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return chunks[:top_k]
