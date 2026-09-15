"""Cross-encoder Reranker: scores each (query, chunk) pair jointly via a
CrossEncoder model, then reorders by that score.

A cross-encoder attends query and passage together in one forward pass,
which is typically far more accurate than the bi-encoder cosine similarity
used at retrieval time (query and chunk embedded independently, compared
after the fact) — the standard reason to retrieve a wide candidate pool and
rerank it down, rather than trusting dense retrieval's own top-k ordering.

CrossEncoder ships inside sentence-transformers (already a dependency) — no
new package needed. Satisfies the Reranker protocol
(src/retrieval/protocols.py).
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.retrieval.models import RetrievedChunk

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks retrieved chunks by a cross-encoder relevance score."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)

        rescored = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(chunks, scores)
        ]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_k]
