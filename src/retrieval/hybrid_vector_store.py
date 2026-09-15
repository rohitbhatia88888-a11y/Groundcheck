"""Hybrid VectorStore: fuses BM25 (sparse, exact-term) and dense (embedding
similarity) retrieval with a configurable linear weight.

BM25 catches exact-term/entity/number queries that dense embeddings often
miss entirely — e.g. "What is the purpose of Regulation (EU) 2019/816?" is
built around a specific identifier that a bi-encoder has no special reason
to embed near its defining passage, but BM25 will match directly. Dense
catches paraphrase/semantic matches BM25 can't (no shared vocabulary).

Both scores are computed against the FULL indexed corpus for every query —
not just each method's own top-k — so fusion never misses a BM25-only or
dense-only candidate the way "rerank what dense already found" approaches
would (a reranker can only reorder chunks retrieval already surfaced; it
can't reach past that pool for one BM25 alone would have found).

Wraps a real QdrantVectorStore for the dense half rather than reimplementing
vector search. Satisfies the VectorStore protocol
(src/retrieval/protocols.py) — query_text is required here, unlike pure
dense stores, since BM25 needs the raw query.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.retrieval.models import EmbeddedChunk, RetrievedChunk
from src.retrieval.qdrant_vector_store import QdrantVectorStore

_TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = scores.values()
    lo, hi = min(values), max(values)
    if hi == lo:
        return dict.fromkeys(scores, 0.0)
    return {chunk_id: (v - lo) / (hi - lo) for chunk_id, v in scores.items()}


class HybridVectorStore:
    """VectorStore that fuses dense cosine similarity and BM25 term-overlap
    scores: `fused = dense_weight * dense_norm + (1 - dense_weight) * bm25_norm`,
    each min-max normalized across the query's full result set first (raw
    BM25 and cosine scores live on incomparable scales)."""

    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        dense_weight: float = 0.5,
        location: str = ":memory:",
    ) -> None:
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError(f"dense_weight must be in [0, 1], got {dense_weight}")

        self.dense_weight = dense_weight
        self._dense_store = QdrantVectorStore(collection_name, vector_size, location)
        self._chunks: dict[str, EmbeddedChunk] = {}
        self._bm25: BM25Okapi | None = None
        self._corpus_ids: list[str] = []

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return
        self._dense_store.upsert(chunks)
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self._corpus_ids = list(self._chunks)
        tokenized_corpus = [_tokenize(self._chunks[cid].text) for cid in self._corpus_ids]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def query(
        self, query_vector: list[float], top_k: int, query_text: str | None = None
    ) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        if not query_text:
            raise ValueError("HybridVectorStore.query requires query_text (for BM25 scoring)")

        # Full-corpus scores from each method — see module docstring on why
        # this beats scoring only each method's own top-k.
        dense_hits = self._dense_store.query(query_vector, top_k=len(self._chunks))
        dense_scores = {hit.chunk_id: hit.score for hit in dense_hits}

        bm25_raw = self._bm25.get_scores(_tokenize(query_text))
        bm25_scores = dict(zip(self._corpus_ids, bm25_raw))

        dense_norm = _min_max_normalize(dense_scores)
        bm25_norm = _min_max_normalize(bm25_scores)

        fused = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=(
                    self.dense_weight * dense_norm.get(chunk_id, 0.0)
                    + (1 - self.dense_weight) * bm25_norm.get(chunk_id, 0.0)
                ),
            )
            for chunk_id, chunk in self._chunks.items()
        ]
        fused.sort(key=lambda c: c.score, reverse=True)
        return fused[:top_k]
