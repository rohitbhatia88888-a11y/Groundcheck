"""Data models shared by every embedder, vector store, and reranker implementation.

Both extend ingestion.Chunk rather than duplicating its fields, so chunk_id/text/
metadata stay identical across the whole pipeline and citations never drift.
"""

from __future__ import annotations

from src.ingestion.models import Chunk


class EmbeddedChunk(Chunk):
    """A Chunk plus its embedding vector, ready to be upserted into a VectorStore."""

    vector: list[float]


class RetrievedChunk(Chunk):
    """A chunk returned by a VectorStore or Reranker, with a relevance score.

    Score meaning is stage-dependent (cosine similarity after VectorStore.query,
    rerank score after Reranker.rerank) but higher is always more relevant.
    """

    score: float
