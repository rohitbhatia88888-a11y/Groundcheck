"""Stable interfaces for embedding, vector storage, and reranking.

Each stage is selected per-experiment by name/params in a configs/*.yaml file
(see src/eval/config.py). Calling code should depend on these protocols, never
on a specific implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.ingestion.models import Chunk
from src.retrieval.models import EmbeddedChunk, RetrievedChunk


@runtime_checkable
class Embedder(Protocol):
    """Turns chunk text (and query text) into vectors."""

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]: ...

    def embed_query(self, query: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persists embedded chunks and returns the top-k nearest to a query vector."""

    def upsert(self, chunks: list[EmbeddedChunk]) -> None: ...

    def query(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieved chunks for a given query, keeping at most top_k."""

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...
