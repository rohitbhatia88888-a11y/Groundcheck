"""Qdrant-backed VectorStore.

Defaults to an in-process, in-memory Qdrant instance (`location=":memory:"`) so
it works without Docker for tests and quick iteration. Point `location` at a
running server (e.g. "http://localhost:6333") for real experiments, per the
Qdrant/Docker entry in CLAUDE.md's stack — selected via configs/*.yaml.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.ingestion.models import ChunkMetadata
from src.retrieval.models import EmbeddedChunk, RetrievedChunk


def _point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be an unsigned int or a UUID; chunk_id is neither,
    so derive a stable UUID from it. Re-upserting the same chunk_id overwrites
    the same point rather than duplicating it."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    """VectorStore backed by a Qdrant collection using cosine similarity."""

    def __init__(
        self, collection_name: str, vector_size: int, location: str = ":memory:"
    ) -> None:
        self.collection_name = collection_name
        self._client = QdrantClient(location=location)
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size, distance=qmodels.Distance.COSINE
                ),
            )

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return
        points = [
            qmodels.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=chunk.vector,
                payload={"chunk_id": chunk.chunk_id, "text": chunk.text, **chunk.metadata.model_dump()},
            )
            for chunk in chunks
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def query(
        self, query_vector: list[float], top_k: int, query_text: str | None = None
    ) -> list[RetrievedChunk]:
        # query_text is accepted for VectorStore protocol conformance (a
        # sparse/hybrid store needs it; pure dense search here doesn't) and
        # deliberately unused.
        del query_text
        response = self._client.query_points(
            collection_name=self.collection_name, query=query_vector, limit=top_k
        )
        return [
            RetrievedChunk(
                chunk_id=point.payload["chunk_id"],
                text=point.payload["text"],
                metadata=ChunkMetadata(
                    doc_id=point.payload["doc_id"],
                    page=point.payload["page"],
                    section=point.payload.get("section"),
                    source_path=point.payload["source_path"],
                ),
                score=point.score,
            )
            for point in response.points
        ]
