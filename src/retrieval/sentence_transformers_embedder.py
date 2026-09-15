"""sentence-transformers-backed Embedder: local, no API key, reproducible.

Default model is all-MiniLM-L6-v2 (384-dim, fast, a solid baseline). Swappable
via `model_name`, selected per-experiment from configs/*.yaml.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.ingestion.models import Chunk
from src.retrieval.models import EmbeddedChunk


class SentenceTransformersEmbedder:
    """Embeds chunk text and queries with a local sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []
        vectors = self._model.encode([c.text for c in chunks], convert_to_numpy=True)
        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                vector=vector.tolist(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

    def embed_query(self, query: str) -> list[float]:
        return self._model.encode(query, convert_to_numpy=True).tolist()
