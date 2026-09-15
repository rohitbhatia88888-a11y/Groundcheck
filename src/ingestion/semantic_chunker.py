"""Semantic Chunker: groups consecutive sentences into chunks, splitting where
sentence-embedding similarity drops — a rough topic-shift heuristic — instead
of at a fixed size or a heading boundary.

Chunks never cross a page boundary, so ChunkMetadata.page stays unambiguous.
Satisfies the Chunker protocol (src/ingestion/protocols.py).

Uses an Embedder (src/retrieval/protocols.py) purely as an internal
sentence-similarity tool at chunk time; it has nothing to do with the
indexing embedder chosen elsewhere in an experiment config, though by default
it's the same implementation (SentenceTransformersEmbedder).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.ingestion.models import Chunk, ChunkMetadata, ParsedDocument

if TYPE_CHECKING:
    # Type-hint only: ingestion must not runtime-depend on retrieval (it's the
    # more foundational package — retrieval depends on ingestion, not the
    # reverse). `from __future__ import annotations` already defers evaluation
    # of the annotation below, so this import never actually runs — a real
    # top-level import here creates a circular import via
    # retrieval.models -> ingestion.models -> ingestion.__init__ -> this file.
    from src.retrieval.protocols import Embedder

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(int(p * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


class SemanticChunker:
    """Splits page text into sentences and groups them at embedding-similarity
    breakpoints: wherever consecutive-sentence similarity falls in the bottom
    `breakpoint_percentile` of that page's similarities, start a new chunk.
    `max_chunk_chars` is a hard cap so a page with few/no breakpoints can't
    produce one runaway chunk.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        breakpoint_percentile: float = 0.25,
        max_chunk_chars: int = 2000,
    ) -> None:
        if not 0.0 < breakpoint_percentile < 1.0:
            raise ValueError(
                f"breakpoint_percentile must be in (0, 1), got {breakpoint_percentile}"
            )
        if max_chunk_chars <= 0:
            raise ValueError(f"max_chunk_chars must be positive, got {max_chunk_chars}")

        if embedder is None:
            from src.retrieval.sentence_transformers_embedder import (
                SentenceTransformersEmbedder,
            )

            embedder = SentenceTransformersEmbedder()

        self._embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chunk_chars = max_chunk_chars

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []

        for page in document.pages:
            sentences = _split_sentences(page.text)
            if not sentences:
                continue

            for index, group in enumerate(self._group_sentences(sentences)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}-p{page.page_number}-s{index}",
                        text=" ".join(group),
                        metadata=ChunkMetadata(
                            doc_id=document.doc_id,
                            page=page.page_number,
                            section=page.section,
                            source_path=document.source_path,
                        ),
                    )
                )

        return chunks

    def _group_sentences(self, sentences: list[str]) -> list[list[str]]:
        if len(sentences) == 1:
            return [sentences]

        # One embed_query call per sentence rather than a batch embed_chunks
        # call — these are throwaway similarity vectors, not Chunks to index,
        # and Embedder has no raw-text batch method. Fine for offline chunking,
        # not a hot path.
        vectors = [self._embedder.embed_query(s) for s in sentences]
        similarities = [
            _cosine_similarity(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
        ]
        threshold = _percentile(similarities, self.breakpoint_percentile)

        groups: list[list[str]] = []
        current = [sentences[0]]
        current_chars = len(sentences[0])

        for i, sentence in enumerate(sentences[1:], start=1):
            is_breakpoint = similarities[i - 1] < threshold
            over_length = current_chars + len(sentence) > self.max_chunk_chars
            if is_breakpoint or over_length:
                groups.append(current)
                current = [sentence]
                current_chars = len(sentence)
            else:
                current.append(sentence)
                current_chars += len(sentence)

        groups.append(current)
        return groups
