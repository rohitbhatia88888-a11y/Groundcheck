"""Data models for generation output."""

from __future__ import annotations

from pydantic import BaseModel

from src.retrieval.models import RetrievedChunk


class Citation(BaseModel):
    """One citation backing a claim in a generated answer."""

    chunk_id: str
    doc_id: str
    page: int


class Answer(BaseModel):
    """A generated answer plus the citations that support it.

    `citations` only ever contains chunk_ids that were actually present in the
    context passed to the generator — see citation enforcement in
    src/generation/claude_generator.py. `unsupported_citation_markers` surfaces
    any bracketed marker in `text` that did NOT match a real chunk_id, so
    hallucinated citations are visible rather than silently dropped.
    """

    text: str
    citations: list[Citation]
    context: list[RetrievedChunk]
    unsupported_citation_markers: list[str] = []
