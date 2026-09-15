"""Data models shared by every parser and chunker implementation.

These are the stable shapes that make parsing/chunking swappable (see CLAUDE.md):
any DocumentParser must produce a ParsedDocument, any Chunker must consume one and
produce Chunks. Nothing downstream should depend on *how* a chunk was produced.
"""

from __future__ import annotations

from pydantic import BaseModel


class ParsedPage(BaseModel):
    """One page of parsed text from a source document, before chunking."""

    page_number: int
    text: str
    section: str | None = None


class ParsedDocument(BaseModel):
    """Output of a DocumentParser: a source file broken into pages, not yet chunked."""

    doc_id: str
    source_path: str
    pages: list[ParsedPage]


class ChunkMetadata(BaseModel):
    """Travels with every chunk and every vector derived from it.

    Citations depend on these fields being present and correct — never construct
    a Chunk without them, and never strip metadata when moving a chunk through
    embedding/retrieval/reranking.
    """

    doc_id: str
    page: int
    section: str | None = None
    source_path: str


class Chunk(BaseModel):
    """A unit of retrievable text plus the metadata needed to cite it."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
