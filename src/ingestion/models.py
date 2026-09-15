"""Data models shared by every parser and chunker implementation.

These are the stable shapes that make parsing/chunking swappable (see CLAUDE.md):
any DocumentParser must produce a ParsedDocument, any Chunker must consume one and
produce Chunks. Nothing downstream should depend on *how* a chunk was produced.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Heading(BaseModel):
    """A detected heading line within a page, used by SectionAwareChunker to
    slice the page's text at section boundaries."""

    text: str
    char_offset: int  # offset into the owning ParsedPage.text where this heading starts


class ParsedPage(BaseModel):
    """One page of parsed text from a source document, before chunking.

    `section` is the section this page *inherits* — the most recent heading
    seen in an earlier page, i.e. the section active at the START of this
    page, before any of this page's own `headings` apply. Chunkers that don't
    do their own section splitting (FixedSizeChunker, ...) just tag every
    chunk from this page with it. `headings` are this page's own section
    transitions, each with the character offset in `text` where it starts —
    SectionAwareChunker uses these to split more precisely than per-page.
    """

    page_number: int
    text: str
    section: str | None = None
    headings: list[Heading] = []


class ExtractedTable(BaseModel):
    """A table detected on a page, kept out of ParsedPage.text and handled as
    its own unit — tables are never split by a Chunker (see
    src/ingestion/tables.py)."""

    doc_id: str
    page: int
    table_index: int  # position among tables found on this page, 0-indexed
    source_path: str
    markdown: str


class ParsedDocument(BaseModel):
    """Output of a DocumentParser: a source file broken into pages (prose) and
    tables, not yet chunked."""

    doc_id: str
    source_path: str
    pages: list[ParsedPage]
    tables: list[ExtractedTable] = []


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
    content_type: Literal["prose", "table"] = "prose"


class Chunk(BaseModel):
    """A unit of retrievable text plus the metadata needed to cite it."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
