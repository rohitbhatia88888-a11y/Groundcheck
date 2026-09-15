"""Stable interfaces for parsing and chunking.

Concrete implementations (PyMuPDF-backed parser, unstructured-backed parser,
fixed-size chunker, semantic chunker, ...) live alongside these protocols and are
selected per-experiment by name/params in a configs/*.yaml file. Calling code
should depend on DocumentParser / Chunker, never on a specific implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from src.ingestion.models import Chunk, ParsedDocument


@runtime_checkable
class DocumentParser(Protocol):
    """Turns a raw source file into a ParsedDocument (page-level text, no chunking)."""

    def parse(self, path: Path) -> ParsedDocument: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a ParsedDocument into Chunks.

    Every returned Chunk must carry complete ChunkMetadata (doc_id, page, section) —
    citations depend on it. See CLAUDE.md.
    """

    def chunk(self, document: ParsedDocument) -> list[Chunk]: ...
