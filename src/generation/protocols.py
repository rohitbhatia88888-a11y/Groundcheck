"""Stable interface for answer generation.

Selected per-experiment by name/params in a configs/*.yaml file. Calling code
should depend on Generator, never on a specific implementation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from src.generation.models import Answer
from src.retrieval.models import RetrievedChunk


@runtime_checkable
class Generator(Protocol):
    """Answers a query from a fixed set of retrieved chunks."""

    def generate(self, query: str, context: list[RetrievedChunk]) -> Answer: ...


@runtime_checkable
class StreamingGenerator(Protocol):
    """Optional capability: streams answer text incrementally, one delta at
    a time. NOT part of Generator — eval only ever needs a complete Answer,
    so it shouldn't force every implementation to support streaming. The
    live API (src/api/app.py) requires it explicitly via an isinstance
    check at startup instead.

    Citations can't be checked until the full text is known (a marker can
    appear anywhere, including the last token) — callers must accumulate
    the yielded deltas and run extract_citations() themselves once the
    iterator is exhausted; this only streams text.
    """

    def generate_stream(self, query: str, context: list[RetrievedChunk]) -> Iterator[str]: ...
