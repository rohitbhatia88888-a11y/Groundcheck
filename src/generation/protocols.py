"""Stable interface for answer generation.

Selected per-experiment by name/params in a configs/*.yaml file. Calling code
should depend on Generator, never on a specific implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.generation.models import Answer
from src.retrieval.models import RetrievedChunk


@runtime_checkable
class Generator(Protocol):
    """Answers a query from a fixed set of retrieved chunks."""

    def generate(self, query: str, context: list[RetrievedChunk]) -> Answer: ...
