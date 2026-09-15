"""Schema and loader for the frozen golden set (eval/golden_set.json).

Relevant chunks are identified by (doc_id, page), not chunk_id — chunk_id
depends on the chunking strategy under test, but a question's relevant source
pages don't. This is what lets the same golden set grade every experiment
config without being rewritten per chunker.

eval/golden_set.json is FROZEN once created (see CLAUDE.md): never edit it to
improve scores. It isn't generated here — it must be authored against real
source documents in data/raw/, with real vetted reference answers.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class RelevantChunkRef(BaseModel):
    """A source location that should be retrieved for a golden question."""

    doc_id: str
    page: int


class GoldenSetItem(BaseModel):
    """One golden question: what should be retrieved, and a reference answer
    for judging generation quality."""

    id: str
    question: str
    relevant_chunks: list[RelevantChunkRef]
    expected_answer: str


class GoldenSet(BaseModel):
    items: list[GoldenSetItem]

    @classmethod
    def load(cls, path: str | Path) -> GoldenSet:
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)
