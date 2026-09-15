"""Schema and loader/writer for the golden set (eval/golden_set.json).

Relevant chunks are identified by (doc_id, page), not chunk_id — chunk_id
depends on the chunking strategy under test, but a question's relevant source
pages don't. This is what lets the same golden set grade every experiment
config without being rewritten per chunker.

Per CLAUDE.md, eval/golden_set.json is FROZEN once an entry exists: never
edit or reorder an existing entry to improve scores. Entries are proposed via
src/eval/sample_candidates.py (LLM-generated candidates from sampled chunks)
and added one at a time through human review in
src/eval/review_candidates.py — `GoldenSet.append_item` below is the only
sanctioned way to grow the file, and it only ever appends; see its docstring.
Hand-authored hard cases (multi_hop / unanswerable / distractor) can also be
added directly by editing the JSON, since those aren't produced by the
single-chunk candidate generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

QuestionType = Literal["simple", "multi_hop", "unanswerable", "distractor"]


class RelevantChunkRef(BaseModel):
    """A source location that should be retrieved for a golden question."""

    doc_id: str
    page: int


class GoldenSetItem(BaseModel):
    """One golden question: what should be retrieved, and a reference answer
    for judging generation quality.

    question_type lets hard cases be told apart from the simple, auto-
    generated majority: multi_hop needs several relevant_chunks to answer;
    unanswerable should have no chunk that actually supports an answer
    (relevant_chunks is typically empty, expected_answer states that the
    context doesn't contain it); distractor pairs a plausible-looking but
    wrong chunk against the real one, to test that retrieval/generation isn't
    fooled by surface similarity.
    """

    id: str
    question: str
    relevant_chunks: list[RelevantChunkRef]
    expected_answer: str
    question_type: QuestionType = "simple"


class GoldenSet(BaseModel):
    items: list[GoldenSetItem]

    @classmethod
    def load(cls, path: str | Path) -> GoldenSet:
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)

    @classmethod
    def append_item(cls, path: str | Path, item: GoldenSetItem) -> None:
        """Appends one item to the golden set at `path`, creating the file
        (as `{"items": [item]}`) if it doesn't exist yet.

        This is the ONLY way this codebase writes eval/golden_set.json, and it
        never touches an existing entry: it loads whatever is already there
        unchanged, adds `item` to the end, and writes the result back
        atomically (temp file + rename). Raises ValueError instead of writing
        anything if `item.id` already exists, so a re-run or a bug can never
        silently overwrite a prior entry.
        """
        path = Path(path)
        existing = cls.load(path).items if path.exists() else []

        if any(existing_item.id == item.id for existing_item in existing):
            raise ValueError(f"golden set already has an item with id={item.id!r}")

        updated = cls(items=[*existing, item])

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(updated.model_dump(mode="json"), indent=2) + "\n")
        tmp_path.replace(path)  # atomic on POSIX: readers never see a half-written file
