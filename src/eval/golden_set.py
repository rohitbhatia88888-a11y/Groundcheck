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

import hashlib
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

    @staticmethod
    def content_hash(path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_golden_set_hash(path: str | Path) -> str:
    """Locks eval/golden_set.json's content hash the first time this is
    called, and fails loudly on every later call if the file no longer
    matches — a run must never silently score against a golden set that
    changed since it was locked, even a well-intentioned addition.

    The lock lives next to the golden set, e.g. eval/golden_set.sha256 for
    eval/golden_set.json. Returns the current hash (also the locked one,
    once this returns without raising).

    To accept a deliberate change (e.g. new entries via review_candidates.py),
    delete the lock file and re-run — this is a conscious step, not automatic,
    by design.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Build it first (see eval/README.md: "
            f"sample_candidates.py + review_candidates.py, or hand-author it)."
        )

    current_hash = GoldenSet.content_hash(path)
    lock_path = path.with_suffix(".sha256")

    if not lock_path.exists():
        lock_path.write_text(current_hash + "\n")
        print(f"Locked {path} at hash {current_hash[:12]}... (see {lock_path})")
        return current_hash

    locked_hash = lock_path.read_text().strip()
    if locked_hash != current_hash:
        raise RuntimeError(
            f"{path} has changed since it was locked!\n"
            f"  locked:  {locked_hash}\n"
            f"  current: {current_hash}\n"
            f"Results from before and after this change are not comparable. "
            f"If this change is deliberate and reviewed (e.g. new entries via "
            f"review_candidates.py), delete {lock_path} and re-run to accept "
            f"the new hash. If it isn't, something modified the frozen golden "
            f"set unexpectedly — investigate before trusting any eval numbers."
        )
    return current_hash
