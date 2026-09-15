"""Schema and JSONL read/write for eval/candidates.jsonl — the staging area
between LLM-generated candidate questions (src/eval/sample_candidates.py) and
the reviewed, frozen golden set (src/eval/golden_set.py).

Unlike golden_set.json, this file IS freely rewritten on every review
decision (status/edits) — it's working state, not the frozen artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.eval.golden_set import QuestionType

CandidateStatus = Literal["pending", "accepted", "rejected"]


class Candidate(BaseModel):
    """One LLM-generated (question, answer) pair proposed from a single
    sampled chunk, awaiting human review."""

    candidate_id: str
    chunk_id: str
    doc_id: str
    page: int
    chunk_text: str
    question: str
    expected_answer: str
    question_type: QuestionType = "simple"
    status: CandidateStatus = "pending"
    generated_at: str  # ISO-8601 UTC
    reviewed_at: str | None = None


def load_candidates(path: str | Path) -> list[Candidate]:
    path = Path(path)
    if not path.exists():
        return []
    return [
        Candidate.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def save_candidates(path: str | Path, candidates: list[Candidate]) -> None:
    """Overwrites `path` with `candidates`, one JSON object per line.
    Atomic (temp file + rename) so an interrupted write can't corrupt
    already-reviewed progress."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for candidate in candidates:
            f.write(candidate.model_dump_json() + "\n")
    tmp_path.replace(path)
