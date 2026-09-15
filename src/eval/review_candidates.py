"""Terminal review loop for eval/candidates.jsonl: show each pending
candidate's chunk + generated question/answer, accept / edit / reject / skip.
Accepted items are appended to eval/golden_set.json via
GoldenSet.append_item — append-only, existing entries are never touched.

Usage:
    python -m src.eval.review_candidates
    python -m src.eval.review_candidates --candidates eval/candidates.jsonl --golden-set eval/golden_set.json
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from src.eval.candidates import Candidate, load_candidates, save_candidates
from src.eval.golden_set import GoldenSet, GoldenSetItem, QuestionType, RelevantChunkRef

_QUESTION_TYPES: list[QuestionType] = ["simple", "multi_hop", "unanswerable", "distractor"]

InputFunc = Callable[[str], str]


def _prompt_question_type(input_func: InputFunc, default: QuestionType) -> QuestionType:
    labels = ", ".join(t.upper() if t == default else t for t in _QUESTION_TYPES)
    while True:
        raw = input_func(f"  question_type [{labels}] (Enter for {default}): ").strip().lower()
        if not raw:
            return default
        if raw in _QUESTION_TYPES:
            return raw  # type: ignore[return-value]
        print(f"  not a valid type — pick one of: {', '.join(_QUESTION_TYPES)}")


def _show(candidate: Candidate) -> None:
    print("\n" + "=" * 72)
    print(f"chunk_id: {candidate.chunk_id}  (doc={candidate.doc_id}, page={candidate.page})")
    print("-" * 72)
    print(candidate.chunk_text[:1000])
    print("-" * 72)
    print(f"Q: {candidate.question}")
    print(f"A: {candidate.expected_answer}")
    print("=" * 72)


def _review_one(candidate: Candidate, input_func: InputFunc) -> tuple[str, Candidate]:
    """Returns (decision, possibly-edited candidate); decision is one of
    'accept', 'reject', 'skip', 'quit'."""
    _show(candidate)

    while True:
        choice = input_func("[a]ccept / [e]dit / [r]eject / [s]kip / [q]uit: ").strip().lower()

        if choice in ("a", "accept"):
            question_type = _prompt_question_type(input_func, "simple")
            return "accept", candidate.model_copy(update={"question_type": question_type})

        if choice in ("e", "edit"):
            new_question = input_func(f"  question [{candidate.question}]: ").strip() or candidate.question
            new_answer = (
                input_func(f"  answer [{candidate.expected_answer}]: ").strip()
                or candidate.expected_answer
            )
            question_type = _prompt_question_type(input_func, "simple")
            edited = candidate.model_copy(
                update={
                    "question": new_question,
                    "expected_answer": new_answer,
                    "question_type": question_type,
                }
            )
            return "accept", edited

        if choice in ("r", "reject"):
            return "reject", candidate

        if choice in ("s", "skip"):
            return "skip", candidate

        if choice in ("q", "quit"):
            return "quit", candidate

        print("  please enter a, e, r, s, or q")


def run_review(
    candidates_path: str | Path,
    golden_set_path: str | Path,
    input_func: InputFunc = input,
) -> None:
    candidates_path = Path(candidates_path)
    golden_set_path = Path(golden_set_path)

    candidates = load_candidates(candidates_path)
    pending_indices = [i for i, c in enumerate(candidates) if c.status == "pending"]

    if not pending_indices:
        print(f"No pending candidates in {candidates_path}.")
        return

    print(f"{len(pending_indices)} pending candidate(s) to review.")

    for i in pending_indices:
        decision, result = _review_one(candidates[i], input_func)

        if decision == "quit":
            print("Stopping — remaining candidates are left pending.")
            break
        if decision == "skip":
            continue  # left pending, nothing to persist

        reviewed_at = datetime.now(UTC).isoformat()

        if decision == "accept":
            item = GoldenSetItem(
                id=result.candidate_id,
                question=result.question,
                relevant_chunks=[RelevantChunkRef(doc_id=result.doc_id, page=result.page)],
                expected_answer=result.expected_answer,
                question_type=result.question_type,
            )
            GoldenSet.append_item(golden_set_path, item)
            candidates[i] = result.model_copy(update={"status": "accepted", "reviewed_at": reviewed_at})
            print(f"  -> accepted, appended to {golden_set_path}")
        else:  # reject
            candidates[i] = result.model_copy(update={"status": "rejected", "reviewed_at": reviewed_at})
            print("  -> rejected")

        save_candidates(candidates_path, candidates)  # persist progress after every decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review eval/candidates.jsonl and build eval/golden_set.json."
    )
    parser.add_argument("--candidates", default="eval/candidates.jsonl")
    parser.add_argument("--golden-set", default="eval/golden_set.json")
    args = parser.parse_args()
    run_review(args.candidates, args.golden_set)


if __name__ == "__main__":
    main()
