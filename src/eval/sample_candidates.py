"""Samples chunks stratified across source documents and generates a
candidate (question, answer) pair for each via an LLM, writing
eval/candidates.jsonl for human review (src/eval/review_candidates.py).

Never resamples a chunk_id that already has a candidate on record (any
status) — safe to re-run as documents are added, without duplicating LLM
calls or review work.

Usage:
    python -m src.eval.sample_candidates --config configs/baseline.yaml -n 20
"""

from __future__ import annotations

import argparse
import random
from datetime import UTC, datetime
from pathlib import Path

from src.eval.candidate_generation import CandidateGenerator
from src.eval.candidates import Candidate, load_candidates, save_candidates
from src.eval.config import ExperimentConfig
from src.ingestion import Chunk
from src.ingestion.run import collect_chunks


def stratified_sample(chunks: list[Chunk], n: int, seed: int | None = None) -> list[Chunk]:
    """Samples up to n chunks, spread as evenly as possible across doc_ids —
    round-robin one chunk per document at a time, so representation stays
    fair even when document sizes differ wildly. Returns fewer than n if the
    pool is smaller than n.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    rng = random.Random(seed)
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.metadata.doc_id, []).append(chunk)
    for pool in by_doc.values():
        rng.shuffle(pool)

    doc_ids = sorted(by_doc)  # deterministic iteration order given a seed
    selected: list[Chunk] = []
    pointers = dict.fromkeys(doc_ids, 0)

    while len(selected) < n and any(pointers[d] < len(by_doc[d]) for d in doc_ids):
        for doc_id in doc_ids:
            if len(selected) == n:
                break
            pool = by_doc[doc_id]
            idx = pointers[doc_id]
            if idx < len(pool):
                selected.append(pool[idx])
                pointers[doc_id] = idx + 1

    return selected


def sample_and_generate(
    config: ExperimentConfig,
    n: int,
    candidates_path: str | Path,
    generator: CandidateGenerator | None = None,
    seed: int | None = None,
) -> int:
    """Samples up to n never-before-seen chunks, generates a candidate for
    each, and appends them to candidates_path. Returns how many were written
    (a chunk whose generation call raises is skipped with a warning, not
    recorded — so it's naturally eligible for resampling next run)."""
    generator = generator or CandidateGenerator()

    existing_candidates = load_candidates(candidates_path)
    already_seen = {c.chunk_id for c in existing_candidates}
    pool = [c for c in collect_chunks(config) if c.chunk_id not in already_seen]
    sampled = stratified_sample(pool, n, seed=seed)

    new_candidates: list[Candidate] = []
    for chunk in sampled:
        try:
            question, answer = generator.generate(chunk.text)
        except Exception as exc:  # noqa: BLE001 - one bad chunk shouldn't kill the whole run
            print(f"  skipped {chunk.chunk_id}: generation failed ({exc})")
            continue

        new_candidates.append(
            Candidate(
                candidate_id=f"cand-{chunk.chunk_id}",
                chunk_id=chunk.chunk_id,
                doc_id=chunk.metadata.doc_id,
                page=chunk.metadata.page,
                chunk_text=chunk.text,
                question=question,
                expected_answer=answer,
                generated_at=datetime.now(UTC).isoformat(),
            )
        )

    if new_candidates:
        save_candidates(candidates_path, existing_candidates + new_candidates)
    return len(new_candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample chunks stratified across documents and generate candidate Q/A pairs."
    )
    parser.add_argument("--config", required=True, help="Path to an experiment YAML (configs/*.yaml)")
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("--candidates", default="eval/candidates.jsonl")
    parser.add_argument("--seed", type=int, default=None, help="Set for reproducible sampling")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    count = sample_and_generate(config, args.num_samples, args.candidates, seed=args.seed)
    print(f"Wrote {count} new candidate(s) to {args.candidates}")


if __name__ == "__main__":
    main()
