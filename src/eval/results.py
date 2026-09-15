"""Result models and writers for src/eval/runner.py's two output artifacts:

- results/experiments.csv: one row per experiment RUN (ExperimentResult),
  appended to — never rewritten — so configs stay comparable over time.
- results/runs/<config_name>_<timestamp>.jsonl: one row per golden-set
  QUESTION for that run (QuestionResult), for inspecting individual failures.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel

from src.eval.golden_set import QuestionType
from src.generation.models import Citation


class RetrievedRef(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    score: float


class QuestionResult(BaseModel):
    """Everything about how one golden-set question fared, for debugging a
    specific failure without re-running the whole experiment."""

    item_id: str
    question: str
    question_type: QuestionType

    retrieved: list[RetrievedRef]  # reranked pool, ordered — used for recall@k
    context_chunk_ids: list[str]  # prefix of `retrieved` actually shown to the generator
    has_relevant_chunks: bool  # False for most unanswerable items; excluded from
    # aggregate recall@k/MRR when False — recall against "nothing to retrieve" is
    # definitionally 0 and would misreport as a retrieval failure, not what it is.
    recall_at_k: dict[str, float]  # "1"/"3"/"5"/"10" -> value
    mrr: float

    answer_text: str
    citations: list[Citation]
    unsupported_citation_markers: list[str]
    citation_validity_rate: float | None  # None if the answer cited nothing at all
    model_used: str

    faithfulness: float
    faithfulness_reasoning: str
    answer_relevance: float
    answer_relevance_reasoning: str

    refused: bool | None  # only meaningful when question_type == "unanswerable"
    refusal_reasoning: str | None

    latency_seconds: float
    cost_usd: float | None


class ExperimentResult(BaseModel):
    """One row of results/experiments.csv — the aggregate outcome of running
    the full golden set against one config."""

    config_name: str
    config_path: str
    timestamp: str  # ISO-8601 UTC; matches the runs_path filename
    golden_set_hash: str
    num_questions: int

    chunker_type: str
    embedder_model: str
    reranker_type: str
    generator_model: str
    judge_model: str

    num_retrieval_eligible: int  # items with has_relevant_chunks=True; recall@k/MRR's denominator
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    mrr: float | None

    faithfulness: float
    answer_relevance: float
    citation_validity_rate: float | None  # pooled across every citation marker in the run

    num_unanswerable: int
    refusal_accuracy: float | None

    latency_p50_seconds: float
    latency_p95_seconds: float
    cost_per_query_usd: float | None

    runs_path: str


EXPERIMENT_CSV_FIELDS = list(ExperimentResult.model_fields.keys())


def write_question_results(path: str | Path, results: list[QuestionResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")


def append_experiment_result(path: str | Path, result: ExperimentResult) -> None:
    """Appends one row to results/experiments.csv, writing the header only if
    the file doesn't exist yet. Never rewrites prior rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(result.model_dump(mode="json"))
