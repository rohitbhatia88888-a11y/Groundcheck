"""Runs the full golden set against one experiment config end-to-end and
writes two things:

- results/experiments.csv: one aggregate row for THIS run (recall@1/3/5/10,
  MRR, faithfulness, answer relevance, citation validity rate, refusal
  accuracy, p50/p95 latency, cost per query) — appended to, never rewritten,
  so configs stay comparable over time.
- results/runs/<config_name>_<timestamp>.jsonl: one row per golden-set
  question with full detail (retrieved chunks, generated text, every
  judge's reasoning, latency, cost), for inspecting individual failures.

Retrieval / generation / behaviour / ops metrics are computed and reported
as strictly separate columns (see CLAUDE.md) — never blended into one score.
temperature=0 on every LLM call for determinism where the provider honors
it; every question's answer_text records the exact model that served it
(model_used, from the API response, in case of upstream routing).

Fails loudly, before any work happens (see golden_set.verify_golden_set_hash),
if eval/golden_set.json has changed since it was locked — results from
before and after such a change are not comparable and must never be
silently conflated into one experiments.csv.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from src.eval.behaviour_metrics import refusal_accuracy
from src.eval.config import ExperimentConfig
from src.eval.generation_metrics import OpenRouterJudge, citation_validity_rate
from src.eval.golden_set import GoldenSet, GoldenSetItem, verify_golden_set_hash
from src.eval.ops_metrics import mean_cost_usd, percentile
from src.eval.registry import Pipeline, build_pipeline
from src.eval.results import (
    ExperimentResult,
    QuestionResult,
    RetrievedRef,
    append_experiment_result,
    write_question_results,
)
from src.eval.retrieval_metrics import mean_reciprocal_rank, recall_at_k
from src.ingestion import PyMuPDFParser, table_to_chunk
from src.ingestion.models import Chunk

RECALL_K_VALUES = (1, 3, 5, 10)
EXPERIMENTS_CSV_PATH = Path("results/experiments.csv")
RUNS_DIR = Path("results/runs")


def ingest_raw_documents(config: ExperimentConfig, pipeline: Pipeline) -> int:
    """Parses, chunks, embeds, and upserts every *.pdf under raw_data_dir.
    Prose is chunked by the configured Chunker; tables are extracted
    separately and indexed as their own chunks (see src/ingestion/tables.py).
    Returns the number of chunks indexed.

    Parsing isn't a swappable module (see CLAUDE.md), so PyMuPDFParser is used
    directly here rather than coming from the config-driven Pipeline.
    """
    parser = PyMuPDFParser()
    raw_dir = Path(config.raw_data_dir)
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.glob("**/*.pdf")):
        document = parser.parse(path)
        chunks.extend(pipeline.chunker.chunk(document))
        chunks.extend(table_to_chunk(t) for t in document.tables)

    embedded = pipeline.embedder.embed_chunks(chunks)
    pipeline.vector_store.upsert(embedded)
    return len(embedded)


def _pool_size(config: ExperimentConfig) -> int:
    """How many chunks to retrieve+rerank per question — enough to compute
    every recall@k in RECALL_K_VALUES honestly, regardless of how small the
    config's own retrieval_top_k/rerank_top_k are (those still control what's
    actually shown to the generator, via the slice in _evaluate_item)."""
    return max(config.retrieval_top_k, config.rerank_top_k, *RECALL_K_VALUES)


def run_experiment(
    config_path: str | Path,
    judge: OpenRouterJudge | None = None,
    experiments_csv_path: str | Path = EXPERIMENTS_CSV_PATH,
    runs_dir: str | Path = RUNS_DIR,
) -> ExperimentResult:
    """Runs `config_path` against the golden set it references. Returns the
    aggregate ExperimentResult (also the row appended to experiments_csv_path);
    per-question detail is written to runs_dir."""
    config = ExperimentConfig.from_yaml(config_path)

    # Fail fast, before loading any model or touching the vector store.
    golden_set_hash = verify_golden_set_hash(config.golden_set_path)

    pipeline = build_pipeline(config)
    judge = judge or OpenRouterJudge()

    ingest_raw_documents(config, pipeline)
    golden_set = GoldenSet.load(config.golden_set_path)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    pool_size = _pool_size(config)

    question_results = [
        _evaluate_item(item, config, pipeline, judge, pool_size) for item in golden_set.items
    ]

    runs_path = Path(runs_dir) / f"{config.name}_{timestamp}.jsonl"
    write_question_results(runs_path, question_results)

    result = _aggregate(
        config=config,
        config_path=str(config_path),
        timestamp=timestamp,
        golden_set_hash=golden_set_hash,
        results=question_results,
        pipeline=pipeline,
        judge=judge,
        runs_path=str(runs_path),
    )
    append_experiment_result(experiments_csv_path, result)

    return result


def _evaluate_item(
    item: GoldenSetItem,
    config: ExperimentConfig,
    pipeline: Pipeline,
    judge: OpenRouterJudge,
    pool_size: int,
) -> QuestionResult:
    # Timed: exactly what a real user's query would go through. Judge calls
    # happen after, outside this block — they're eval overhead, not part of
    # what's being measured as "ops."
    start = time.perf_counter()

    query_vector = pipeline.embedder.embed_query(item.question)
    retrieved = pipeline.vector_store.query(query_vector, top_k=pool_size, query_text=item.question)
    reranked = pipeline.reranker.rerank(item.question, retrieved, top_k=pool_size)
    context = reranked[: config.rerank_top_k]

    answer = pipeline.generator.generate(item.question, context)

    latency_seconds = time.perf_counter() - start

    recall_by_k = {str(k): recall_at_k(reranked, item.relevant_chunks, k) for k in RECALL_K_VALUES}
    mrr = mean_reciprocal_rank(reranked, item.relevant_chunks)

    faithfulness = judge.score_faithfulness(answer)
    relevance = judge.score_relevancy(item.question, answer)

    refused: bool | None = None
    refusal_reasoning: str | None = None
    if item.question_type == "unanswerable":
        judgment = judge.judge_refusal(answer)
        refused, refusal_reasoning = judgment.refused, judgment.reasoning

    return QuestionResult(
        item_id=item.id,
        question=item.question,
        question_type=item.question_type,
        retrieved=[
            RetrievedRef(chunk_id=c.chunk_id, doc_id=c.metadata.doc_id, page=c.metadata.page, score=c.score)
            for c in reranked
        ],
        context_chunk_ids=[c.chunk_id for c in context],
        has_relevant_chunks=bool(item.relevant_chunks),
        recall_at_k=recall_by_k,
        mrr=mrr,
        answer_text=answer.text,
        citations=answer.citations,
        unsupported_citation_markers=answer.unsupported_citation_markers,
        citation_validity_rate=citation_validity_rate(
            answer.citations, answer.unsupported_citation_markers
        ),
        model_used=answer.model_used,
        faithfulness=faithfulness.score,
        faithfulness_reasoning=faithfulness.reasoning,
        answer_relevance=relevance.score,
        answer_relevance_reasoning=relevance.reasoning,
        refused=refused,
        refusal_reasoning=refusal_reasoning,
        latency_seconds=latency_seconds,
        cost_usd=answer.cost_usd,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(
    config: ExperimentConfig,
    config_path: str,
    timestamp: str,
    golden_set_hash: str,
    results: list[QuestionResult],
    pipeline: Pipeline,
    judge: OpenRouterJudge,
    runs_path: str,
) -> ExperimentResult:
    unanswerable_refusals = [r.refused for r in results if r.refused is not None]
    retrieval_eligible = [r for r in results if r.has_relevant_chunks]

    return ExperimentResult(
        config_name=config.name,
        config_path=config_path,
        timestamp=timestamp,
        golden_set_hash=golden_set_hash,
        num_questions=len(results),
        chunker_type=config.chunker.type,
        embedder_model=getattr(pipeline.embedder, "model_name", type(pipeline.embedder).__name__),
        reranker_type=config.reranker.type,
        generator_model=getattr(pipeline.generator, "model", type(pipeline.generator).__name__),
        judge_model=judge.model,
        num_retrieval_eligible=len(retrieval_eligible),
        recall_at_1=_mean_or_none([r.recall_at_k["1"] for r in retrieval_eligible]),
        recall_at_3=_mean_or_none([r.recall_at_k["3"] for r in retrieval_eligible]),
        recall_at_5=_mean_or_none([r.recall_at_k["5"] for r in retrieval_eligible]),
        recall_at_10=_mean_or_none([r.recall_at_k["10"] for r in retrieval_eligible]),
        mrr=_mean_or_none([r.mrr for r in retrieval_eligible]),
        faithfulness=_mean([r.faithfulness for r in results]),
        answer_relevance=_mean([r.answer_relevance for r in results]),
        # Pooled across every citation marker in the run (raw counts, not an
        # average of per-question rates) — a question with 10 citations
        # shouldn't count the same as one with 1 in the overall rate.
        citation_validity_rate=citation_validity_rate(
            [c for r in results for c in r.citations],
            [m for r in results for m in r.unsupported_citation_markers],
        ),
        num_unanswerable=len(unanswerable_refusals),
        refusal_accuracy=refusal_accuracy(unanswerable_refusals),
        latency_p50_seconds=percentile([r.latency_seconds for r in results], 0.50),
        latency_p95_seconds=percentile([r.latency_seconds for r in results], 0.95),
        cost_per_query_usd=mean_cost_usd([r.cost_usd for r in results]),
        runs_path=runs_path,
    )
