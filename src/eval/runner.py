"""Runs one experiment end-to-end against the frozen golden set and writes
retrieval metrics and generation metrics to results/<experiment>.csv.

Retrieval and generation metrics are always reported as separate columns
(see CLAUDE.md) — never blended into one score.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.eval.config import ExperimentConfig
from src.eval.generation_metrics import ClaudeJudge
from src.eval.golden_set import GoldenSet
from src.eval.registry import Pipeline, build_pipeline
from src.eval.retrieval_metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from src.ingestion.models import Chunk

FIELDNAMES = [
    "item_id",
    "question",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "faithfulness",
    "answer_relevancy",
    "unsupported_citations",
]


def ingest_raw_documents(config: ExperimentConfig, pipeline: Pipeline) -> int:
    """Parses, chunks, embeds, and upserts every file under raw_data_dir.
    Returns the number of chunks indexed."""
    raw_dir = Path(config.raw_data_dir)
    chunks: list[Chunk] = []
    for path in sorted(p for p in raw_dir.glob("**/*") if p.is_file()):
        document = pipeline.parser.parse(path)
        chunks.extend(pipeline.chunker.chunk(document))

    embedded = pipeline.embedder.embed_chunks(chunks)
    pipeline.vector_store.upsert(embedded)
    return len(embedded)


def run_experiment(config_path: str | Path, judge: ClaudeJudge | None = None) -> Path:
    """Runs the experiment named in `config_path` and returns the results CSV path."""
    config = ExperimentConfig.from_yaml(config_path)
    pipeline = build_pipeline(config)
    judge = judge or ClaudeJudge()

    ingest_raw_documents(config, pipeline)
    golden_set = GoldenSet.load(config.golden_set_path)

    results_path = Path(config.results_path or f"results/{config.name}.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for item in golden_set.items:
            writer.writerow(_evaluate_item(item, config, pipeline, judge))

    return results_path


def _evaluate_item(item, config: ExperimentConfig, pipeline: Pipeline, judge: ClaudeJudge) -> dict:
    query_vector = pipeline.embedder.embed_query(item.question)
    retrieved = pipeline.vector_store.query(query_vector, top_k=config.retrieval_top_k)
    reranked = pipeline.reranker.rerank(item.question, retrieved, top_k=config.rerank_top_k)

    answer = pipeline.generator.generate(item.question, reranked)

    return {
        "item_id": item.id,
        "question": item.question,
        "precision_at_k": precision_at_k(reranked, item.relevant_chunks, config.rerank_top_k),
        "recall_at_k": recall_at_k(reranked, item.relevant_chunks, config.rerank_top_k),
        "mrr": mean_reciprocal_rank(reranked, item.relevant_chunks),
        "faithfulness": judge.score_faithfulness(answer).score,
        "answer_relevancy": judge.score_relevancy(item.question, answer).score,
        "unsupported_citations": len(answer.unsupported_citation_markers),
    }
