"""Tests for src/eval: retrieval metrics (pure) and config/golden-set schema.

The full run_experiment integration tests live in tests/test_runner.py.
"""

from __future__ import annotations

import json

from src.eval import (
    ExperimentConfig,
    GoldenSet,
    RelevantChunkRef,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from src.ingestion.models import ChunkMetadata
from src.retrieval.models import RetrievedChunk


def _retrieved(doc_id: str, page: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}-{page}",
        text="x",
        score=score,
        metadata=ChunkMetadata(doc_id=doc_id, page=page, source_path="x.pdf"),
    )


RETRIEVED = [_retrieved("bio", 1, 0.9), _retrieved("geo", 3, 0.7),
             _retrieved("bio", 2, 0.5), _retrieved("geo", 1, 0.3)]
RELEVANT = [RelevantChunkRef(doc_id="bio", page=1), RelevantChunkRef(doc_id="bio", page=2)]


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, RELEVANT, k=4) == 0.5
    assert precision_at_k(RETRIEVED, RELEVANT, k=1) == 1.0
    assert precision_at_k([], RELEVANT, k=5) == 0.0


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, RELEVANT, k=4) == 1.0
    assert recall_at_k(RETRIEVED, RELEVANT, k=1) == 0.5
    assert recall_at_k(RETRIEVED, [], k=5) == 0.0  # no relevant refs -> 0, not div-by-zero


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank(RETRIEVED, RELEVANT) == 1.0
    assert mean_reciprocal_rank([_retrieved("x", 9, 0.9)], RELEVANT) == 0.0


def test_golden_set_matches_by_doc_id_and_page_not_chunk_id(tmp_path):
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps({
        "items": [{
            "id": "q1", "question": "Q?",
            "relevant_chunks": [{"doc_id": "bio", "page": 1}],
            "expected_answer": "A.",
        }]
    }))
    golden_set = GoldenSet.load(golden_path)
    assert golden_set.items[0].relevant_chunks[0].doc_id == "bio"
    assert golden_set.items[0].question_type == "simple"  # default


def test_experiment_config_has_no_parser_or_results_path_field():
    # Parsing isn't a swappable module (CLAUDE.md); output paths are fixed
    # conventions (results/experiments.csv, results/runs/...), not per-config.
    assert "parser" not in ExperimentConfig.model_fields
    assert "results_path" not in ExperimentConfig.model_fields
