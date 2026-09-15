"""Tests for src/eval: retrieval metrics (pure), config schema, and a full
run_experiment integration test with only the OpenRouter network call mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pymupdf
import pytest

from src.eval import (
    ExperimentConfig,
    GoldenSet,
    RelevantChunkRef,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    run_experiment,
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


def test_experiment_config_has_no_parser_field():
    # Parsing isn't a swappable module per CLAUDE.md; it must never re-appear
    # as a config-driven stage.
    assert "parser" not in ExperimentConfig.model_fields


@pytest.fixture
def openrouter_mock(monkeypatch):
    """Mocks the OpenRouter network call for both the generator (plain text
    response) and the judge (forced tool-call response) based on which kwargs
    the call was made with."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-test-key")

    def fake_create(*args, **kwargs):
        if "tools" in kwargs:
            tool_call = MagicMock()
            tool_call.function.arguments = json.dumps({"score": 0.8, "reasoning": "looks fine"})
            message = MagicMock(tool_calls=[tool_call])
        else:
            message = MagicMock(content="Mitochondria produce ATP [bio-p1-c0].")
        return MagicMock(choices=[MagicMock(message=message)])

    with patch("src.generation.openrouter_client.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = fake_create
        yield


def test_run_experiment_end_to_end(tmp_path, openrouter_mock):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    doc = pymupdf.open()
    doc.new_page().insert_textbox(
        pymupdf.Rect(72, 72, 523, 770),
        "Mitochondria produce ATP through cellular respiration.",
    )
    doc.save(raw_dir / "bio.pdf")
    doc.close()

    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps({
        "items": [{
            "id": "q1",
            "question": "What do mitochondria produce?",
            "relevant_chunks": [{"doc_id": "bio", "page": 1}],
            "expected_answer": "ATP.",
        }]
    }))

    results_path = tmp_path / "results" / "baseline.csv"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
name: baseline
raw_data_dir: {raw_dir}
golden_set_path: {golden_path}
results_path: {results_path}
chunker: {{type: fixed_size, params: {{chunk_size: 500, chunk_overlap: 50}}}}
embedder: {{type: sentence_transformers}}
vector_store: {{type: qdrant, params: {{collection_name: baseline, location: ':memory:'}}}}
reranker: {{type: identity}}
generator: {{type: openrouter}}
retrieval_top_k: 5
rerank_top_k: 3
""")

    out_path = run_experiment(config_path)
    assert out_path == results_path

    rows = results_path.read_text().splitlines()
    assert len(rows) == 2  # header + one golden-set item
    row = dict(zip(rows[0].split(","), rows[1].split(",")))
    assert row["precision_at_k"] == "1.0"
    assert row["recall_at_k"] == "1.0"
    assert row["faithfulness"] == "0.8"
    assert row["answer_relevancy"] == "0.8"
