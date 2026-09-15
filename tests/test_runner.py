"""Tests for src/eval/runner.py — the Phase 4 eval runner: golden-set hash
locking, retrieval-eligibility-aware recall@k/MRR, refusal judging, ops
tracking, and the two output artifacts (results/experiments.csv,
results/runs/*.jsonl).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest

from src.eval.golden_set import GoldenSet
from src.eval.results import EXPERIMENT_CSV_FIELDS
from src.eval.runner import run_experiment


@pytest.fixture
def experiment_setup(tmp_path):
    """One small PDF, a golden set with one answerable ("simple") and one
    unanswerable question, and a config pointing at both."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    doc = pymupdf.open()
    doc.new_page().insert_textbox(
        pymupdf.Rect(72, 72, 523, 770),
        "Mitochondria produce ATP through cellular respiration. " * 3,
    )
    doc.save(raw_dir / "bio.pdf")
    doc.close()

    golden_path = tmp_path / "eval" / "golden_set.json"
    golden_path.parent.mkdir(parents=True)
    golden_path.write_text(json.dumps({
        "items": [
            {
                "id": "q1", "question": "What do mitochondria produce?",
                "relevant_chunks": [{"doc_id": "bio", "page": 1}],
                "expected_answer": "ATP.", "question_type": "simple",
            },
            {
                "id": "q2", "question": "What is the capital of France?",
                "relevant_chunks": [], "expected_answer": "The context does not say.",
                "question_type": "unanswerable",
            },
        ]
    }))

    results_dir = tmp_path / "results"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
name: test_exp
raw_data_dir: {raw_dir}
golden_set_path: {golden_path}
chunker: {{type: fixed_size, params: {{chunk_size: 500, chunk_overlap: 50}}}}
embedder: {{type: sentence_transformers}}
vector_store: {{type: qdrant, params: {{collection_name: test_exp, location: ':memory:'}}}}
reranker: {{type: identity}}
generator: {{type: openrouter}}
retrieval_top_k: 5
rerank_top_k: 3
""")
    return config_path, golden_path, results_dir


def _fake_create(*args, **kwargs):
    messages = kwargs.get("messages", [])
    user_content = messages[-1]["content"] if messages else ""
    tools = kwargs.get("tools")

    if tools:
        tool_name = tools[0]["function"]["name"]
        if tool_name == "submit_refusal_judgment":
            arguments = json.dumps({"refused": True, "reasoning": "declines correctly"})
        else:  # submit_score
            arguments = json.dumps({"score": 0.9, "reasoning": "looks good"})
        tool_call = MagicMock()
        tool_call.function.arguments = arguments
        message = MagicMock(tool_calls=[tool_call])
    elif "capital of France" in user_content:
        message = MagicMock(content="The context does not contain this information.")
    else:
        message = MagicMock(content="Mitochondria produce ATP [bio-p1-c0].")

    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    response.model = "openai/gpt-4o-mini"
    response.usage = MagicMock(cost=0.00042)
    return response


@pytest.fixture
def openrouter_mock(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-test-key")
    with patch("src.generation.openrouter_client.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = _fake_create
        yield


class TestRunExperiment:
    def test_full_run_produces_correct_aggregate_and_per_question_output(
        self, experiment_setup, openrouter_mock
    ):
        config_path, _golden_path, results_dir = experiment_setup
        experiments_csv = results_dir / "experiments.csv"
        runs_dir = results_dir / "runs"

        result = run_experiment(config_path, experiments_csv_path=experiments_csv, runs_dir=runs_dir)

        assert result.num_questions == 2
        # q2 (unanswerable) is excluded from the retrieval denominator, so
        # recall@k reflects only q1, not dragged down by q2's empty
        # relevant_chunks.
        assert result.num_retrieval_eligible == 1
        assert result.recall_at_1 == 1.0
        assert result.mrr == 1.0
        assert result.num_unanswerable == 1
        assert result.refusal_accuracy == 1.0
        assert result.faithfulness == pytest.approx(0.9)
        assert result.answer_relevance == pytest.approx(0.9)
        assert result.cost_per_query_usd == pytest.approx(0.00042)
        assert result.latency_p50_seconds >= 0
        assert result.chunker_type == "fixed_size"
        assert result.embedder_model == "all-MiniLM-L6-v2"
        assert result.generator_model == "openai/gpt-4o-mini"

        rows = list(csv.DictReader(experiments_csv.open()))
        assert len(rows) == 1
        assert rows[0]["config_name"] == "test_exp"
        assert set(rows[0].keys()) == set(EXPERIMENT_CSV_FIELDS)

        jsonl_lines = Path(result.runs_path).read_text().splitlines()
        assert len(jsonl_lines) == 2
        by_id = {json.loads(line)["item_id"]: json.loads(line) for line in jsonl_lines}
        assert by_id["q1"]["recall_at_k"]["1"] == 1.0
        assert by_id["q1"]["has_relevant_chunks"] is True
        assert by_id["q1"]["refused"] is None  # not applicable to a simple question
        assert by_id["q2"]["has_relevant_chunks"] is False
        assert by_id["q2"]["refused"] is True
        assert by_id["q2"]["refusal_reasoning"] == "declines correctly"

    def test_appends_second_run_without_touching_first_row(self, experiment_setup, openrouter_mock):
        config_path, _golden_path, results_dir = experiment_setup
        experiments_csv = results_dir / "experiments.csv"
        runs_dir = results_dir / "runs"

        run_experiment(config_path, experiments_csv_path=experiments_csv, runs_dir=runs_dir)
        run_experiment(config_path, experiments_csv_path=experiments_csv, runs_dir=runs_dir)

        rows = list(csv.DictReader(experiments_csv.open()))
        assert len(rows) == 2
        assert rows[0]["config_name"] == rows[1]["config_name"] == "test_exp"
        assert rows[0]["timestamp"] != rows[1]["timestamp"]
        assert rows[0]["golden_set_hash"] == rows[1]["golden_set_hash"]

    def test_golden_set_hash_locked_on_first_run_and_enforced_on_second(
        self, experiment_setup, openrouter_mock
    ):
        config_path, golden_path, results_dir = experiment_setup
        lock_path = golden_path.with_suffix(".sha256")
        assert not lock_path.exists()

        run_experiment(
            config_path,
            experiments_csv_path=results_dir / "experiments.csv",
            runs_dir=results_dir / "runs",
        )
        assert lock_path.exists()
        assert lock_path.read_text().strip() == GoldenSet.content_hash(golden_path)

        # simulate an unreviewed edit to the "frozen" golden set
        data = json.loads(golden_path.read_text())
        data["items"][0]["question"] = "A DIFFERENT question entirely"
        golden_path.write_text(json.dumps(data))

        with pytest.raises(RuntimeError, match="changed since it was locked"):
            run_experiment(
                config_path,
                experiments_csv_path=results_dir / "experiments.csv",
                runs_dir=results_dir / "runs",
            )

    def test_missing_golden_set_fails_before_any_work(self, tmp_path):
        # No openrouter_mock, no API key set at all — if this reaches pipeline
        # construction it fails for the wrong reason. It must fail at the
        # hash check, before that.
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"""
name: t
raw_data_dir: {raw_dir}
golden_set_path: {tmp_path / "eval" / "golden_set.json"}
chunker: {{type: fixed_size}}
embedder: {{type: sentence_transformers}}
vector_store: {{type: qdrant, params: {{collection_name: t, location: ':memory:'}}}}
reranker: {{type: identity}}
generator: {{type: openrouter}}
""")
        with pytest.raises(FileNotFoundError):
            run_experiment(config_path, experiments_csv_path=tmp_path / "e.csv", runs_dir=tmp_path / "r")
