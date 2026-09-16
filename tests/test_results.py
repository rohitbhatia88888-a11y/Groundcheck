"""Tests for src/eval/results.py's experiments.csv writer — specifically the
header-migration logic, added after a real bug: appending a row with more
columns than the file's existing header produced a ragged CSV (values after
the new field silently misaligned for every row written before the schema
grew). See _migrate_header_if_schema_grew's docstring.
"""

from __future__ import annotations

import csv

from src.eval.results import (
    EXPERIMENT_CSV_FIELDS,
    ExperimentResult,
    append_experiment_result,
)


def _result(**overrides) -> ExperimentResult:
    defaults = {
        "config_name": "c",
        "config_path": "configs/c.yaml",
        "timestamp": "20260101T000000Z",
        "golden_set_hash": "deadbeef",
        "num_questions": 1,
        "chunker_type": "fixed_size",
        "embedder_model": "all-MiniLM-L6-v2",
        "reranker_type": "identity",
        "generator_model": "openai/gpt-4o-mini",
        "judge_model": "openai/gpt-4o-mini",
        "num_retrieval_eligible": 1,
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "mrr": 1.0,
        "faithfulness": 1.0,
        "answer_relevance": 1.0,
        "citation_validity_rate": 1.0,
        "num_unanswerable": 0,
        "refusal_accuracy": None,
        "latency_p50_seconds": 1.0,
        "latency_p95_seconds": 1.0,
        "cost_per_query_usd": 0.0001,
        "runs_path": "results/runs/c_20260101T000000Z.jsonl",
    }
    defaults.update(overrides)
    return ExperimentResult(**defaults)


def test_appending_to_a_fresh_file_writes_header_then_row(tmp_path):
    path = tmp_path / "experiments.csv"
    append_experiment_result(path, _result())

    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 1
    assert rows[0]["config_name"] == "c"


def test_appending_twice_never_touches_the_first_row(tmp_path):
    path = tmp_path / "experiments.csv"
    append_experiment_result(path, _result(config_name="first"))
    append_experiment_result(path, _result(config_name="second"))

    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert rows[0]["config_name"] == "first"
    assert rows[1]["config_name"] == "second"


def test_migrates_header_when_schema_has_grown_since_the_file_was_written(tmp_path):
    path = tmp_path / "experiments.csv"

    # Simulate a file written by an OLDER schema: current fields minus
    # citation_validity_rate, exactly what actually happened in this project.
    old_fields = [f for f in EXPERIMENT_CSV_FIELDS if f != "citation_validity_rate"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow(
            {
                "config_name": "baseline",
                "config_path": "configs/baseline.yaml",
                "timestamp": "20260101T000000Z",
                "golden_set_hash": "deadbeef",
                "num_questions": 18,
                "chunker_type": "fixed_size",
                "embedder_model": "all-MiniLM-L6-v2",
                "reranker_type": "identity",
                "generator_model": "openai/gpt-4o-mini",
                "judge_model": "openai/gpt-4o-mini",
                "num_retrieval_eligible": 18,
                "recall_at_1": 0.7222222222222222,
                "recall_at_3": 0.7222222222222222,
                "recall_at_5": 0.7777777777777778,
                "recall_at_10": 0.7777777777777778,
                "mrr": 0.7361111111111112,
                "faithfulness": 0.8888888888888888,
                "answer_relevance": 0.8888888888888888,
                "num_unanswerable": 0,
                "refusal_accuracy": "",
                "latency_p50_seconds": 2.0843,
                "latency_p95_seconds": 7.0172,
                "cost_per_query_usd": 0.000231,
                "runs_path": "results/runs/baseline_20260101T000000Z.jsonl",
            }
        )

    # Now append with the CURRENT (grown) schema, same as a real later run would.
    append_experiment_result(path, _result(config_name="hybrid"))

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert list(reader.fieldnames) == EXPERIMENT_CSV_FIELDS
        rows = list(reader)

    assert len(rows) == 2
    # The old row's real values must survive the migration, landing under
    # the RIGHT column names, not shifted by position.
    assert rows[0]["config_name"] == "baseline"
    assert rows[0]["recall_at_1"] == "0.7222222222222222"
    assert rows[0]["runs_path"] == "results/runs/baseline_20260101T000000Z.jsonl"
    # The field that didn't exist when this row was written must come back
    # blank, not shifted from a neighboring column.
    assert rows[0]["citation_validity_rate"] == ""

    assert rows[1]["config_name"] == "hybrid"
    assert rows[1]["citation_validity_rate"] == "1.0"

    # No row is ragged: every row has exactly as many values as the header.
    with path.open(newline="") as f:
        raw_rows = list(csv.reader(f))
    assert all(len(row) == len(EXPERIMENT_CSV_FIELDS) for row in raw_rows)
