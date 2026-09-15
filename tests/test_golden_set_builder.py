"""Tests for the Phase 3 golden-set builder: stratified sampling, candidate
generation/storage, the review loop, and golden_set.json's append-only
write guarantee.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from src.eval.candidates import Candidate, load_candidates, save_candidates
from src.eval.config import ExperimentConfig
from src.eval.golden_set import GoldenSet, GoldenSetItem
from src.eval.review_candidates import run_review
from src.eval.sample_candidates import sample_and_generate, stratified_sample
from src.ingestion.run import collect_chunks


class _FakeGenerator:
    """Deterministic stand-in for CandidateGenerator: no network call, one
    incrementing fact per call, so tests can assert exactly what was
    generated without mocking an LLM response shape."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, chunk_text: str) -> tuple[str, str]:
        self.calls += 1
        return f"What is fact #{self.calls}?", f"Fact #{self.calls}."


class _FailingGenerator:
    def generate(self, chunk_text: str) -> tuple[str, str]:
        raise RuntimeError("simulated LLM failure")


@pytest.fixture
def two_doc_config(tmp_path) -> ExperimentConfig:
    """An ExperimentConfig pointed at a raw_data_dir with two small PDFs, for
    exercising stratification across documents."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name, text in [("doc_a.pdf", "Alpha content. " * 30), ("doc_b.pdf", "Beta content. " * 30)]:
        doc = pymupdf.open()
        doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 523, 770), text)
        doc.save(raw_dir / name)
        doc.close()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
name: t
raw_data_dir: {raw_dir}
chunker: {{type: fixed_size, params: {{chunk_size: 100, chunk_overlap: 20}}}}
embedder: {{type: sentence_transformers}}
vector_store: {{type: qdrant, params: {{collection_name: t, location: ":memory:"}}}}
reranker: {{type: identity}}
generator: {{type: openrouter}}
""")
    return ExperimentConfig.from_yaml(config_path)


class TestStratifiedSample:
    def test_spreads_evenly_across_documents(self, two_doc_config):
        chunks = collect_chunks(two_doc_config)
        sampled = stratified_sample(chunks, n=4, seed=1)

        assert len(sampled) == 4
        counts = {}
        for c in sampled:
            counts[c.metadata.doc_id] = counts.get(c.metadata.doc_id, 0) + 1
        assert counts == {"doc_a": 2, "doc_b": 2}

    def test_deterministic_with_same_seed(self, two_doc_config):
        chunks = collect_chunks(two_doc_config)
        first = stratified_sample(chunks, n=3, seed=7)
        second = stratified_sample(chunks, n=3, seed=7)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_n_larger_than_pool_returns_everything_without_error(self, two_doc_config):
        chunks = collect_chunks(two_doc_config)
        sampled = stratified_sample(chunks, n=10_000, seed=1)
        assert len(sampled) == len(chunks)
        assert {c.chunk_id for c in sampled} == {c.chunk_id for c in chunks}

    def test_rejects_non_positive_n(self):
        with pytest.raises(ValueError):
            stratified_sample([], n=0)


class TestCandidateStorage:
    def test_jsonl_round_trip(self, tmp_path):
        path = tmp_path / "candidates.jsonl"
        candidates = [
            Candidate(
                candidate_id="cand-1", chunk_id="c1", doc_id="d", page=1,
                chunk_text="text", question="Q?", expected_answer="A.",
                generated_at="2026-01-01T00:00:00+00:00",
            )
        ]
        save_candidates(path, candidates)
        loaded = load_candidates(path)
        assert loaded == candidates

    def test_missing_file_loads_as_empty_list(self, tmp_path):
        assert load_candidates(tmp_path / "nope.jsonl") == []


class TestSampleAndGenerate:
    def test_writes_one_candidate_per_sampled_chunk(self, two_doc_config, tmp_path):
        candidates_path = tmp_path / "eval" / "candidates.jsonl"
        generator = _FakeGenerator()

        count = sample_and_generate(two_doc_config, n=4, candidates_path=candidates_path,
                                      generator=generator, seed=1)

        assert count == 4
        assert generator.calls == 4
        candidates = load_candidates(candidates_path)
        assert len(candidates) == 4
        assert all(c.status == "pending" for c in candidates)
        assert all(c.question_type == "simple" for c in candidates)

    def test_rerun_never_resamples_a_seen_chunk(self, two_doc_config, tmp_path):
        candidates_path = tmp_path / "eval" / "candidates.jsonl"
        generator = _FakeGenerator()

        sample_and_generate(two_doc_config, n=4, candidates_path=candidates_path,
                             generator=generator, seed=1)
        sample_and_generate(two_doc_config, n=4, candidates_path=candidates_path,
                             generator=generator, seed=1)

        candidates = load_candidates(candidates_path)
        chunk_ids = [c.chunk_id for c in candidates]
        assert len(chunk_ids) == len(set(chunk_ids)), "resampled a chunk that was already recorded"

    def test_a_failing_generation_call_is_skipped_not_fatal(self, two_doc_config, tmp_path, capsys):
        candidates_path = tmp_path / "eval" / "candidates.jsonl"
        count = sample_and_generate(two_doc_config, n=2, candidates_path=candidates_path,
                                     generator=_FailingGenerator(), seed=1)
        assert count == 0
        assert load_candidates(candidates_path) == []
        assert "skipped" in capsys.readouterr().out


class TestGoldenSetAppendOnly:
    def test_creates_file_on_first_append(self, tmp_path):
        path = tmp_path / "golden_set.json"
        item = GoldenSetItem(id="q1", question="Q?", relevant_chunks=[], expected_answer="A.")

        GoldenSet.append_item(path, item)

        assert GoldenSet.load(path).items == [item]

    def test_append_adds_without_touching_existing_entries(self, tmp_path):
        path = tmp_path / "golden_set.json"
        first = GoldenSetItem(id="q1", question="Q1?", relevant_chunks=[], expected_answer="A1.")
        second = GoldenSetItem(id="q2", question="Q2?", relevant_chunks=[], expected_answer="A2.")

        GoldenSet.append_item(path, first)
        GoldenSet.append_item(path, second)

        items = GoldenSet.load(path).items
        assert items == [first, second]  # first entry unchanged, second appended after it

    def test_duplicate_id_raises_and_does_not_modify_file(self, tmp_path):
        path = tmp_path / "golden_set.json"
        item = GoldenSetItem(id="q1", question="Q?", relevant_chunks=[], expected_answer="A.")
        GoldenSet.append_item(path, item)
        before = path.read_text()

        dup = GoldenSetItem(id="q1", question="different question", relevant_chunks=[], expected_answer="x")
        with pytest.raises(ValueError):
            GoldenSet.append_item(path, dup)

        assert path.read_text() == before


class TestReviewLoop:
    def _make_candidates(self, tmp_path, n=3) -> Path:
        candidates_path = tmp_path / "eval" / "candidates.jsonl"
        candidates = [
            Candidate(
                candidate_id=f"cand-{i}", chunk_id=f"c{i}", doc_id="d", page=1,
                chunk_text=f"chunk text {i}", question=f"Q{i}?", expected_answer=f"A{i}.",
                generated_at="2026-01-01T00:00:00+00:00",
            )
            for i in range(n)
        ]
        save_candidates(candidates_path, candidates)
        return candidates_path

    def test_accept_edit_reject_skip_quit_flow(self, tmp_path):
        candidates_path = self._make_candidates(tmp_path, n=4)
        golden_path = tmp_path / "eval" / "golden_set.json"

        scripted = iter([
            "a", "",                                       # accept c0, default type
            "e", "Edited Q?", "Edited A.", "multi_hop",     # edit + accept c1
            "r",                                            # reject c2
            "q",                                             # quit before c3
        ])
        run_review(candidates_path, golden_path, input_func=lambda _prompt: next(scripted))

        golden = GoldenSet.load(golden_path)
        assert len(golden.items) == 2
        assert golden.items[0].question_type == "simple"
        assert golden.items[1] == GoldenSetItem(
            id="cand-1", question="Edited Q?",
            relevant_chunks=golden.items[1].relevant_chunks,
            expected_answer="Edited A.", question_type="multi_hop",
        )

        statuses = {c.candidate_id: c.status for c in load_candidates(candidates_path)}
        assert statuses == {
            "cand-0": "accepted", "cand-1": "accepted",
            "cand-2": "rejected", "cand-3": "pending",
        }

    def test_no_pending_candidates_is_a_no_op(self, tmp_path, capsys):
        candidates_path = self._make_candidates(tmp_path, n=1)
        golden_path = tmp_path / "eval" / "golden_set.json"

        # accept the only one: "a" for the decision, "" (default type) for the follow-up prompt
        first_pass = iter(["a", ""])
        run_review(candidates_path, golden_path, input_func=lambda _prompt: next(first_pass))
        before = load_candidates(candidates_path)

        run_review(candidates_path, golden_path, input_func=lambda _prompt: pytest.fail("should not prompt"))

        assert load_candidates(candidates_path) == before
        assert "No pending" in capsys.readouterr().out

    def test_invalid_question_type_is_reprompted(self, tmp_path):
        candidates_path = self._make_candidates(tmp_path, n=1)
        golden_path = tmp_path / "eval" / "golden_set.json"

        scripted = iter(["a", "not_a_real_type", "distractor"])
        run_review(candidates_path, golden_path, input_func=lambda _prompt: next(scripted))

        assert GoldenSet.load(golden_path).items[0].question_type == "distractor"
