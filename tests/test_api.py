"""Tests for src/api/app.py: startup, streaming /query, citation reporting,
and rate limiting. Only the OpenRouter network call is mocked — everything
else (ingestion, embedding, retrieval) runs for real against a small
fixture PDF.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, limiter


def _wait_until_ready(timeout: float = 10.0) -> None:
    """Indexing now runs as a background task (see src/api/app.py) rather
    than blocking startup, so entering the TestClient context no longer
    guarantees it's finished — poll app.state.ready instead of assuming."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app.state.ready.is_set():
            return
        time.sleep(0.02)
    raise TimeoutError("app did not finish indexing within the test timeout")


def _fake_stream_chunks(text: str):
    for word in text.split(" "):
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content=word + " "))]
        yield chunk


def _fake_create(*args, **kwargs):
    assert kwargs.get("stream") is True, "the live API must always call the streaming path"
    return _fake_stream_chunks("Mitochondria produce ATP [bio-p1-c0].")


@pytest.fixture
def client(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    doc = pymupdf.open()
    doc.new_page().insert_textbox(
        pymupdf.Rect(72, 72, 523, 770), "Mitochondria produce ATP through cellular respiration."
    )
    doc.save(raw_dir / "bio.pdf")
    doc.close()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
name: api_test
raw_data_dir: {raw_dir}
golden_set_path: {tmp_path / "golden_set.json"}
chunker: {{type: fixed_size, params: {{chunk_size: 500, chunk_overlap: 50}}}}
embedder: {{type: sentence_transformers}}
vector_store: {{type: qdrant, params: {{collection_name: api_test, location: ':memory:'}}}}
reranker: {{type: identity}}
generator: {{type: openrouter}}
retrieval_top_k: 5
rerank_top_k: 5
""")

    monkeypatch.setenv("RAG_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-test-key")
    limiter.reset()  # isolate rate-limit state from other tests / test order

    with patch("src.generation.openrouter_client.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = _fake_create
        with TestClient(app) as test_client:
            _wait_until_ready()
            yield test_client


class TestReadiness:
    """Regression coverage for a real production failure: indexing used to
    run inside the blocking lifespan, so /health couldn't respond AT ALL
    until the whole corpus finished embedding — on a slow/throttled VM this
    took long enough that Fly's orchestrator kept restarting the machine
    before it ever got there. Indexing is now a background task; /health
    and /query must report clearly (503, not silence) while it's running.
    """

    def test_health_and_query_report_not_ready_before_indexing_completes(
        self, tmp_path, monkeypatch
    ):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        doc = pymupdf.open()
        doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 523, 770), "Some content.")
        doc.save(raw_dir / "doc.pdf")
        doc.close()

        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"""
name: readiness_test
raw_data_dir: {raw_dir}
golden_set_path: {tmp_path / "golden_set.json"}
chunker: {{type: fixed_size}}
embedder: {{type: sentence_transformers}}
vector_store: {{type: qdrant, params: {{collection_name: readiness_test, location: ':memory:'}}}}
reranker: {{type: identity}}
generator: {{type: openrouter}}
""")
        monkeypatch.setenv("RAG_CONFIG_PATH", str(config_path))
        monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-test-key")
        limiter.reset()

        with patch("src.generation.openrouter_client.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = _fake_create
            with TestClient(app) as test_client:
                # Immediately after startup: asyncio.create_task only
                # schedules the background ingest, it doesn't run it
                # synchronously — this is the real "not ready" window a
                # live deploy sits in until embedding finishes.
                assert not app.state.ready.is_set()

                health = test_client.get("/health")
                assert health.status_code == 503
                assert health.json()["status"] == "indexing"

                query = test_client.post("/query", json={"question": "anything"})
                assert query.status_code == 503

                _wait_until_ready()
                assert test_client.get("/health").json() == {"status": "ok"}


def _read_ndjson(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


class TestHealthAndIndex:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_index_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "RAG Query" in response.text


class TestQuery:
    def test_streams_deltas_then_done_with_correct_citations(self, client):
        response = client.post("/query", json={"question": "What do mitochondria produce?"})
        assert response.status_code == 200

        events = _read_ndjson(response)
        deltas = [e for e in events if e["type"] == "delta"]
        done_events = [e for e in events if e["type"] == "done"]

        assert len(deltas) > 0
        assert "".join(e["text"] for e in deltas) == "Mitochondria produce ATP [bio-p1-c0]. "

        assert len(done_events) == 1
        done = done_events[0]
        assert done["citations"] == [{"chunk_id": "bio-p1-c0", "doc_id": "bio", "page": 1}]
        assert done["unsupported_citation_markers"] == []

        assert len(done["context"]) == 1
        assert done["context"][0]["chunk_id"] == "bio-p1-c0"
        assert done["context"][0]["cited"] is True
        assert "Mitochondria produce ATP" in done["context"][0]["text"]

    def test_rejects_empty_question(self, client):
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_rejects_missing_question(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422


class TestRateLimiting:
    def test_11th_request_within_a_minute_is_rejected(self, client):
        for i in range(10):
            response = client.post("/query", json={"question": f"question {i}"})
            assert response.status_code == 200, f"request {i} should succeed"

        response = client.post("/query", json={"question": "one too many"})
        assert response.status_code == 429
