"""Tests for src/api/app.py: startup, streaming /query, citation reporting,
and rate limiting. Only the OpenRouter network call is mocked — everything
else (ingestion, embedding, retrieval) runs for real against a small
fixture PDF.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, limiter


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
            yield test_client


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
