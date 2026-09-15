"""Tests for src/generation: citation extraction + OpenRouterGenerator (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.generation import Answer, Generator, OpenRouterGenerator, extract_citations
from src.ingestion.models import ChunkMetadata
from src.retrieval.models import RetrievedChunk


def _context() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="c1",
            text="Mitochondria produce ATP.",
            metadata=ChunkMetadata(doc_id="bio", page=1, source_path="bio.pdf"),
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Paris is in France.",
            metadata=ChunkMetadata(doc_id="geo", page=3, source_path="geo.pdf"),
            score=0.4,
        ),
    ]


def test_extract_citations_dedupes_valid_and_flags_hallucinated():
    context = _context()
    text = "ATP is produced by mitochondria [c1]. Also see [c1] again and [c99] which does not exist."

    citations, unsupported = extract_citations(text, context)

    assert [c.chunk_id for c in citations] == ["c1"]
    assert unsupported == ["c99"]


def test_openrouter_generator_builds_answer_from_mocked_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-test-key")
    context = _context()

    # `usage` explicitly set (not left as an auto-vivified MagicMock attribute):
    # MagicMock supports __float__ by default and returns 1.0, so an
    # unconfigured `.usage.cost` would silently produce a fake cost_usd=1.0
    # instead of the real "provider didn't report cost" None.
    message = MagicMock(content="Mitochondria produce ATP [c1]. Unrelated fact [c2].")
    response = MagicMock(choices=[MagicMock(message=message)], model="openai/gpt-4o-mini", usage=None)

    with patch("src.generation.openrouter_client.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = response
        generator = OpenRouterGenerator()
        assert isinstance(generator, Generator)
        answer = generator.generate("What do mitochondria do?", context)
        call_kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs

    assert isinstance(answer, Answer)
    assert {c.chunk_id for c in answer.citations} == {"c1", "c2"}
    assert answer.unsupported_citation_markers == []
    assert call_kwargs["model"] == "openai/gpt-4o-mini"
    assert call_kwargs["temperature"] == 0
    assert answer.model_used == "openai/gpt-4o-mini"
    assert answer.cost_usd is None
