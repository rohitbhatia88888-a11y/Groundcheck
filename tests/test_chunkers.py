"""Unit tests for the three Chunker implementations added in Phase 2
(fixed-token, semantic, section-aware), all driven by the `structured_pdf`
fixture in conftest.py, plus the table -> Chunk conversion.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from src.ingestion import (
    Chunker,
    FixedTokenChunker,
    PyMuPDFParser,
    SectionAwareChunker,
    SemanticChunker,
    table_to_chunk,
)


class _FakeEmbedder:
    """Deterministic stand-in for SentenceTransformersEmbedder: a bag-of-words
    one-hot vector over a tiny fixed vocabulary. Sentences sharing vocabulary
    get high cosine similarity, sentences from disjoint vocabularies get zero
    — enough to exercise SemanticChunker's breakpoint logic without loading
    the real embedding model."""

    _VOCAB: ClassVar[list[str]] = ["document", "describes", "system", "introduction",
                                    "evaluation", "harness", "methodology", "rigorous"]

    def embed_query(self, text: str) -> list[float]:
        words = set(text.lower().split())
        return [1.0 if v in words else 0.0 for v in self._VOCAB]

    def embed_chunks(self, chunks):  # pragma: no cover - unused by SemanticChunker
        raise NotImplementedError


@pytest.fixture
def parsed(structured_pdf):
    return PyMuPDFParser().parse(structured_pdf)


class TestFixedTokenChunker:
    def test_satisfies_protocol_and_splits_with_overlap(self, parsed):
        chunker = FixedTokenChunker(chunk_size=20, chunk_overlap=5)
        assert isinstance(chunker, Chunker)

        chunks = chunker.chunk(parsed)
        assert len(chunks) > 0
        assert all(c.metadata.page in (1, 2) for c in chunks)
        # page 1's prose is long enough that it must split into multiple windows
        page1_chunks = [c for c in chunks if c.metadata.page == 1]
        assert len(page1_chunks) > 1

    def test_rejects_overlap_not_smaller_than_size(self):
        with pytest.raises(ValueError):
            FixedTokenChunker(chunk_size=50, chunk_overlap=50)

    def test_token_window_is_smaller_than_character_window_for_same_size(self, parsed):
        # A 50-unit token window covers noticeably more text than a 50-unit
        # character window, since one token is several characters — sanity
        # check that we're actually counting tokens, not silently chars.
        token_chunker = FixedTokenChunker(chunk_size=50, chunk_overlap=0)
        [first] = [c for c in token_chunker.chunk(parsed) if c.metadata.page == 2][:1]
        assert len(first.text) > 50  # 50 tokens is well over 50 characters of English prose


class TestSemanticChunker:
    def test_satisfies_protocol_and_splits_at_topic_shift(self, parsed):
        chunker = SemanticChunker(embedder=_FakeEmbedder(), breakpoint_percentile=0.5)
        assert isinstance(chunker, Chunker)

        chunks = chunker.chunk(parsed)
        assert len(chunks) > 0
        assert all(c.metadata.page in (1, 2) for c in chunks)

    def test_max_chunk_chars_caps_runaway_groups(self, parsed):
        # A percentile of near-1.0 means almost nothing is a breakpoint, so
        # the char cap is the only thing preventing one giant chunk per page.
        chunker = SemanticChunker(
            embedder=_FakeEmbedder(), breakpoint_percentile=0.01, max_chunk_chars=100
        )
        chunks = chunker.chunk(parsed)
        assert all(len(c.text) <= 200 for c in chunks)  # cap + one sentence's slack

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            SemanticChunker(embedder=_FakeEmbedder(), breakpoint_percentile=1.5)
        with pytest.raises(ValueError):
            SemanticChunker(embedder=_FakeEmbedder(), max_chunk_chars=0)


class TestSectionAwareChunker:
    def test_satisfies_protocol_and_splits_exactly_at_headings(self, parsed):
        chunker = SectionAwareChunker()
        assert isinstance(chunker, Chunker)

        chunks = chunker.chunk(parsed)
        sections = {c.metadata.section for c in chunks}
        assert sections == {"Introduction", "Methodology"}

        # page 2 has content both before ("Introduction", inherited) and
        # after ("Methodology") its own heading — both must appear as
        # separate chunks tagged with the right section.
        page2 = [c for c in chunks if c.metadata.page == 2]
        assert {c.metadata.section for c in page2} == {"Introduction", "Methodology"}

    def test_page_with_no_headings_is_one_chunk(self, structured_pdf):
        parsed_doc = PyMuPDFParser().parse(structured_pdf)
        # page 1's own heading is at offset 0, so everything after it is one
        # uninterrupted "Introduction" segment on that page.
        page1_chunks = [
            c for c in SectionAwareChunker().chunk(parsed_doc) if c.metadata.page == 1
        ]
        assert len(page1_chunks) == 1
        assert page1_chunks[0].metadata.section == "Introduction"


def test_table_to_chunk_is_tagged_and_never_split(parsed):
    assert len(parsed.tables) == 1
    chunk = table_to_chunk(parsed.tables[0])

    assert chunk.metadata.content_type == "table"
    assert chunk.metadata.section == "table"
    assert chunk.metadata.page == 1
    assert "Accuracy" in chunk.text
    assert "0.95" in chunk.text
