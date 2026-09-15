"""Tests for src/ingestion: PyMuPDFParser -> FixedSizeChunker."""

from __future__ import annotations

import pytest

from src.ingestion import Chunker, DocumentParser, FixedSizeChunker, PyMuPDFParser


def test_pymupdf_parser_extracts_pages_and_rejects_non_pdf(make_pdf, tmp_path):
    pdf_path = make_pdf("sample.pdf", ["Page one content.", "Page two content."])
    parser = PyMuPDFParser()
    assert isinstance(parser, DocumentParser)

    parsed = parser.parse(pdf_path)
    assert parsed.doc_id == "sample"
    assert [p.page_number for p in parsed.pages] == [1, 2]
    assert "Page one content." in parsed.pages[0].text

    with pytest.raises(ValueError):
        parser.parse(tmp_path / "notes.txt")


def test_fixed_size_chunker_splits_with_overlap_and_skips_blank_pages(make_pdf):
    long_text = "Lorem ipsum dolor sit amet. " * 40
    pdf_path = make_pdf("sample.pdf", [long_text, "Short page two.", "   "])

    parsed = PyMuPDFParser().parse(pdf_path)
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=50)
    assert isinstance(chunker, Chunker)

    chunks = chunker.chunk(parsed)

    assert all(c.metadata.page != 3 for c in chunks)  # blank page produced nothing
    page1_chunks = [c for c in chunks if c.metadata.page == 1]
    assert len(page1_chunks) > 1
    assert page1_chunks[0].text[-50:] == page1_chunks[1].text[:50]  # overlap holds


def test_fixed_size_chunker_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, chunk_overlap=100)
