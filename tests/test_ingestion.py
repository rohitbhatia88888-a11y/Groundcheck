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


def test_parser_detects_headings_and_carries_section_across_pages(structured_pdf):
    parsed = PyMuPDFParser().parse(structured_pdf)

    page1, page2 = parsed.pages
    assert page1.section is None  # nothing precedes page 1
    assert [h.text for h in page1.headings] == ["Introduction"]
    assert page1.headings[0].char_offset == 0  # heading is the first line on the page

    assert page2.section == "Introduction"  # inherited from page 1's heading
    assert [h.text for h in page2.headings] == ["Methodology"]
    # the offset must land exactly on "Methodology" in this page's own text
    offset = page2.headings[0].char_offset
    assert page2.text[offset : offset + len("Methodology")] == "Methodology"


def test_parser_extracts_table_separately_from_prose(structured_pdf):
    parsed = PyMuPDFParser().parse(structured_pdf)

    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert table.page == 1
    assert table.table_index == 0
    assert "Accuracy" in table.markdown
    assert "0.95" in table.markdown

    # table cell text must not leak into the page's prose text
    assert "Metric" not in parsed.pages[0].text
    assert "Accuracy" not in parsed.pages[0].text
