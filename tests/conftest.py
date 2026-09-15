"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def make_pdf(tmp_path):
    """Factory fixture: make_pdf(name, texts) writes a PDF with one page per
    string in `texts` (wrapped via insert_textbox, not clipped) and returns
    its path. An empty/whitespace string produces a genuinely blank page."""

    def _make(name: str, texts: list[str]) -> Path:
        doc = pymupdf.open()
        rect = pymupdf.Rect(72, 72, 523, 770)
        for text in texts:
            page = doc.new_page()
            if text.strip():
                page.insert_textbox(rect, text)
        path = tmp_path / name
        doc.save(path)
        doc.close()
        return path

    return _make


@pytest.fixture
def structured_pdf(tmp_path) -> Path:
    """A small 2-page fixture PDF exercising every parser feature at once:
    a heading (large font) that starts a new section, wrapped prose at body
    size, and a bordered table — on one page, plus a second page whose prose
    inherits the first page's section before its own new heading appears.
    Used by the chunker and parser tests.
    """
    doc = pymupdf.open()

    page1 = doc.new_page()
    page1.insert_text((72, 90), "Introduction", fontsize=20)
    page1.insert_textbox(
        pymupdf.Rect(72, 110, 523, 300),
        "This document describes the system. " * 15,
        fontsize=11,
    )
    x0, y0, x1, y1 = 72, 400, 300, 480
    page1.draw_rect(pymupdf.Rect(x0, y0, x1, y1))
    page1.draw_line((x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2))
    page1.draw_line(((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1))
    page1.insert_text((80, 430), "Metric")
    page1.insert_text((200, 430), "Value")
    page1.insert_text((80, 470), "Accuracy")
    page1.insert_text((200, 470), "0.95")

    page2 = doc.new_page()
    page2.insert_textbox(
        pymupdf.Rect(72, 72, 523, 150),
        "More introduction text continues here. " * 5,
        fontsize=11,
    )
    page2.insert_text((72, 200), "Methodology", fontsize=20)
    page2.insert_textbox(
        pymupdf.Rect(72, 220, 523, 400),
        "We used a rigorous evaluation harness. " * 10,
        fontsize=11,
    )

    path = tmp_path / "structured.pdf"
    doc.save(path)
    doc.close()
    return path
