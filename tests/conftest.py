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
