"""PyMuPDF-backed DocumentParser: extracts page-level text from PDF files.

Text extraction only — no section/heading detection, so ParsedPage.section is
always None here. Satisfies the DocumentParser protocol (src/ingestion/protocols.py);
callers should depend on that protocol, not on this class directly.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from src.ingestion.models import ParsedDocument, ParsedPage


class PyMuPDFParser:
    """Parses a PDF into a ParsedDocument, one ParsedPage per PDF page."""

    def parse(self, path: Path) -> ParsedDocument:
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PyMuPDFParser only handles .pdf files, got: {path}")

        with pymupdf.open(path) as pdf:
            pages = [
                ParsedPage(page_number=index + 1, text=page.get_text())
                for index, page in enumerate(pdf)
            ]

        return ParsedDocument(
            doc_id=path.stem,
            source_path=str(path),
            pages=pages,
        )
