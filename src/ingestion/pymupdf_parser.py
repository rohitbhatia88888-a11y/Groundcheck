"""PyMuPDF-backed DocumentParser: extracts page-level prose text (with
detected section headings) and tables, separately, from PDF files.

Heading detection is a font-size heuristic: any text line whose font size is
notably larger than the document's body-text size (the median span size
across the whole document) counts as a heading. Table regions are detected
first (pymupdf's built-in page.find_tables()) and excluded from prose
extraction, so a table's cell text never leaks into ParsedPage.text and its
values never get mistaken for headings.

Satisfies the DocumentParser protocol (src/ingestion/protocols.py); callers
should depend on that protocol, not on this class directly.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from src.ingestion.models import ExtractedTable, Heading, ParsedDocument, ParsedPage

_HEADING_SIZE_RATIO = 1.15  # a line at >=15% larger than body text counts as a heading
_DEFAULT_BODY_SIZE = 11.0  # fallback when a document has no extractable text at all
_TABLE_OVERLAP_THRESHOLD = 0.5  # a text block >=50% inside a table's bbox is table content


def _median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _body_font_size(page_dicts: list[dict]) -> float:
    sizes = [
        span["size"]
        for page_dict in page_dicts
        for block in page_dict["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if span["text"].strip()
    ]
    return _median(sizes) if sizes else _DEFAULT_BODY_SIZE


def _overlap_fraction(block_bbox: tuple[float, float, float, float], table_bbox) -> float:
    bx0, by0, bx1, by1 = block_bbox
    x0, y0 = max(bx0, table_bbox.x0), max(by0, table_bbox.y0)
    x1, y1 = min(bx1, table_bbox.x1), min(by1, table_bbox.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    block_area = (bx1 - bx0) * (by1 - by0)
    if block_area <= 0:
        return 0.0
    return ((x1 - x0) * (y1 - y0)) / block_area


def _is_table_block(block_bbox: tuple[float, float, float, float], table_bboxes: list) -> bool:
    return any(
        _overlap_fraction(block_bbox, tb) >= _TABLE_OVERLAP_THRESHOLD for tb in table_bboxes
    )


class PyMuPDFParser:
    """Parses a PDF into a ParsedDocument: prose pages (with headings) plus
    tables, extracted separately."""

    def parse(self, path: Path) -> ParsedDocument:
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PyMuPDFParser only handles .pdf files, got: {path}")

        with pymupdf.open(path) as pdf:
            page_dicts = [page.get_text("dict") for page in pdf]
            body_size = _body_font_size(page_dicts)

            pages: list[ParsedPage] = []
            tables: list[ExtractedTable] = []
            current_section: str | None = None

            for page_index in range(len(pdf)):
                page = pdf[page_index]
                page_number = page_index + 1
                section_at_start = current_section  # inherited, before this page's own headings

                found_tables = page.find_tables()
                table_bboxes = [pymupdf.Rect(t.bbox) for t in found_tables.tables]
                for table_index, table in enumerate(found_tables.tables):
                    tables.append(
                        ExtractedTable(
                            doc_id=path.stem,
                            page=page_number,
                            table_index=table_index,
                            source_path=str(path),
                            markdown=table.to_markdown(),
                        )
                    )

                text_parts: list[str] = []
                headings: list[Heading] = []
                offset = 0

                for block in page_dicts[page_index]["blocks"]:
                    if block.get("type") != 0:  # skip images and other non-text blocks
                        continue
                    if _is_table_block(block["bbox"], table_bboxes):
                        continue  # this text belongs to a table, handled above instead

                    for line in block["lines"]:
                        line_text = "".join(span["text"] for span in line["spans"]).strip()
                        if not line_text:
                            continue

                        max_size = max((span["size"] for span in line["spans"]), default=body_size)
                        if max_size >= body_size * _HEADING_SIZE_RATIO:
                            headings.append(Heading(text=line_text, char_offset=offset))
                            current_section = line_text

                        text_parts.append(line_text)
                        offset += len(line_text) + 1  # +1 for the "\n" joiner below

                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text="\n".join(text_parts),
                        section=section_at_start,
                        headings=headings,
                    )
                )

        return ParsedDocument(doc_id=path.stem, source_path=str(path), pages=pages, tables=tables)
