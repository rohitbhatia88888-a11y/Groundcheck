"""Section-aware Chunker: splits each page's text at its detected heading
boundaries (ParsedPage.headings), so every chunk maps to exactly one section
instead of a fixed size or a similarity breakpoint.

A page with no headings becomes one chunk, tagged with its inherited ambient
section (ParsedPage.section). Chunks never cross a page boundary, so
ChunkMetadata.page stays unambiguous. Satisfies the Chunker protocol
(src/ingestion/protocols.py).
"""

from __future__ import annotations

from src.ingestion.models import Chunk, ChunkMetadata, ParsedDocument, ParsedPage


class SectionAwareChunker:
    """One chunk per section-segment on a page: content before the page's
    first heading is tagged with the page's inherited section; content after
    each heading is tagged with that heading's text."""

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []

        for page in document.pages:
            if not page.text.strip():
                continue

            for index, (section, segment_text) in enumerate(self._segments(page)):
                segment_text = segment_text.strip()
                if not segment_text:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}-p{page.page_number}-sec{index}",
                        text=segment_text,
                        metadata=ChunkMetadata(
                            doc_id=document.doc_id,
                            page=page.page_number,
                            section=section,
                            source_path=document.source_path,
                        ),
                    )
                )

        return chunks

    @staticmethod
    def _segments(page: ParsedPage) -> list[tuple[str | None, str]]:
        """Splits page.text at each heading's char_offset, returning
        [(section_label, segment_text), ...] — one entry per heading, plus one
        leading entry (page.section) for any text before the first heading."""
        text = page.text
        cut_points = [0, *(h.char_offset for h in page.headings), len(text)]
        labels: list[str | None] = [page.section, *(h.text for h in page.headings)]

        return [
            (labels[i], text[cut_points[i] : cut_points[i + 1]]) for i in range(len(labels))
        ]
