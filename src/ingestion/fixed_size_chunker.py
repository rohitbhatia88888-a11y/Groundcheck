"""Fixed-size Chunker: splits each page's text into overlapping character windows.

Chunks never cross a page boundary, so ChunkMetadata.page is always unambiguous.
Satisfies the Chunker protocol (src/ingestion/protocols.py); callers should depend
on that protocol, not on this class directly.
"""

from __future__ import annotations

from src.ingestion._windowing import sliding_windows, validate_window_params
from src.ingestion.models import Chunk, ChunkMetadata, ParsedDocument


class FixedSizeChunker:
    """Splits each page into overlapping windows of `chunk_size` characters.

    `chunk_overlap` characters are shared between consecutive windows on the same
    page. Empty/whitespace-only pages produce no chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        validate_window_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []

        for page in document.pages:
            text = page.text.strip()
            if not text:
                continue

            for index, (start, end) in enumerate(
                sliding_windows(len(text), self.chunk_size, self.chunk_overlap)
            ):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}-p{page.page_number}-c{index}",
                        text=text[start:end],
                        metadata=ChunkMetadata(
                            doc_id=document.doc_id,
                            page=page.page_number,
                            section=page.section,
                            source_path=document.source_path,
                        ),
                    )
                )

        return chunks
