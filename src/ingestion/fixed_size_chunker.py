"""Fixed-size Chunker: splits each page's text into overlapping character windows.

Chunks never cross a page boundary, so ChunkMetadata.page is always unambiguous.
Satisfies the Chunker protocol (src/ingestion/protocols.py); callers should depend
on that protocol, not on this class directly.
"""

from __future__ import annotations

from src.ingestion.models import Chunk, ChunkMetadata, ParsedDocument


class FixedSizeChunker:
    """Splits each page into overlapping windows of `chunk_size` characters.

    `chunk_overlap` characters are shared between consecutive windows on the same
    page. Empty/whitespace-only pages produce no chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        stride = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []

        for page in document.pages:
            text = page.text.strip()
            if not text:
                continue

            index = 0
            start = 0
            while start < len(text):
                window = text[start : start + self.chunk_size]
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}-p{page.page_number}-c{index}",
                        text=window,
                        metadata=ChunkMetadata(
                            doc_id=document.doc_id,
                            page=page.page_number,
                            section=page.section,
                            source_path=document.source_path,
                        ),
                    )
                )
                index += 1
                start += stride

        return chunks
