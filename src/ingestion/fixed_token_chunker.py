"""Token-based Chunker: splits each page's text into overlapping windows of
`chunk_size` tokens (tiktoken cl100k_base), rather than raw characters —
useful for sizing chunks against an LLM's actual context budget instead of a
character-count proxy.

Chunks never cross a page boundary, so ChunkMetadata.page stays unambiguous.
Satisfies the Chunker protocol (src/ingestion/protocols.py).
"""

from __future__ import annotations

import tiktoken

from src.ingestion._windowing import sliding_windows, validate_window_params
from src.ingestion.models import Chunk, ChunkMetadata, ParsedDocument

_ENCODING_NAME = "cl100k_base"


class FixedTokenChunker:
    """Splits each page into overlapping windows of `chunk_size` tokens.

    `chunk_overlap` tokens are shared between consecutive windows on the same
    page. Empty/whitespace-only pages produce no chunks. The encoding is a
    fixed token-counting proxy (cl100k_base) — it doesn't need to match
    whatever model actually embeds or generates downstream.
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32) -> None:
        validate_window_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoding = tiktoken.get_encoding(_ENCODING_NAME)

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []

        for page in document.pages:
            text = page.text.strip()
            if not text:
                continue

            tokens = self._encoding.encode(text)
            for index, (start, end) in enumerate(
                sliding_windows(len(tokens), self.chunk_size, self.chunk_overlap)
            ):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}-p{page.page_number}-t{index}",
                        text=self._encoding.decode(tokens[start:end]),
                        metadata=ChunkMetadata(
                            doc_id=document.doc_id,
                            page=page.page_number,
                            section=page.section,
                            source_path=document.source_path,
                        ),
                    )
                )

        return chunks
