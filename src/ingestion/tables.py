"""Converts an extracted table into an indexable Chunk.

Tables are handled separately from prose: they are never fed through a
Chunker (a table is already a natural, atomic unit — splitting it would
break its structure), and their Chunk is tagged
`ChunkMetadata.content_type == "table"` so retrieval/eval can tell prose and
table hits apart.
"""

from __future__ import annotations

from src.ingestion.models import Chunk, ChunkMetadata, ExtractedTable


def table_to_chunk(table: ExtractedTable) -> Chunk:
    return Chunk(
        chunk_id=f"{table.doc_id}-p{table.page}-table{table.table_index}",
        text=table.markdown,
        metadata=ChunkMetadata(
            doc_id=table.doc_id,
            page=table.page,
            section="table",
            source_path=table.source_path,
            content_type="table",
        ),
    )
