"""Tests for src/retrieval: Embedder -> VectorStore -> Reranker."""

from __future__ import annotations

from src.ingestion.models import Chunk, ChunkMetadata
from src.retrieval import (
    Embedder,
    IdentityReranker,
    QdrantVectorStore,
    Reranker,
    SentenceTransformersEmbedder,
    VectorStore,
)


def _chunk(chunk_id: str, text: str, doc_id: str, page: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(doc_id=doc_id, page=page, source_path=f"{doc_id}.pdf"),
    )


def test_embed_store_query_rerank_pipeline():
    chunks = [
        _chunk("c1", "The mitochondria is the powerhouse of the cell.", "bio", 1),
        _chunk("c2", "Paris is the capital of France.", "geo", 3),
        _chunk("c3", "Mitochondrial DNA is inherited maternally.", "bio", 2),
    ]

    embedder = SentenceTransformersEmbedder()
    assert isinstance(embedder, Embedder)
    embedded = embedder.embed_chunks(chunks)
    assert all(len(e.vector) == embedder.dimension for e in embedded)

    store = QdrantVectorStore(collection_name="test", vector_size=embedder.dimension)
    assert isinstance(store, VectorStore)
    store.upsert(embedded)
    store.upsert(embedded)  # re-upserting the same chunk_ids must not duplicate points

    results = store.query(embedder.embed_query("What does the mitochondria do?"), top_k=3)
    assert len(results) == 3
    assert results[0].chunk_id == "c1"

    reranker = IdentityReranker()
    assert isinstance(reranker, Reranker)
    assert reranker.rerank("what does the mitochondria do?", results, top_k=2) == results[:2]


def test_sentence_transformers_embedder_handles_empty_input():
    assert SentenceTransformersEmbedder().embed_chunks([]) == []
