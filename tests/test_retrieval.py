"""Tests for src/retrieval: Embedder -> VectorStore -> Reranker."""

from __future__ import annotations

from src.ingestion.models import Chunk, ChunkMetadata
from src.retrieval import (
    CrossEncoderReranker,
    Embedder,
    IdentityReranker,
    QdrantVectorStore,
    Reranker,
    SentenceTransformersEmbedder,
    VectorStore,
)
from src.retrieval.models import RetrievedChunk


def _chunk(chunk_id: str, text: str, doc_id: str, page: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(doc_id=doc_id, page=page, source_path=f"{doc_id}.pdf"),
    )


def _retrieved(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, score=score,
        metadata=ChunkMetadata(doc_id="d", page=1, source_path="d.pdf"),
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


def test_cross_encoder_reranker_reorders_by_true_relevance_not_bi_encoder_score():
    # c2 has the highest incoming (bi-encoder) score but is irrelevant to the
    # query; a working cross-encoder must demote it regardless of that score.
    chunks = [
        _retrieved("c1", "Paris is the capital of France.", score=0.5),
        _retrieved("c2", "The mitochondria is the powerhouse of the cell.", score=0.9),
        _retrieved("c3", "France, officially the French Republic, has Paris as its capital.", score=0.4),
    ]

    reranker = CrossEncoderReranker()
    assert isinstance(reranker, Reranker)

    result = reranker.rerank("What is the capital of France?", chunks, top_k=2)

    assert len(result) == 2
    assert {r.chunk_id for r in result} == {"c1", "c3"}
    assert result[0].score >= result[1].score  # sorted descending by the new score


def test_cross_encoder_reranker_handles_empty_input():
    assert CrossEncoderReranker().rerank("q", [], top_k=5) == []
