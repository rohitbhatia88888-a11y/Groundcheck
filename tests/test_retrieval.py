"""Tests for src/retrieval: Embedder -> VectorStore -> Reranker."""

from __future__ import annotations

from typing import ClassVar

import pytest

from src.ingestion.models import Chunk, ChunkMetadata
from src.retrieval import (
    CrossEncoderReranker,
    Embedder,
    HybridVectorStore,
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


class TestHybridVectorStore:
    _CORPUS: ClassVar[list[Chunk]] = [
        _chunk("c1", "The ZX9942-Regulation requires quarterly audits of financial institutions.", "reg", 1),
        _chunk("c2", "Bananas are a good source of potassium and fiber.", "food", 1),
        _chunk("c3", "The stock market fluctuated significantly during the third quarter.", "fin", 1),
    ]

    def _indexed_store(self, dense_weight: float) -> tuple[HybridVectorStore, SentenceTransformersEmbedder]:
        embedder = SentenceTransformersEmbedder()
        store = HybridVectorStore(
            collection_name=f"hybrid-test-{dense_weight}", vector_size=embedder.dimension,
            dense_weight=dense_weight,
        )
        store.upsert(embedder.embed_chunks(self._CORPUS))
        return store, embedder

    def test_satisfies_protocol(self):
        store, _ = self._indexed_store(dense_weight=0.5)
        assert isinstance(store, VectorStore)

    def test_pure_bm25_finds_exact_term_match(self):
        # dense_weight=0.0: BM25 only. A made-up identifier like
        # "ZX9942-Regulation" has no reason to embed distinctively, but BM25
        # must match it exactly — this is the whole point of hybrid retrieval
        # (see module docstring: catches what baseline's dense-only retrieval
        # missed on named-entity/identifier queries in the real eval).
        store, embedder = self._indexed_store(dense_weight=0.0)
        query = "What does the ZX9942-Regulation require?"

        results = store.query(embedder.embed_query(query), top_k=3, query_text=query)

        assert results[0].chunk_id == "c1"
        assert results[0].score > 0

    def test_pure_dense_matches_plain_qdrant_ranking(self):
        # dense_weight=1.0 should rank identically to a bare QdrantVectorStore
        # over the same corpus/query — an invariant that doesn't depend on
        # guessing what the embedding model "should" consider similar.
        embedder = SentenceTransformersEmbedder()
        query = "What does the ZX9942-Regulation require?"
        query_vector = embedder.embed_query(query)

        plain_store = QdrantVectorStore(collection_name="plain-compare", vector_size=embedder.dimension)
        plain_store.upsert(embedder.embed_chunks(self._CORPUS))
        plain_order = [r.chunk_id for r in plain_store.query(query_vector, top_k=3)]

        hybrid_store = HybridVectorStore(
            collection_name="hybrid-compare", vector_size=embedder.dimension, dense_weight=1.0
        )
        hybrid_store.upsert(embedder.embed_chunks(self._CORPUS))
        hybrid_order = [r.chunk_id for r in hybrid_store.query(query_vector, top_k=3, query_text=query)]

        assert hybrid_order == plain_order

    def test_requires_query_text(self):
        store, embedder = self._indexed_store(dense_weight=0.5)
        with pytest.raises(ValueError):
            store.query(embedder.embed_query("anything"), top_k=3, query_text=None)

    def test_empty_store_returns_empty(self):
        embedder = SentenceTransformersEmbedder()
        store = HybridVectorStore(collection_name="empty-hybrid", vector_size=embedder.dimension)
        assert store.query(embedder.embed_query("q"), top_k=3, query_text="q") == []

    def test_rejects_dense_weight_out_of_range(self):
        with pytest.raises(ValueError):
            HybridVectorStore(collection_name="bad", vector_size=384, dense_weight=1.5)
