"""Maps config `type` strings to concrete classes, so a YAML config selects an
implementation by name without calling code importing it directly. Adding a
new implementation means adding one line here, not touching the runner.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from src.eval.config import ComponentConfig, ExperimentConfig
from src.generation import Generator, OpenRouterGenerator
from src.ingestion import Chunker, FixedSizeChunker
from src.retrieval import (
    Embedder,
    IdentityReranker,
    QdrantVectorStore,
    Reranker,
    SentenceTransformersEmbedder,
    VectorStore,
)

# Parsing isn't a swappable module (see CLAUDE.md non-negotiables — only
# chunking/embedding/retrieval/reranking are); PyMuPDFParser is called
# directly in src/eval/runner.py instead of going through this registry.
CHUNKERS: dict[str, type[Chunker]] = {"fixed_size": FixedSizeChunker}
EMBEDDERS: dict[str, type[Embedder]] = {"sentence_transformers": SentenceTransformersEmbedder}
VECTOR_STORES: dict[str, type[VectorStore]] = {"qdrant": QdrantVectorStore}
RERANKERS: dict[str, type[Reranker]] = {"identity": IdentityReranker}
GENERATORS: dict[str, type[Generator]] = {"openrouter": OpenRouterGenerator}


def _build(registry: dict[str, type], config: ComponentConfig, **extra_params: Any) -> Any:
    try:
        cls = registry[config.type]
    except KeyError:
        raise ValueError(
            f"Unknown type '{config.type}'. Available: {sorted(registry)}"
        ) from None
    return cls(**{**extra_params, **config.params})


class Pipeline(NamedTuple):
    chunker: Chunker
    embedder: Embedder
    vector_store: VectorStore
    reranker: Reranker
    generator: Generator


def build_pipeline(config: ExperimentConfig) -> Pipeline:
    embedder = _build(EMBEDDERS, config.embedder)

    # vector_size must match the embedder's output dimension; only fall back to
    # it when the config doesn't pin one explicitly.
    vector_store = _build(VECTOR_STORES, config.vector_store, vector_size=embedder.dimension)

    return Pipeline(
        chunker=_build(CHUNKERS, config.chunker),
        embedder=embedder,
        vector_store=vector_store,
        reranker=_build(RERANKERS, config.reranker),
        generator=_build(GENERATORS, config.generator),
    )
