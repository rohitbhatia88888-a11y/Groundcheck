"""Maps config `type` strings to concrete classes, so a YAML config selects an
implementation by name without calling code importing it directly.

CHUNKERS/EMBEDDERS/VECTOR_STORES/build_component are the SAME registry
src/ingestion/run.py uses standalone (single source of truth — previously
duplicated here with only one chunker registered, which meant a config could
select fixed_token/semantic/section_aware for the standalone ingestion CLI
but not for a full eval run; fixed now). RERANKERS/GENERATORS are eval-only,
since ingestion doesn't need them.
"""

from __future__ import annotations

from typing import NamedTuple

from src.eval.config import ExperimentConfig
from src.generation import Generator, OpenRouterGenerator
from src.ingestion import Chunker
from src.ingestion.run import CHUNKERS, EMBEDDERS, VECTOR_STORES, build_component
from src.retrieval import Embedder, IdentityReranker, Reranker, VectorStore

RERANKERS: dict[str, type[Reranker]] = {"identity": IdentityReranker}
GENERATORS: dict[str, type[Generator]] = {"openrouter": OpenRouterGenerator}


class Pipeline(NamedTuple):
    chunker: Chunker
    embedder: Embedder
    vector_store: VectorStore
    reranker: Reranker
    generator: Generator


def build_pipeline(config: ExperimentConfig) -> Pipeline:
    embedder = build_component(EMBEDDERS, config.embedder)

    # vector_size must match the embedder's output dimension; only fall back to
    # it when the config doesn't pin one explicitly.
    vector_store = build_component(VECTOR_STORES, config.vector_store, vector_size=embedder.dimension)

    return Pipeline(
        chunker=build_component(CHUNKERS, config.chunker),
        embedder=embedder,
        vector_store=vector_store,
        reranker=build_component(RERANKERS, config.reranker),
        generator=build_component(GENERATORS, config.generator),
    )
