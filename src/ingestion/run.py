"""CLI: parse raw PDFs, extract tables, chunk prose, embed, and index into
Qdrant — the ingestion half of an experiment, runnable on its own.

Usage:
    python -m src.ingestion.run --config configs/baseline.yaml

Reads only the chunker/embedder/vector_store sections of the experiment
config; reranker/generator are irrelevant here and ignored. Run `make eval`
for the full retrieve+rerank+generate+score pipeline instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.eval.config import ComponentConfig, ExperimentConfig
from src.ingestion import (
    Chunk,
    FixedSizeChunker,
    FixedTokenChunker,
    PyMuPDFParser,
    SectionAwareChunker,
    SemanticChunker,
    table_to_chunk,
)
from src.retrieval import QdrantVectorStore, SentenceTransformersEmbedder

# Local to ingestion, deliberately not shared with src/eval/registry.py's
# registries (see CLAUDE.md: this phase doesn't touch retrieval/generation
# wiring) — keep both in sync by hand if a new chunker/embedder/store is added.
CHUNKERS: dict[str, type] = {
    "fixed_size": FixedSizeChunker,
    "fixed_token": FixedTokenChunker,
    "semantic": SemanticChunker,
    "section_aware": SectionAwareChunker,
}
EMBEDDERS: dict[str, type] = {"sentence_transformers": SentenceTransformersEmbedder}
VECTOR_STORES: dict[str, type] = {"qdrant": QdrantVectorStore}


def _build(registry: dict[str, type], config: ComponentConfig, **extra_params: Any) -> Any:
    try:
        cls = registry[config.type]
    except KeyError:
        raise ValueError(f"Unknown type '{config.type}'. Available: {sorted(registry)}") from None
    return cls(**{**extra_params, **config.params})


def ingest(config: ExperimentConfig) -> int:
    """Parses+chunks+embeds+indexes every *.pdf under config.raw_data_dir.
    Returns the number of chunks indexed (prose + table)."""
    parser = PyMuPDFParser()
    chunker = _build(CHUNKERS, config.chunker)
    embedder = _build(EMBEDDERS, config.embedder)
    vector_store = _build(VECTOR_STORES, config.vector_store, vector_size=embedder.dimension)

    raw_dir = Path(config.raw_data_dir)
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.glob("**/*.pdf")):
        document = parser.parse(path)
        chunks.extend(chunker.chunk(document))
        chunks.extend(table_to_chunk(t) for t in document.tables)

    embedded = embedder.embed_chunks(chunks)
    vector_store.upsert(embedded)
    return len(embedded)


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="Parse, chunk, and index an experiment's raw PDFs into Qdrant."
    )
    arg_parser.add_argument(
        "--config", required=True, help="Path to an experiment YAML (configs/*.yaml)"
    )
    args = arg_parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    count = ingest(config)
    collection = config.vector_store.params.get("collection_name", "?")
    print(f"Indexed {count} chunks from '{config.raw_data_dir}' into collection '{collection}'.")


if __name__ == "__main__":
    main()
