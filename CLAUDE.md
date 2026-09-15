# Project: RAG System with Rigorous Evaluation

## What this is
A portfolio-grade RAG system. The differentiator is the **evaluation harness**,
not the chatbot. Every architectural decision must be measurable.

## Non-negotiables
- Chunking, embedding, retrieval, and reranking are SWAPPABLE MODULES behind
  stable interfaces. Never hardcode a strategy.
- Every experiment is driven by a YAML config in `configs/`. Never edit code to
  run a different experiment.
- Retrieval metrics and generation metrics are computed and reported SEPARATELY.
- Chunk metadata (doc_id, page, section) travels with every vector. Citations
  depend on it.
- `eval/golden_set.json` is FROZEN once created. Never modify it to improve scores.

## Stack
- Python 3.11+, uv for deps
- Vector DB: Qdrant (local, Docker)
- Parsing: pymupdf + unstructured
- Eval: RAGAS (or custom — justify the choice)
- API: FastAPI
- Config: pydantic-settings + YAML

## Layout
data/raw/          # untouched source documents
data/processed/    # parsed + chunked output
configs/           # one YAML per experiment
src/ingestion/     # parsing, chunking
src/retrieval/     # embedding, vector store, reranking
src/generation/    # prompting, citation enforcement
src/eval/          # runner, metrics
eval/golden_set.json
results/           # experiment CSVs

## Working style
- Ask before adding a dependency.
- Write the interface/protocol before the implementation.
- No feature creep. If it's not in the current phase, don't build it.
