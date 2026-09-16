"""FastAPI serving layer: streams answers from a live RAG pipeline, built
once at startup from one experiment config selected via the RAG_CONFIG_PATH
env var — so a deploy can point at any configs/*.yaml (baseline, hybrid,
...) without a code change, same as every other entry point in this project.

Usage:
    RAG_CONFIG_PATH=configs/hybrid.yaml uv run uvicorn src.api.app:app

See DEPLOY.md for Fly.io deployment; Dockerfile at the repo root.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from collections.abc import Generator as GeneratorType
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.schemas import QueryRequest
from src.eval.config import ExperimentConfig
from src.eval.registry import Pipeline, build_pipeline
from src.eval.runner import ingest_raw_documents
from src.generation import StreamingGenerator, extract_citations
from src.retrieval.models import RetrievedChunk

DEFAULT_CONFIG_PATH = "configs/baseline.yaml"
STATIC_DIR = Path(__file__).parent / "static"

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config_path = os.environ.get("RAG_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    config = ExperimentConfig.from_yaml(config_path)
    pipeline = build_pipeline(config)

    if not isinstance(pipeline.generator, StreamingGenerator):
        # RuntimeError, not TypeError: this is a deploy/config misconfiguration
        # (the wrong generator type in configs/*.yaml for a live deploy), not a
        # plain Python type error.
        raise RuntimeError(  # noqa: TRY004
            f"generator type '{config.generator.type}' (from {config_path}) does not "
            f"support streaming (no generate_stream method) — the live API requires "
            f"a StreamingGenerator. OpenRouterGenerator ('openrouter') does."
        )

    # Re-embeds+re-indexes raw_data_dir fresh at every startup — simple and
    # correct (no stale-index risk), fine for this corpus size. See
    # DEPLOY.md for the tradeoff on larger corpora / cold-start latency.
    ingest_raw_documents(config, pipeline)

    app.state.config = config
    app.state.pipeline = pipeline
    app.state.config_path = str(config_path)
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _ndjson(obj: dict[str, Any]) -> str:
    return json.dumps(obj) + "\n"


def _stream_answer(
    pipeline: Pipeline, question: str, reranked: list[RetrievedChunk]
) -> GeneratorType[str, None, None]:
    parts: list[str] = []
    try:
        for delta in pipeline.generator.generate_stream(question, reranked):
            parts.append(delta)
            yield _ndjson({"type": "delta", "text": delta})
    except Exception as exc:  # noqa: BLE001 - surface to the client rather than hang up silently
        yield _ndjson({"type": "error", "message": str(exc)})
        return

    full_text = "".join(parts)
    citations, unsupported = extract_citations(full_text, reranked)
    cited_ids = {c.chunk_id for c in citations}

    yield _ndjson(
        {
            "type": "done",
            "citations": [c.model_dump() for c in citations],
            "unsupported_citation_markers": unsupported,
            "context": [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.metadata.doc_id,
                    "page": chunk.metadata.page,
                    "text": chunk.text,
                    "score": chunk.score,
                    "cited": chunk.chunk_id in cited_ids,
                }
                for chunk in reranked
            ],
        }
    )


@app.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest) -> StreamingResponse:
    pipeline: Pipeline = request.app.state.pipeline
    config: ExperimentConfig = request.app.state.config

    query_vector = pipeline.embedder.embed_query(body.question)
    retrieved = pipeline.vector_store.query(
        query_vector, top_k=config.retrieval_top_k, query_text=body.question
    )
    reranked = pipeline.reranker.rerank(body.question, retrieved, top_k=config.rerank_top_k)

    return StreamingResponse(
        _stream_answer(pipeline, body.question, reranked),
        media_type="application/x-ndjson",
    )
