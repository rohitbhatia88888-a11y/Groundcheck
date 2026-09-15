"""Shared OpenRouter client construction, used by both the Generator
(src/generation/openrouter_generator.py) and the eval-time LLM judge
(src/eval/generation_metrics.py) — OpenRouter is the only LLM provider this
project talks to (see CLAUDE.md); it exposes an OpenAI-compatible API.
"""

from __future__ import annotations

import os

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_openrouter_client() -> OpenAI:
    """Builds an OpenRouter client from OPENROUTER_API_KEY in the environment.

    The key is never hardcoded here or anywhere else in this codebase — set it
    yourself with `export OPENROUTER_API_KEY=...` or in a local, gitignored
    `.env` file (see .env.example).
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Set it in your shell "
            "(export OPENROUTER_API_KEY=...) or a local, gitignored .env file "
            "— see .env.example."
        )
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def extract_cost_usd(response) -> float | None:
    """Pulls the per-request USD cost OpenRouter reports when a call passes
    `extra_body={"usage": {"include": True}}` — this isn't part of the
    standard OpenAI response schema, so the openai SDK's pydantic models
    (extra="allow") keep it under `usage.model_extra` rather than a typed
    field. Returns None if the provider behind this particular model/route
    didn't report a cost — not every one does, and that's not treated as an
    error, just missing data (see src/eval/runner.py's cost aggregation).
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    cost = getattr(usage, "cost", None)
    if cost is not None:
        return float(cost)

    extra = getattr(usage, "model_extra", None) or {}
    cost = extra.get("cost")
    return float(cost) if cost is not None else None
