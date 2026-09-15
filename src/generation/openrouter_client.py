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
