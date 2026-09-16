# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Dependencies first so this layer caches independently of source changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Source + the corpus the service indexes at startup (see src/api/app.py) +
# every experiment config, so RAG_CONFIG_PATH can select any of them without
# rebuilding the image.
COPY src/ src/
COPY configs/ configs/
COPY data/raw/ data/raw/
COPY eval/golden_set.json eval/golden_set.json

RUN uv sync --frozen --no-dev

ENV RAG_CONFIG_PATH=configs/baseline.yaml
# OPENROUTER_API_KEY is required at runtime — set as a platform secret
# (`fly secrets set OPENROUTER_API_KEY=...`), never baked into the image.

EXPOSE 8080

# Invoke the venv's uvicorn directly, NOT `uv run uvicorn ...`: uv run
# re-syncs the project's dependency groups on every invocation, and without
# an explicit --no-dev it pulls in dev-only tools (ruff, pytest) again even
# though the build-time `uv sync --frozen --no-dev` above already excluded
# them — wasted startup time and a network call on every cold start, for no
# reason (found this the hard way in a real deploy's logs).
CMD [".venv/bin/uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
