.PHONY: setup lint test eval serve

setup:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest

# Usage: make eval CONFIG=configs/baseline.yaml (defaults to baseline)
CONFIG ?= configs/baseline.yaml
eval:
	uv run python -m src.eval $(CONFIG)

# Usage: make serve CONFIG=configs/hybrid.yaml (defaults to baseline)
serve:
	RAG_CONFIG_PATH=$(CONFIG) uv run uvicorn src.api.app:app --reload
