.PHONY: setup lint test eval

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
