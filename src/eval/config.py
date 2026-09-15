"""Experiment configuration: every experiment run is driven by one of these,
loaded from a YAML file in configs/. Never hardcode a strategy in calling
code — change the YAML instead (see CLAUDE.md non-negotiables).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ComponentConfig(BaseModel):
    """Which implementation to use for one swappable pipeline stage, and the
    kwargs to construct it with. `type` is looked up in the registries in
    src/eval/registry.py."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Output paths are NOT configurable here — src/eval/runner.py writes to
    the fixed conventions results/experiments.csv (one row per run) and
    results/runs/<name>_<timestamp>.jsonl (per-question detail), so results
    across different configs always land somewhere comparable."""

    name: str
    raw_data_dir: str = "data/raw"
    golden_set_path: str = "eval/golden_set.json"

    chunker: ComponentConfig
    embedder: ComponentConfig
    vector_store: ComponentConfig
    reranker: ComponentConfig
    generator: ComponentConfig

    retrieval_top_k: int = 10
    rerank_top_k: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
