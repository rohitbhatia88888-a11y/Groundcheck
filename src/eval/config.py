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
    name: str
    raw_data_dir: str = "data/raw"
    golden_set_path: str = "eval/golden_set.json"
    results_path: str | None = None  # defaults to results/<name>.csv

    parser: ComponentConfig
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
