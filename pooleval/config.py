from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def paper_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or ROOT / "configs" / "paper.yaml")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    id: str
    backend: str
    task: str
    base: str | None = None
    auxiliary: str | None = None
    runtime_repo: str | None = None


def load_model_pool(task: str, pool_dir: str | Path | None = None) -> list[ModelSpec]:
    """Read one appendix model-pool definition."""
    directory = Path(pool_dir) if pool_dir else ROOT / "model_pools"
    raw = load_yaml(directory / f"{task}.yaml")
    if raw.get("task") != task:
        raise ValueError(f"model-pool task mismatch: requested {task!r}")
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"model pool {task!r} contains no models")
    specs = [ModelSpec(task=task, **row) for row in models]
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate model name in {task!r} pool")
    return specs
