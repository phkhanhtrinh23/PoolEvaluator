"""Task-independent Stage 1-3 driver shared by the Text2SQL, image, and node pipelines."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .estimator import Estimate, PoolEvaluator, calibration_parameters


def task_settings(config: Mapping[str, Any], task: str) -> dict[str, Any]:
    return dict(config.get("tasks", {}).get(task, {}))


def subset_budget(requested: int | None, settings: Mapping[str, Any]) -> int:
    budget = int(settings.get("subset_budget", 0) if requested is None else requested)
    if budget < 0:
        raise ValueError("the subset budget K must be non-negative")
    return budget


def build_evaluator(config: Mapping[str, Any], seed: int) -> PoolEvaluator:
    method = config["method"]
    init = method.get("random_init", {})
    return PoolEvaluator(
        method["max_em_iterations"],
        method["tolerance"],
        method["smoothing"],
        seed=seed,
        init_mean=init.get("mean", 0.5),
        init_std=init.get("std", 0.25),
    )


def estimate_pool(
    target: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    *,
    valid: np.ndarray | None = None,
    source: np.ndarray | None = None,
    source_gold: Sequence[Any] | None = None,
    subset_ids: Sequence[Any] | None = None,
    judge: Callable[[int, list[Any], list[float]], Any] | None = None,
    rounds: int = 0,
) -> tuple[Estimate, str]:
    """Stage 1 initialization, Stage 2 EM, and optional Stage 3 judge refinement.

    Returns the estimate and the initialization used: "retrieved-subsets" when labeled
    calibration responses are supplied (K > 0), otherwise "random-truncated-normal".
    """
    evaluator = build_evaluator(config, seed)
    if source is not None and len(source):
        initial = calibration_parameters(
            source, source_gold, config["method"]["smoothing"], subset_ids=subset_ids
        )
        initialization = "retrieved-subsets"
    else:
        initial = None
        initialization = "random-truncated-normal"
    estimate = evaluator.fit(target, initial)
    if judge is not None and int(rounds) > 0:
        estimate = evaluator.refine(
            target,
            (estimate.alpha, estimate.beta, estimate.gamma),
            judge,
            rounds=int(rounds),
            batch_size=config["method"]["judge_batch_size"],
            valid=valid,
        )
    return estimate, initialization


def unique_failures(labels: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Give every invalid output its own identity so failures never agree with each other."""
    responses = np.asarray(labels, dtype=object).copy()
    for code, (i, j) in enumerate(zip(*np.nonzero(invalid)), start=1):
        responses[i, j] = -code
    return responses


def judge_history(estimate: Estimate) -> list[dict[str, Any]]:
    return [
        {key: (value if isinstance(value, (str, int, float)) else repr(value)) for key, value in row.items()}
        for row in estimate.history
        if "round" in row
    ]


def release(*objects: Any) -> None:
    del objects
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[saved] {path}")


def safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)
