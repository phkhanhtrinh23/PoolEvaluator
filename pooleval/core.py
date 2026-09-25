"""Task-independent Stage 1-3 driver shared by the Text2SQL, image, and node pipelines."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
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


def judge_rounds(requested: int | None, settings: Mapping[str, Any]) -> int:
    rounds = int(settings.get("judge_rounds", 0) if requested is None else requested)
    if rounds < 0:
        raise ValueError("the judge-round budget V must be non-negative")
    return rounds


def repeat_settings(args: Any, config: Mapping[str, Any]) -> tuple[list[int], int]:
    """Pool sizes and repeat count for ``--repeats`` runs (CLI overrides the config)."""
    experiments = config.get("experiments", {})
    sizes = getattr(args, "pool_sizes", None) or experiments.get("pool_sizes", [5, 10, 15, 20, 25, 30])
    repeats = int(getattr(args, "repeats", 0) or experiments.get("repeats", 10))
    return [int(size) for size in sizes], repeats


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


@dataclass
class PoolRun:
    """One PoolEvaluator run: the estimate plus what the paper's efficiency figures need."""

    estimate: Estimate
    initialization: str
    stage2_alpha: np.ndarray
    stage2_iterations: int
    rounds: list[dict[str, Any]]
    seconds: dict[str, float]

    def round_table(self, truth: np.ndarray | None = None) -> list[dict[str, Any]]:
        """Per judge round: iterations (warm, cold, reduction) and, with truth, the MAE.

        Round 0 is Stage 2.  ``iteration_reduction`` is 1 - warm/cold for that round and
        ``cumulative_iteration_reduction`` is 1 - sum(warm)/sum(cold) over rounds 1..v.
        """
        rows: list[dict[str, Any]] = [{"round": 0, "iterations_warm": self.stage2_iterations}]
        if truth is not None:
            rows[0]["mae"] = float(np.mean(np.abs(self.stage2_alpha - truth)))
        warm_total = cold_total = 0
        for row in self.rounds:
            out = {
                key: row[key]
                for key in ("round", "item", "information_gain", "iterations_warm",
                            "iterations_cold", "judge_seconds", "em_seconds")
                if key in row
            }
            out["answer"] = row["answer"] if isinstance(row["answer"], (str, int, float)) else repr(row["answer"])
            if "iterations_cold" in row:
                warm_total += row["iterations_warm"]
                cold_total += row["iterations_cold"]
                out["iteration_reduction"] = 1.0 - row["iterations_warm"] / max(row["iterations_cold"], 1)
                out["cumulative_iteration_reduction"] = 1.0 - warm_total / max(cold_total, 1)
            if truth is not None:
                out["mae"] = float(np.mean(np.abs(np.asarray(row["alpha"]) - truth)))
            rows.append(out)
        return rows


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
) -> PoolRun:
    """Stage 1 initialization, Stage 2 EM, and optional Stage 3 judge refinement.

    The initialization is "retrieved-subsets" when labeled calibration responses are
    supplied (K > 0) and "random-truncated-normal" otherwise.  Stage timings are wall
    seconds; ``seconds["judge"]`` counts the judge's own time (original time for
    cached votes).
    """
    evaluator = build_evaluator(config, seed)
    seconds: dict[str, float] = {}
    start = time.perf_counter()
    if source is not None and len(source):
        initial = calibration_parameters(
            source, source_gold, config["method"]["smoothing"], subset_ids=subset_ids
        )
        initialization = "retrieved-subsets"
    else:
        initial = None
        initialization = "random-truncated-normal"
    seconds["stage1_prior"] = time.perf_counter() - start
    start = time.perf_counter()
    estimate = evaluator.fit(target, initial)
    seconds["stage2_em"] = time.perf_counter() - start
    stage2_alpha, stage2_iterations = estimate.alpha.copy(), int(estimate.iterations)
    rounds_log: list[dict[str, Any]] = []
    seconds["stage3_select_em"] = seconds["judge"] = 0.0
    if judge is not None and int(rounds) > 0:
        start = time.perf_counter()
        estimate = evaluator.refine(
            target,
            (estimate.alpha, estimate.beta, estimate.gamma),
            judge,
            rounds=int(rounds),
            batch_size=config["method"]["judge_batch_size"],
            valid=valid,
            cold_initial=initial,
        )
        rounds_log = [row for row in estimate.history if "round" in row]
        judge_seconds = sum(row["judge_seconds"] for row in rounds_log)
        # Cold-start refits are diagnostics for the iteration-reduction figure, not part
        # of the method, so their time is excluded.
        seconds["stage3_select_em"] = sum(row["select_seconds"] + row["em_seconds"] for row in rounds_log)
        seconds["judge"] = judge_seconds
        seconds["stage3_wall_including_diagnostics"] = time.perf_counter() - start
    return PoolRun(estimate, initialization, stage2_alpha, stage2_iterations, rounds_log, seconds)


def unique_failures(labels: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Give every invalid output its own identity so failures never agree with each other."""
    responses = np.asarray(labels, dtype=object).copy()
    for code, (i, j) in enumerate(zip(*np.nonzero(invalid)), start=1):
        responses[i, j] = -code
    return responses


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
