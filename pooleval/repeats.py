"""Repeated runs over sampled model pools with 95% confidence intervals (paper Sec. 5).

The paper averages every result over 10 repeated runs whose pools are sampled from the
full model pool (5 to 30 models).  Model outputs are computed once for the whole pool;
each run samples a pool, reruns Stages 1-3 on those columns, and scores the estimate
against the true accuracy of the sampled models.

A "slot" is one pool member.  A slot may have several interchangeable columns (the
node architectures trained with three seeds); each run draws one of them.

End-to-end latency of a run = shared costs (retrieval, database instances, ...) + the
recorded cost of producing every sampled model's outputs + the run's own estimation
time + the judge time (original time of cached votes).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .core import estimate_pool
from .metrics import evaluate

_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
    40: 2.021, 60: 2.000, 120: 1.980,
}


def t_critical(df: int) -> float:
    """Two-sided 95% Student-t critical value."""
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except ImportError:
        eligible = [k for k in _T975 if k <= df]
        return _T975[max(eligible)] if eligible else _T975[1]


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean, sample SD, and the t-based 95% CI of the mean."""
    data = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if data.size == 0:
        return {"n": 0}
    mean = float(data.mean())
    if data.size < 2:
        return {"n": 1, "mean": mean, "sd": 0.0, "ci95_low": mean, "ci95_high": mean}
    sd = float(data.std(ddof=1))
    half = t_critical(data.size - 1) * sd / math.sqrt(data.size)
    return {"n": int(data.size), "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def run_repeated(
    target: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    *,
    slot_names: Sequence[str],
    slots: Sequence[Sequence[int]] | None = None,
    column_names: Sequence[str] | None = None,
    correct: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    source: np.ndarray | None = None,
    source_gold: Sequence[Any] | None = None,
    subset_ids: Sequence[Any] | None = None,
    judge: Callable[[int, list[Any], list[float]], Any] | None = None,
    rounds: int = 0,
    sizes: Sequence[int] = (5, 10, 15, 20, 25, 30),
    repeats: int = 10,
    column_seconds: Sequence[float | None] | None = None,
    fixed_seconds: float = 0.0,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=object)
    width = target.shape[1]
    slots = [list(s) for s in (slots or [[j] for j in range(width)])]
    column_names = list(column_names or slot_names)
    costs = None if column_seconds is None else np.asarray(
        [np.nan if c is None else float(c) for c in column_seconds], dtype=float
    )
    usable = sorted({int(m) for m in sizes if 2 <= int(m) <= len(slots)})
    skipped = sorted({int(m) for m in sizes} - set(usable))
    runs: list[dict[str, Any]] = []
    for size in usable:
        for rep in range(int(repeats)):
            rng = np.random.default_rng([seed, size, rep])
            chosen = np.sort(rng.choice(len(slots), size=size, replace=False))
            cols = [slots[s][int(rng.integers(len(slots[s])))] for s in chosen]
            run_seed = int(rng.integers(2**31 - 1))
            run = estimate_pool(
                target[:, cols], config, run_seed,
                valid=None if valid is None else valid[:, cols],
                source=None if source is None else source[:, cols],
                source_gold=source_gold, subset_ids=subset_ids,
                judge=judge, rounds=rounds,
            )
            truth = None if correct is None else correct[:, cols].astype(float).mean(axis=0)
            record: dict[str, Any] = {
                "size": size,
                "repeat": rep,
                "models": [column_names[c] for c in cols],
                "initialization": run.initialization,
                "estimated_accuracy": run.estimate.alpha.tolist(),
                "estimation_seconds": {k: float(v) for k, v in run.seconds.items()},
                "rounds": run.round_table(truth),
            }
            method_seconds = sum(
                run.seconds[k] for k in ("stage1_prior", "stage2_em", "stage3_select_em", "judge")
            )
            model_cost = None if costs is None else float(np.sum(costs[cols]))
            record["latency_seconds"] = (
                None if model_cost is None or math.isnan(model_cost)
                else float(fixed_seconds) + model_cost + method_seconds
            )
            if truth is not None:
                record["true_accuracy"] = truth.tolist()
                record["metrics"] = evaluate(run.estimate.alpha, truth)
            runs.append(record)
            print(f"[repeat] size={size} rep={rep + 1}/{repeats}"
                  + (f" mae={record['metrics']['mae']:.4f}" if truth is not None else ""))

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"runs": len(selected)}
        if selected and "metrics" in selected[0]:
            out["metrics"] = {
                name: summarize([r["metrics"][name] for r in selected]) for name in selected[0]["metrics"]
            }
        out["latency_seconds"] = summarize([r["latency_seconds"] for r in selected])
        return out

    per_round: list[dict[str, Any]] = []
    for v in range(int(rounds) + 1):
        mae: list[float] = []
        reduction: list[float] = []
        cumulative: list[float] = []
        for r in runs:
            table = r["rounds"]
            row = table[min(v, len(table) - 1)]  # a run that stopped early keeps its last estimate
            if "mae" in row:
                mae.append(row["mae"])
            if v < len(table) and "iteration_reduction" in table[v]:
                reduction.append(table[v]["iteration_reduction"])
            if "cumulative_iteration_reduction" in row:
                cumulative.append(row["cumulative_iteration_reduction"])
        per_round.append({
            "round": v,
            "mae": summarize(mae),
            "iteration_reduction": summarize(reduction),
            "cumulative_iteration_reduction": summarize(cumulative),
        })
    return {
        "protocol": {
            "pool_sizes": usable,
            "skipped_pool_sizes": skipped,
            "repeats": int(repeats),
            "pool_slots": list(slot_names),
            "seeded_by": [seed, "size", "repeat"],
        },
        "overall": aggregate(runs),
        "by_pool_size": {str(size): aggregate([r for r in runs if r["size"] == size]) for size in usable},
        "judge_rounds": per_round,
        "runs": runs,
    }
