"""Shared helpers for PoolEval experiment scripts."""
import os
import sys
import json
import numpy as np
from scipy.stats import pearsonr

# allow `python experiments/run_xxx.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import Config, PoolEval, simulate, metrics          # noqa: E402
from baselines import all_baselines                                # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
os.makedirs(RESULTS, exist_ok=True)

METRIC_KEYS = ["MAE", "Flip", "Kendall", "Top1", "Top3"]


def mean_ci(values):
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    ci = float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return m, ci


def aggregate(per_seed):
    """per_seed: list of metric dicts -> {metric: (mean, ci)}."""
    return {k: mean_ci([d[k] for d in per_seed]) for k in per_seed[0]}


def coverage(acc_hat, acc_true, half_width):
    acc_hat = np.asarray(acc_hat)
    acc_true = np.asarray(acc_true)
    half_width = np.asarray(half_width)
    return float(np.mean((acc_true >= acc_hat - half_width) &
                         (acc_true <= acc_hat + half_width)))


def calibration_error(probs, labels, bins=10):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probs.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs <= hi if hi == 1.0 else probs < hi)
        if not np.any(mask):
            continue
        ece += (mask.mean() * abs(probs[mask].mean() - labels[mask].mean()))
    return float(ece)


def pearson(a, b):
    r, _ = pearsonr(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return float(r) if np.isfinite(r) else 0.0


def evaluate_pool(make_cfg, seeds=5, include_baselines=True, methods=None):
    """Run methods over `seeds` resampled pools. Returns {name: aggregated}."""
    rows = {}
    extra = methods or {}
    for s in range(seeds):
        cfg = make_cfg(s)
        run = simulate(cfg)
        if include_baselines:
            for b in all_baselines():
                m = metrics.all_metrics(b.evaluate(run, cfg), run.true_acc)
                rows.setdefault(b.name, []).append(m)
        out = PoolEval(cfg).evaluate(run)
        rows.setdefault("PoolEval (ours)", []).append(
            metrics.all_metrics(out["acc"], run.true_acc))
        for name, fn in extra.items():
            rows.setdefault(name, []).append(fn(run, cfg, out))
    return {k: aggregate(v) for k, v in rows.items()}


def print_table(title, agg, keys=METRIC_KEYS, order=None):
    print(f"\n{title}")
    print("-" * (22 + 14 * len(keys)))
    print(f"{'Method':22s}" + "".join(f"{k:>14s}" for k in keys))
    names = order or list(agg.keys())
    for name in names:
        if name not in agg:
            continue
        cells = []
        for k in keys:
            m, ci = agg[name][k]
            cells.append(f"{m:6.2f}+/-{ci:4.2f}")
        print(f"{name:22s}" + "".join(f"{c:>14s}" for c in cells))


def save_json(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[saved] {path}")
