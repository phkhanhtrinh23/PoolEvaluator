"""Shared helpers for PoolEval experiment scripts."""
import os
import sys
import json
import numpy as np

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
