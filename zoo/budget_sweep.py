"""Real-data judge-budget sweep: how ranking quality moves as the gpt-5-mini judge
validates 0 / 5 / 10 / 20 % of the questions (hybrid-submodular selection). Reuses the
saved PoolRun + cached predictions + cached judge calls, so re-runs are cheap.

    python -m zoo.budget_sweep --datasets spider bird bird_minidev sqlflow spider2local
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import (Config, ActivePoolEval, ActiveConfig, metrics,  # noqa: E402
                      PoolEval, SimulatedJudge)
from zoo.datasets import load_split                                  # noqa: E402
from zoo.judge import RealJudge                                      # noqa: E402
from zoo.analyze import load_run                                     # noqa: E402
from zoo.config import ARTIFACT_ROOT                                 # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
FRACS = [0.0, 0.05, 0.10, 0.20]


def sweep(ds, model="gpt-5-mini", mock=False):
    run = load_run(ds)
    names = list(np.load(os.path.join(ARTIFACT_ROOT, f"poolrun_{ds}.npz"),
                         allow_pickle=True)["names"])
    names = [str(x) for x in names]
    dev = None if mock else load_split(ds, "dev", run.N, seed=0)
    preds = None if mock else json.load(
        open(os.path.join(ARTIFACT_ROOT, f"preds_{ds}_dev.json")))
    cfg = Config(real_data=True)
    rows = []
    for fr in FRACS:
        b = int(round(fr * run.N))
        if b == 0:
            acc = PoolEval(cfg).evaluate(run)["acc"]; calls = 0
        else:
            judge = SimulatedJudge() if mock else \
                RealJudge(dev, names, preds, run.true_class, model=model)
            out = ActivePoolEval(cfg, ActiveConfig(budget=b, rounds=min(b, 6),
                                 strategy="hybrid_submodular"), judge=judge).run(run)
            acc = out["acc"]; calls = out["judge_calls"]
        m = metrics.all_metrics(acc, run.true_acc)
        rows.append(dict(frac=fr, budget=b, calls=calls, MAE=m["MAE"],
                        Kendall=m["Kendall"], Top1=m["Top1"]))
        print(f"  [{ds}] {int(fr*100):>2d}% (b={b:>2d}): MAE={m['MAE']:.2f} "
              f"Kendall={m['Kendall']:.2f} Top1={m['Top1']:.0f}")
    return rows


def main(datasets, mock=False):
    out = {}
    for ds in datasets:
        print(f"\n=== {ds} ===")
        out[ds] = sweep(ds, mock=mock)
    with open(os.path.join(RESULTS, "zoo_budget_sweep.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[saved] {os.path.join(RESULTS, 'zoo_budget_sweep.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["spider", "sqlflow", "bird", "bird_minidev", "spider2local"])
    ap.add_argument("--mock", action="store_true")
    main(ap.parse_args().datasets, mock=ap.parse_args().mock)
