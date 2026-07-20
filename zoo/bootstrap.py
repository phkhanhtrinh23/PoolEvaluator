"""Item-level bootstrap CIs for the real Spider zoo run.

One real pool has no pool-resampling axis, so uncertainty comes from resampling the
N items (columns of true_class). For each resample we recompute true EX and every
method's estimate, then report mean +/- 95% percentile CI.
"""
import json
import os
import numpy as np

from pooleval.data.simulator import PoolRun
from pooleval import Config, PoolEval, metrics
from baselines import Independent, Majority, DawidSkene, AgreementLine
from zoo.config import ARTIFACT_ROOT

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _run_from_cols(base, cols):
    tc = base.true_class[:, cols]
    ta = (tc == 0).mean(axis=1)
    return PoolRun(tc, ta, base.group, base.prior, base.prior_sigma,
                   np.zeros(len(cols)), np.zeros(len(cols)),
                   base.verifier_guess[cols], base.verifier_correct[cols],
                   base.M, len(cols), base.n_groups)


def main(n_boot=300, seed=0):
    d = np.load(os.path.join(ARTIFACT_ROOT, "poolrun_dev.npz"), allow_pickle=True)
    base = PoolRun(d["true_class"], d["true_acc"], d["group"], d["prior"],
                   d["prior_sigma"], np.zeros(d["true_class"].shape[1]),
                   np.zeros(d["true_class"].shape[1]), d["verifier_guess"],
                   d["verifier_guess"] == 0, d["true_class"].shape[0],
                   d["true_class"].shape[1], int(d["group"].max()) + 1)
    cfg = Config(real_data=True)
    methods = {"B1 Independent": Independent(), "B2 Majority/self-cons.": Majority(),
               "B3 Dawid--Skene": DawidSkene(), "B4 Agreement-on-line": AgreementLine()}
    keys = ["MAE", "Flip", "Kendall", "Top1", "Top3"]
    acc = {name: {k: [] for k in keys} for name in list(methods) + ["PoolEval-SQL (ours)"]}
    rng = np.random.default_rng(seed)
    N = base.N
    for _ in range(n_boot):
        cols = rng.integers(0, N, size=N)
        run = _run_from_cols(base, cols)
        for name, b in methods.items():
            m = metrics.all_metrics(b.evaluate(run, cfg), run.true_acc)
            for k in keys:
                acc[name][k].append(m[k])
        out = PoolEval(cfg).evaluate(run)
        m = metrics.all_metrics(out["acc"], run.true_acc)
        for k in keys:
            acc["PoolEval-SQL (ours)"][k].append(m[k])

    summary = {}
    print(f"\n=== item-bootstrap ({n_boot} resamples) mean [2.5%, 97.5%] ===")
    print(f"{'Method':24s}" + "".join(f"{k:>18s}" for k in keys))
    for name in list(methods) + ["PoolEval-SQL (ours)"]:
        summary[name] = {}
        cells = ""
        for k in keys:
            v = np.array(acc[name][k])
            lo, hi = np.percentile(v, [2.5, 97.5])
            summary[name][k] = [float(v.mean()), float(lo), float(hi)]
            cells += f"{v.mean():6.2f}[{lo:5.2f},{hi:5.2f}]"
        print(f"{name:24s}{cells}")
    with open(os.path.join(RESULTS, "zoo_spider_bootstrap.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {os.path.join(RESULTS, 'zoo_spider_bootstrap.json')}")


if __name__ == "__main__":
    main()
