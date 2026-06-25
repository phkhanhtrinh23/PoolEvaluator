"""Smoke tests: the pipeline runs and PoolEval beats the baselines on ranking."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import Config, PoolEval, simulate, metrics       # noqa: E402
from baselines import all_baselines                             # noqa: E402


def test_pipeline_runs():
    cfg = Config(seed=0, N=200)
    out = PoolEval(cfg).evaluate(simulate(cfg))
    assert out["acc"].shape == (cfg.M,)
    assert 1.0 <= out["Meff"] <= cfg.M + 1e-6
    assert 0.0 <= out["kernel_recall"] <= 1.0


def test_pooleval_best_flip():
    """Averaged over seeds, PoolEval has the lowest pairwise-flip rate."""
    flips = {b.name: [] for b in all_baselines()}
    flips["PoolEval"] = []
    for s in range(6):
        cfg = Config(seed=s)
        run = simulate(cfg)
        for b in all_baselines():
            flips[b.name].append(metrics.flip_rate(b.evaluate(run, cfg), run.true_acc))
        out = PoolEval(cfg).evaluate(run)
        flips["PoolEval"].append(metrics.flip_rate(out["acc"], run.true_acc))
    means = {k: float(np.mean(v)) for k, v in flips.items()}
    assert means["PoolEval"] <= min(means[k] for k in means if k != "PoolEval")


if __name__ == "__main__":
    test_pipeline_runs()
    test_pooleval_best_flip()
    print("ok")
