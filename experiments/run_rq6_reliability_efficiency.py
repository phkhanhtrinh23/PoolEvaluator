"""RQ6 -- Reliability and efficiency.

Reports interval coverage, calibration quality, and cost / budget curves for the
paper's reliability-and-efficiency section.

  python experiments/run_rq6_reliability_efficiency.py [--seeds 5]
"""
import argparse
import numpy as np
from scipy.stats import norm

from _shared import (Config, PoolEval, simulate, coverage, calibration_error,
                     pearson, mean_ci, save_json)
from pooleval import metrics
from baselines import Independent


NOMINALS = [0.80, 0.85, 0.90, 0.95]
BUDGETS = [100, 250, 500, 1000, 1500]


def interval_half_width(out, nominal, widened=True):
    z = float(norm.ppf(0.5 + nominal / 2.0))
    half = z * out["a_sigma"]
    if widened:
        half = half + 0.5 * (1.0 - out["kernel_precision"])
    return half


def win_probabilities(out):
    probs = []
    labels = []
    acc = out["acc"]
    truth = out["truth"]
    M = len(acc)
    for i in range(M):
        for j in range(i + 1, M):
            d = acc[i] - acc[j]
            s = np.sqrt(out["a_sigma"][i] ** 2 + out["a_sigma"][j] ** 2) + 1e-9
            p = float(norm.cdf(d / s))
            probs.append(p)
            labels.append(float(truth[i] > truth[j]))
    return np.asarray(probs), np.asarray(labels)


def one_seed(seed, N=1500):
    cfg = Config(seed=seed, N=N)
    run = simulate(cfg)
    pe = PoolEval(cfg).evaluate(run)
    b1 = Independent().evaluate(run, cfg)

    cover = {"widened": [], "unwidened": []}
    for nominal in NOMINALS:
        cover["widened"].append(
            coverage(pe["acc"], run.true_acc, interval_half_width(pe, nominal, True))
        )
        cover["unwidened"].append(
            coverage(pe["acc"], run.true_acc, interval_half_width(pe, nominal, False))
        )

    probs, labels = win_probabilities(dict(
        acc=pe["acc"], a_sigma=pe["a_sigma"], truth=run.true_acc
    ))
    return dict(
        coverage=cover,
        pe_pearson=pearson(pe["acc"], run.true_acc),
        b1_pearson=pearson(b1, run.true_acc),
        pe_ece=calibration_error(probs, labels),
        cost=dict(
            b1=dict(output_collection=cfg.M, em=0.0, reexec=0.0, total=cfg.M),
            pooleval=dict(output_collection=cfg.M, em=0.05, reexec=1.0,
                         total=cfg.M + 1.05),
        ),
    )


def main(seeds=5):
    rows = [one_seed(s) for s in range(seeds)]
    summary = dict(
        coverage=dict(
            widened={k: mean_ci([row["coverage"]["widened"][i] for row in rows])
                     for i, k in enumerate(NOMINALS)},
            unwidened={k: mean_ci([row["coverage"]["unwidened"][i] for row in rows])
                       for i, k in enumerate(NOMINALS)},
        ),
        pearson=dict(
            pooleval=mean_ci([row["pe_pearson"] for row in rows]),
            independent=mean_ci([row["b1_pearson"] for row in rows]),
        ),
        ece=mean_ci([row["pe_ece"] for row in rows]),
        cost=rows[0]["cost"],
        budget=dict(
            budget_thousands=[Config().M * N / 1000.0 for N in BUDGETS],
            pooleval_flip=[None] * len(BUDGETS),
            independent_flip=[None] * len(BUDGETS),
        ),
        seeds=seeds,
    )

    # Recompute the budget curve with the same seeds on varying N.
    pe_flips = {N: [] for N in BUDGETS}
    b1_flips = {N: [] for N in BUDGETS}
    for seed in range(seeds):
        for N in BUDGETS:
            cfg = Config(seed=seed, N=N)
            run = simulate(cfg)
            pe = PoolEval(cfg).evaluate(run)
            b1 = Independent().evaluate(run, cfg)
            pe_flips[N].append(metrics.flip_rate(pe["acc"], run.true_acc))
            b1_flips[N].append(metrics.flip_rate(b1, run.true_acc))
    summary["budget"] = dict(
        budget_thousands=[round(Config().M * N / 1000.0, 3) for N in BUDGETS],
        pooleval_flip=[mean_ci(pe_flips[N])[0] for N in BUDGETS],
        independent_flip=[mean_ci(b1_flips[N])[0] for N in BUDGETS],
    )

    print(f"\nRQ6: reliability / efficiency (mean over {seeds} seeds)")
    print("-" * 78)
    print(f"{'Nominal':>8s}{'Wide cov.':>12s}{'Narrow cov.':>12s}")
    for nominal in NOMINALS:
        print(f"{nominal:8.2f}{summary['coverage']['widened'][nominal][0]:12.2f}"
              f"{summary['coverage']['unwidened'][nominal][0]:12.2f}")
    print(f"\nPearson r: PoolEval={summary['pearson']['pooleval'][0]:.3f} "
          f"Independent={summary['pearson']['independent'][0]:.3f}")
    print(f"ECE: PoolEval={summary['ece'][0]:.3f}")
    print("\nCost proxy / item")
    print(f"  B1        : {summary['cost']['b1']['total']:.2f}")
    print(f"  PoolEval  : {summary['cost']['pooleval']['total']:.2f}")
    print("\nBudget sweep (budget in 10^3 model calls)")
    print(f"{'Budget':>8s}{'B1 flip':>12s}{'PoolEval flip':>16s}")
    for b, bf, pf in zip(summary["budget"]["budget_thousands"],
                         summary["budget"]["independent_flip"],
                         summary["budget"]["pooleval_flip"]):
        print(f"{b:8.2f}{bf:12.3f}{pf:16.3f}")

    save_json("rq6_reliability_efficiency.json", summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    main(**vars(ap.parse_args()))
