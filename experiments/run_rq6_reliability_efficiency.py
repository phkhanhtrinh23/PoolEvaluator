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
CALIB_SEEDS = range(40, 48)   # held-out source pools for split-conformal calibration
                              # (disjoint from the eval seeds and RQ7's warm-start seeds)


def interval_half_width(out, nominal, widened=True, scale=1.0):
    z = float(norm.ppf(0.5 + nominal / 2.0))
    half = z * out["a_sigma"]
    if widened:
        half = half + 0.5 * (1.0 - out["kernel_precision"])
    return scale * half


def calibrate_scales(nominals, N=1500):
    """Split-conformal calibration of the interval scale on held-out source pools.

    The seen prior already comes from a labeled source domain, so a held-out source
    slice carries gold and can calibrate the intervals -- the paper's "conformal
    intervals calibrated on a held-out slice and widened by the equivalence error."
    We normalize each residual by its widened interval and take the finite-sample
    (1-alpha) quantile, so the widened target intervals reach nominal coverage while
    keeping the per-model, kernel-error-widened shape.
    """
    outs, resids = [], []
    for s in CALIB_SEEDS:
        cfg = Config(seed=s, N=N)
        run = simulate(cfg)
        out = PoolEval(cfg).evaluate(run)
        outs.append(out)
        resids.append(np.abs(out["acc"] - run.true_acc))
    scales = {}
    for nominal in nominals:
        scores = []
        for out, resid in zip(outs, resids):
            base = interval_half_width(out, nominal, widened=True, scale=1.0)
            scores.extend((resid / np.maximum(base, 1e-9)).tolist())
        scores = np.asarray(scores)
        level = min(1.0, np.ceil((len(scores) + 1) * nominal) / len(scores))
        scales[nominal] = float(np.quantile(scores, level, method="higher"))
    return scales


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


def one_seed(seed, scales, N=1500):
    cfg = Config(seed=seed, N=N)
    run = simulate(cfg)
    pe = PoolEval(cfg).evaluate(run)
    b1 = Independent().evaluate(run, cfg)

    cover = {"widened": [], "unwidened": []}
    for nominal in NOMINALS:
        k = scales[nominal]
        cover["widened"].append(
            coverage(pe["acc"], run.true_acc,
                     interval_half_width(pe, nominal, True, k))
        )
        cover["unwidened"].append(
            coverage(pe["acc"], run.true_acc,
                     interval_half_width(pe, nominal, False, k))
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
    scales = calibrate_scales(NOMINALS)
    rows = [one_seed(s, scales) for s in range(seeds)]
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
        conformal_scale={k: scales[k] for k in NOMINALS},
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
    print(f"\nPearson r: PoolEval-SQL={summary['pearson']['pooleval'][0]:.3f} "
          f"Independent={summary['pearson']['independent'][0]:.3f}")
    print(f"ECE: PoolEval-SQL={summary['ece'][0]:.3f}")
    print("\nCost proxy / item")
    print(f"  B1           : {summary['cost']['b1']['total']:.2f}")
    print(f"  PoolEval-SQL : {summary['cost']['pooleval']['total']:.2f}")
    print("\nBudget sweep (budget in 10^3 model calls)")
    print(f"{'Budget':>8s}{'B1 flip':>12s}{'PoolEval-SQL flip':>18s}")
    for b, bf, pf in zip(summary["budget"]["budget_thousands"],
                         summary["budget"]["independent_flip"],
                         summary["budget"]["pooleval_flip"]):
        print(f"{b:8.2f}{bf:12.3f}{pf:18.3f}")

    save_json("rq6_reliability_efficiency.json", summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    main(**vars(ap.parse_args()))
