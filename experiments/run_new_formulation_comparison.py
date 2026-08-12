"""Paired comparison of old PoolEval and the closed-form new formulation.

The two methods always consume the same simulated pool and the same graded-kernel
observation. The old estimator produces the fixed pseudo-label used by the new
estimator, so differences after that point come only from the new binary agreement
EM. This script mirrors the main conditions covered by RQ1--RQ8 without changing
the existing experiment scripts.

  python experiments/run_new_formulation_comparison.py --seeds 8
"""
import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import (Config, NewFormulationPoolEval, PoolEval, kernel, metrics,
                      simulate)
from experiments._shared import aggregate, mean_ci, save_json
from experiments.run_rq2_ablation import VARIANTS
from experiments.run_rq8_drift import BATCHES, drifting_acc


def paired(cfg, run=None, alpha_init=None):
    run = simulate(cfg) if run is None else run
    obs = kernel.apply(run.true_class, cfg, level=cfg.kernel_level,
                       rng=np.random.default_rng(cfg.seed + 7))
    old = PoolEval(cfg).evaluate(run, obs=obs)
    new = NewFormulationPoolEval(cfg).evaluate(
        run, obs=obs, pseudo_out=old, alpha_init=alpha_init
    )
    # A kernel false-negative can split a truly correct result (true class 0) into a
    # fresh observed class ID. Count the selected result as correct if its observed
    # class contains a truly correct model output. Gold is diagnostic only.
    pseudo_true = np.asarray([
        np.any((obs[:, i] == label) & (run.true_class[:, i] == 0))
        for i, label in enumerate(new["pseudo_label"])
    ])
    return old, new, dict(
        pseudo_label_accuracy=float(pseudo_true.mean()),
        beta=float(new["beta"]),
        beta_error=float(abs(new["beta"] - pseudo_true.mean())),
        iterations=int(new["n_iters"]),
    )


def metric_pair(cfg, run=None):
    run = simulate(cfg) if run is None else run
    old, new, diag = paired(cfg, run)
    return (metrics.all_metrics(old["acc"], run.true_acc),
            metrics.all_metrics(new["acc"], run.true_acc), diag)


def rq1(seeds):
    old_rows, new_rows, diagnostics = [], [], []
    for seed in range(seeds):
        old, new, diag = metric_pair(Config(seed=seed))
        old_rows.append(old); new_rows.append(new); diagnostics.append(diag)
    return dict(old=aggregate(old_rows), new=aggregate(new_rows), diagnostics={
        key: mean_ci([row[key] for row in diagnostics]) for key in diagnostics[0]
    })


def rq2(seeds):
    result = {}
    for name, overrides in VARIANTS.items():
        old_rows, new_rows = [], []
        for seed in range(seeds):
            base = Config(seed=seed)
            run = simulate(base)
            cfg = dataclasses.replace(base, **overrides)
            old, new, _ = metric_pair(cfg, run)
            old_rows.append(old); new_rows.append(new)
        result[name] = dict(old=aggregate(old_rows), new=aggregate(new_rows))
    return result


def rq3(seeds):
    rows = []
    for collusion in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
        old_mae, new_mae, old_flip, new_flip = [], [], [], []
        for seed in range(seeds):
            cfg = dataclasses.replace(Config(seed=seed), collusion=collusion)
            old, new, _ = metric_pair(cfg)
            old_mae.append(old["MAE"]); new_mae.append(new["MAE"])
            old_flip.append(old["Flip"]); new_flip.append(new["Flip"])
        rows.append(dict(collusion=collusion, old_mae=mean_ci(old_mae),
                         new_mae=mean_ci(new_mae), old_flip=mean_ci(old_flip),
                         new_flip=mean_ci(new_flip)))
    return rows


def rq4(seeds):
    rows = []
    for level in ["LA0", "LA1", "LA2"]:
        old_rows, new_rows, precision, recall = [], [], [], []
        for seed in range(seeds):
            cfg = dataclasses.replace(Config(seed=seed), kernel_level=level)
            run = simulate(cfg)
            obs = kernel.apply(run.true_class, cfg, level=level,
                               rng=np.random.default_rng(seed + 7))
            p, r = kernel.measure(run.true_class, obs)
            old = PoolEval(cfg).evaluate(run, obs=obs)
            new = NewFormulationPoolEval(cfg).evaluate(run, obs=obs, pseudo_out=old)
            old_rows.append(metrics.all_metrics(old["acc"], run.true_acc))
            new_rows.append(metrics.all_metrics(new["acc"], run.true_acc))
            precision.append(p); recall.append(r)
        rows.append(dict(level=level, precision=mean_ci(precision),
                         recall=mean_ci(recall), old=aggregate(old_rows),
                         new=aggregate(new_rows)))
    return rows


def rq5(seeds):
    rows = []
    for M in [5, 8, 12, 16, 20]:
        old_rows, new_rows = [], []
        for seed in range(seeds):
            cfg = dataclasses.replace(Config(seed=seed), M=M)
            old, new, _ = metric_pair(cfg)
            old_rows.append(old); new_rows.append(new)
        rows.append(dict(M=M, old=aggregate(old_rows), new=aggregate(new_rows)))
    return rows


def rq6(seeds):
    rows = []
    for N in [100, 250, 500, 1000, 1500]:
        old_rows, new_rows = [], []
        for seed in range(seeds):
            cfg = dataclasses.replace(Config(seed=seed), N=N)
            old, new, _ = metric_pair(cfg)
            old_rows.append(old); new_rows.append(new)
        rows.append(dict(N=N, old=aggregate(old_rows), new=aggregate(new_rows)))
    return rows


def rq7(seeds):
    rows = []
    for N in [100, 250, 500, 1000, 1500]:
        meta_metrics, cold_metrics, meta_iters, cold_iters = [], [], [], []
        for seed in range(seeds):
            cfg = Config(seed=seed, N=N)
            run = simulate(cfg)
            obs = kernel.apply(run.true_class, cfg, level=cfg.kernel_level,
                               rng=np.random.default_rng(seed + 7))
            old = PoolEval(cfg).evaluate(run, obs=obs)
            estimator = NewFormulationPoolEval(cfg)
            meta = estimator.evaluate(run, obs=obs, pseudo_out=old)
            cold = estimator.evaluate(run, obs=obs, pseudo_out=old,
                                      alpha_init=np.full(cfg.M, 0.6))
            meta_metrics.append(metrics.all_metrics(meta["acc"], run.true_acc))
            cold_metrics.append(metrics.all_metrics(cold["acc"], run.true_acc))
            meta_iters.append(meta["n_iters"]); cold_iters.append(cold["n_iters"])
        rows.append(dict(N=N, meta=aggregate(meta_metrics), cold=aggregate(cold_metrics),
                         meta_iters=mean_ci(meta_iters), cold_iters=mean_ci(cold_iters)))
    return rows


def rq8(seeds):
    rows = []
    for batch in range(BATCHES):
        old_flips, new_flips = [], []
        for seed in range(seeds):
            base_cfg = Config(seed=seed, N=400)
            source = simulate(base_cfg)
            acc = drifting_acc(source.true_acc, batch)
            cfg = dataclasses.replace(base_cfg, seed=seed * 100 + batch)
            run = simulate(cfg, acc=acc, group_assignment=source.group)
            old, new, _ = metric_pair(cfg, run)
            old_flips.append(old["Flip"]); new_flips.append(new["Flip"])
        rows.append(dict(batch=batch + 1, old_flip=mean_ci(old_flips),
                         new_flip=mean_ci(new_flips)))
    return rows


def show_metric(name, old, new, metric):
    print(f"{name:24s}{old[metric][0]:12.3f}{new[metric][0]:12.3f}")


def main(seeds=8):
    result = dict(seeds=seeds)
    result["rq1_main"] = rq1(seeds)
    result["rq2_ablation"] = rq2(seeds)
    result["rq3_correlation"] = rq3(seeds)
    result["rq4_kernel"] = rq4(seeds)
    result["rq5_scaling"] = rq5(seeds)
    result["rq6_budget"] = rq6(seeds)
    result["rq7_initialization"] = rq7(seeds)
    result["rq8_drift"] = rq8(seeds)

    print(f"\nNew formulation paired comparison ({seeds} seeds, mean +/- 95% CI)")
    print("-" * 60)
    print(f"{'Main metric':24s}{'Old':>12s}{'New':>12s}")
    for metric in ["MAE", "Flip", "Kendall", "Top1", "Top3"]:
        show_metric(metric, result["rq1_main"]["old"],
                    result["rq1_main"]["new"], metric)
    d = result["rq1_main"]["diagnostics"]
    print(f"\nPseudo-label accuracy: {d['pseudo_label_accuracy'][0]:.3f}")
    print(f"Estimated beta       : {d['beta'][0]:.3f}")
    print(f"Absolute beta error  : {d['beta_error'][0]:.3f}")
    print(f"EM iterations        : {d['iterations'][0]:.1f}")

    print("\nCorrelation sweep")
    print(f"{'rho':>6s}{'Old MAE':>12s}{'New MAE':>12s}{'Old Flip':>12s}{'New Flip':>12s}")
    for row in result["rq3_correlation"]:
        print(f"{row['collusion']:6.2f}{row['old_mae'][0]:12.2f}"
              f"{row['new_mae'][0]:12.2f}{row['old_flip'][0]:12.3f}"
              f"{row['new_flip'][0]:12.3f}")

    print("\nKernel ladder")
    print(f"{'level':>7s}{'Old MAE':>12s}{'New MAE':>12s}{'Old Flip':>12s}{'New Flip':>12s}")
    for row in result["rq4_kernel"]:
        print(f"{row['level']:>7s}{row['old']['MAE'][0]:12.2f}"
              f"{row['new']['MAE'][0]:12.2f}{row['old']['Flip'][0]:12.3f}"
              f"{row['new']['Flip'][0]:12.3f}")

    save_json("new_formulation_comparison.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    main(**vars(parser.parse_args()))
