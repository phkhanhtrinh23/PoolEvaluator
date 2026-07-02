"""RQ7 -- Warm-start optimization.

Fits a lightweight meta-initializer on source pools, then compares cold-start
vs warm-start PoolEval-SQL as the number of unlabeled items grows.

  python experiments/run_rq7_warmstart.py [--seeds 5]
"""
import argparse
import dataclasses
import numpy as np

from _shared import Config, PoolEval, simulate, mean_ci, save_json


N_GRID = [100, 250, 500, 1000, 1500]
TRAIN_SEEDS = range(50, 60)


def fit_meta_initializer(base_cfg):
    xs = []
    ys = []
    for seed in TRAIN_SEEDS:
        cfg = dataclasses.replace(base_cfg, seed=seed, N=base_cfg.N)
        run = simulate(cfg)
        xs.extend(run.prior.tolist())
        ys.extend(run.true_acc.tolist())
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(slope), float(intercept)


def warm_init(prior, slope, intercept):
    return np.clip(intercept + slope * np.asarray(prior, dtype=float), 0.02, 0.98)


def main(seeds=5):
    base_cfg = Config(seed=0, N=max(N_GRID))
    slope, intercept = fit_meta_initializer(base_cfg)

    rows = []
    print(f"\nRQ7: warm-start optimization (mean over {seeds} seeds)")
    print("-" * 74)
    print(f"{'N':>6s}{'Cold top1':>12s}{'Warm top1':>12s}"
          f"{'Cold iters':>12s}{'Warm iters':>12s}")

    for N in N_GRID:
        cold_top1 = []
        warm_top1 = []
        cold_iters = []
        warm_iters = []
        for seed in range(seeds):
            cfg = Config(seed=seed, N=N)
            run = simulate(cfg)
            cold = PoolEval(cfg).evaluate(run, init={"a": np.full(cfg.M, 0.6)})
            warm = PoolEval(cfg).evaluate(run, init={
                "a": warm_init(run.prior, slope, intercept)
            })
            cold_top1.append(float(np.argmax(cold["acc"]) == np.argmax(run.true_acc)))
            warm_top1.append(float(np.argmax(warm["acc"]) == np.argmax(run.true_acc)))
            cold_iters.append(float(cold["n_iters"]))
            warm_iters.append(float(warm["n_iters"]))
        row = dict(
            N=N,
            cold_top1=mean_ci(cold_top1)[0],
            warm_top1=mean_ci(warm_top1)[0],
            cold_iters=mean_ci(cold_iters)[0],
            warm_iters=mean_ci(warm_iters)[0],
        )
        rows.append(row)
        print(f"{N:6d}{row['cold_top1']:12.2f}{row['warm_top1']:12.2f}"
              f"{row['cold_iters']:12.1f}{row['warm_iters']:12.1f}")

    save_json("rq7_warmstart.json", dict(
        slope=slope,
        intercept=intercept,
        rows=rows,
    ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    main(**vars(ap.parse_args()))
