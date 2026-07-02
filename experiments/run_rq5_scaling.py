"""RQ5 -- Scaling with pool size (paper Fig 'fig:scaling').

Grows the pool from M=5 to M=20 at fixed diversity/shift and tracks ranking
quality for PoolEval-SQL vs the strongest independent baseline (B1). PoolEval-SQL's
advantage widens with M as each fresh-provenance member adds independent agreement;
B1 scores each model in isolation and is flat in M.

  python experiments/run_rq5_scaling.py [--seeds 6]
"""
import argparse
import dataclasses
import numpy as np
from _shared import Config, PoolEval, simulate, metrics, mean_ci, save_json
from baselines import Independent


def main(seeds=6):
    Ms = [5, 8, 12, 16, 20]
    print(f"\nRQ5: scaling with pool size M (mean over {seeds} seeds). "
          "Flip lower better, Kendall higher better.")
    print("-" * 74)
    print(f"{'M':>4s}{'B1 Flip':>12s}{'PoolEval-SQL Flip':>18s}"
          f"{'B1 Kend':>12s}{'PoolEval-SQL Kend':>18s}")
    table = []
    for M in Ms:
        b1f, pef, b1k, pek = [], [], [], []
        for s in range(seeds):
            cfg = dataclasses.replace(Config(seed=s), M=M)
            run = simulate(cfg)
            b1 = Independent().evaluate(run, cfg)
            out = PoolEval(cfg).evaluate(run)
            b1f.append(metrics.flip_rate(b1, run.true_acc))
            pef.append(metrics.flip_rate(out["acc"], run.true_acc))
            b1k.append(metrics.kendall(b1, run.true_acc))
            pek.append(metrics.kendall(out["acc"], run.true_acc))
        row = dict(M=M, b1_flip=mean_ci(b1f)[0], pe_flip=mean_ci(pef)[0],
                   b1_kend=mean_ci(b1k)[0], pe_kend=mean_ci(pek)[0])
        table.append(row)
        print(f"{M:>4d}{row['b1_flip']:12.3f}{row['pe_flip']:18.3f}"
              f"{row['b1_kend']:12.2f}{row['pe_kend']:18.2f}")
    save_json("rq5_scaling.json", table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    main(**vars(ap.parse_args()))
