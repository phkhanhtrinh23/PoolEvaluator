"""RQ3 -- Correlated errors and independence (paper Fig 'fig:correlation').

Sweeps the pool error-correlation by raising the fraction of within-group
near-clones (collusion). Shows (left) the independence-assuming model (B3
Dawid--Skene) inflates estimated accuracy as correlation grows, while PoolEval's
provenance-grouped model stays close to truth; and (right) PoolEval's reported
M_eff falls from ~M toward 1, tracking the loss of independent evidence.

  python experiments/run_rq3_correlation.py [--seeds 6]
"""
import argparse
import dataclasses
import numpy as np
from _shared import Config, PoolEval, simulate, metrics, mean_ci, save_json
from baselines import DawidSkene


def main(seeds=6):
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    print(f"\nRQ3: correlation sweep (mean over {seeds} seeds). As collusion grows, "
          "B3 inflates accuracy; PoolEval stays accurate and M_eff drops.")
    print("-" * 72)
    print(f"{'collusion':>10s}{'B3 MAE':>10s}{'PoolEval MAE':>14s}"
          f"{'M_eff':>10s}{'PoolEval bias':>16s}")
    table = []
    for c in levels:
        b3m, pem, meff, peb = [], [], [], []
        for s in range(seeds):
            cfg = dataclasses.replace(Config(seed=s), collusion=c)
            run = simulate(cfg)
            b3 = DawidSkene().evaluate(run, cfg)
            out = PoolEval(cfg).evaluate(run)
            b3m.append(metrics.mae(b3, run.true_acc))
            pem.append(metrics.mae(out["acc"], run.true_acc))
            meff.append(out["Meff"])
            peb.append(float(np.mean(out["acc"] - run.true_acc) * 100))  # signed bias
        row = dict(collusion=c, b3_mae=mean_ci(b3m)[0], pe_mae=mean_ci(pem)[0],
                   meff=mean_ci(meff)[0], pe_bias=mean_ci(peb)[0])
        table.append(row)
        print(f"{c:10.2f}{row['b3_mae']:10.2f}{row['pe_mae']:14.2f}"
              f"{row['meff']:10.1f}{row['pe_bias']:+16.2f}")
    save_json("rq3_correlation.json", table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    main(**vars(ap.parse_args()))
