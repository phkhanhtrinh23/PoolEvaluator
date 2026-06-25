"""RQ4 -- The equivalence kernel (paper Table 'tab:kernel').

Reports precision/recall of the graded execution-equivalence kernel against gold
equivalence per canonicalization level: L0 exact (high precision, poor recall),
L1 canonicalized (recovers recall), L2 multi-instance (raises precision).

  python experiments/run_rq4_kernel.py [--seeds 8]
"""
import argparse
import numpy as np
from _shared import Config, simulate, mean_ci, save_json
from pooleval import kernel


def main(seeds=8):
    print(f"\nRQ4: equivalence-kernel precision/recall vs gold (mean over {seeds} "
          "seeds).")
    print("-" * 44)
    print(f"{'Level':>8s}{'Precision':>14s}{'Recall':>14s}")
    table = []
    for level in ["L0", "L1", "L2"]:
        ps, rs = [], []
        for s in range(seeds):
            cfg = Config(seed=s)
            run = simulate(cfg)
            obs = kernel.apply(run.true_class, cfg, level=level,
                               rng=np.random.default_rng(cfg.seed + 7))
            p, r = kernel.measure(run.true_class, obs)
            ps.append(p); rs.append(r)
        row = dict(level=level, precision=mean_ci(ps)[0], recall=mean_ci(rs)[0])
        table.append(row)
        print(f"{level:>8s}{row['precision']:14.3f}{row['recall']:14.3f}")
    save_json("rq4_kernel.json", table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    main(**vars(ap.parse_args()))
