"""RQ8 -- Robustness to workload drift.

Simulates a stream of batches whose true model accuracies drift over time and
compares static, periodic-refit, and bounded online-refresh policies.

  python experiments/run_rq8_drift.py [--seeds 5]
"""
import argparse
import dataclasses
import numpy as np

from _shared import Config, PoolEval, simulate, mean_ci, save_json
from pooleval import metrics


BATCHES = 10


def drifting_acc(base_acc, batch, batches=BATCHES):
    base_acc = np.asarray(base_acc, dtype=float)
    center = np.linspace(-1.0, 1.0, len(base_acc))
    drift = (batch / max(1, batches - 1)) * 0.12 * center
    return np.clip(base_acc + drift, 0.02, 0.98)


def main(seeds=5):
    print(f"\nRQ8: robustness to workload drift (mean over {seeds} seeds)")
    print("-" * 74)
    print(f"{'Batch':>6s}{'Static':>12s}{'Periodic':>12s}{'Online':>12s}")

    per_batch = []
    static_state_by_seed = {}
    online_state_by_seed = {}
    for batch in range(BATCHES):
        static_flips = []
        periodic_flips = []
        online_flips = []
        for seed in range(seeds):
            base_cfg = Config(seed=seed, N=400)
            source_run = simulate(base_cfg)
            acc = drifting_acc(source_run.true_acc, batch)
            drift_cfg = dataclasses.replace(base_cfg, seed=seed * 100 + batch)
            run = simulate(drift_cfg, acc=acc, group_assignment=source_run.group)

            if batch == 0:
                static_state_by_seed[seed] = PoolEval(drift_cfg).evaluate(run)
                online_state_by_seed[seed] = static_state_by_seed[seed]
            static_state = static_state_by_seed[seed]
            static_flips.append(metrics.flip_rate(static_state["acc"], run.true_acc))

            periodic = PoolEval(drift_cfg).evaluate(run)
            periodic_flips.append(metrics.flip_rate(periodic["acc"], run.true_acc))

            if batch == 0:
                online_state_by_seed[seed] = periodic
            else:
                online_state_by_seed[seed] = PoolEval(drift_cfg).evaluate(
                    run,
                    init={"a": online_state_by_seed[seed]["acc"],
                          "b": online_state_by_seed[seed]["b"],
                          "u": online_state_by_seed[seed]["u"]},
                    max_iters=8,
                )
            online_flips.append(metrics.flip_rate(online_state_by_seed[seed]["acc"], run.true_acc))

        row = dict(
            batch=batch + 1,
            static=mean_ci(static_flips)[0],
            periodic=mean_ci(periodic_flips)[0],
            online=mean_ci(online_flips)[0],
        )
        per_batch.append(row)
        print(f"{batch + 1:6d}{row['static']:12.3f}{row['periodic']:12.3f}{row['online']:12.3f}")

    save_json("rq8_drift.json", per_batch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    main(**vars(ap.parse_args()))
