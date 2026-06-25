"""RQ1 -- Ranking a pool (paper Table 'tab:main').

Does evaluating the pool JOINTLY rank its members more reliably than independent
single-model evaluation (B1) and tuned-agreement baselines (B2-B4)? Reports
accuracy MAE and ranking quality on the default diverse M=12 pool.

  python experiments/run_rq1_ranking.py [--seeds 5]
"""
import argparse
from _shared import Config, evaluate_pool, print_table, save_json, METRIC_KEYS


def main(seeds=5):
    make = lambda s: Config(seed=s)            # noqa: E731  (default main-table pool)
    agg = evaluate_pool(make, seeds=seeds)
    order = ["B1 Independent", "B2 Majority/self-cons.", "B3 Dawid--Skene",
             "B4 Agreement-on-line", "PoolEval (ours)"]
    print_table("RQ1: label-free accuracy & ranking on a diverse M=12 pool "
                f"(mean +/- 95% CI over {seeds} seeds; lower MAE/Flip better, "
                "higher Kendall/Top-k better)", agg, order=order)
    save_json("rq1_main.json", {k: {mk: agg[k][mk] for mk in METRIC_KEYS}
                                for k in agg})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    main(**vars(ap.parse_args()))
