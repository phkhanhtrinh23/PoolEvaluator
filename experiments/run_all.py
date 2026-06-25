"""Run all PoolEval experiments end to end and write results/ JSON.

  python experiments/run_all.py [--seeds 8]
"""
import argparse
import run_rq1_ranking
import run_rq2_ablation
import run_rq3_correlation
import run_rq4_kernel
import run_rq5_scaling


def main(seeds=8):
    print("=" * 60, "\nRQ1  Ranking a pool"); run_rq1_ranking.main(seeds=seeds)
    print("=" * 60, "\nRQ2  Ablation");        run_rq2_ablation.main(seeds=seeds)
    print("=" * 60, "\nRQ3  Correlation");     run_rq3_correlation.main(seeds=max(4, seeds // 2 + 1))
    print("=" * 60, "\nRQ4  Kernel");          run_rq4_kernel.main(seeds=seeds)
    print("=" * 60, "\nRQ5  Scaling");         run_rq5_scaling.main(seeds=max(4, seeds // 2 + 1))
    print("\nAll results written to results/*.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    main(**vars(ap.parse_args()))
