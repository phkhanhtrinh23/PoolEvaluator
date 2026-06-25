#!/usr/bin/env bash
# Reproduce the PoolEval paper experiments (RQ1-RQ8). Results -> results/*.json
set -e
cd "$(dirname "$0")/.."
echo ">> RQ1  Ranking a pool (Table: main)"
python experiments/run_rq1_ranking.py --seeds 10
echo ">> RQ2  Ablation (break the gauge)"
python experiments/run_rq2_ablation.py --seeds 8
echo ">> RQ3  Correlated errors and M_eff"
python experiments/run_rq3_correlation.py --seeds 5
echo ">> RQ4  Equivalence kernel precision/recall"
python experiments/run_rq4_kernel.py --seeds 8
echo ">> RQ5  Scaling with pool size"
python experiments/run_rq5_scaling.py --seeds 5
echo ">> RQ6  Reliability and efficiency"
python experiments/run_rq6_reliability_efficiency.py --seeds 5
echo ">> RQ7  Warm-start optimization"
python experiments/run_rq7_warmstart.py --seeds 5
echo ">> RQ8  Robustness to workload drift"
python experiments/run_rq8_drift.py --seeds 5
echo ">> done. See results/*.json"
