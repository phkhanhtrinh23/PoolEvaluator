# Reproduce the main PoolEval experiments (RQ1-RQ5). Results -> results/*.json
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
Write-Host ">> RQ1  Ranking a pool (Table: main)"
python experiments/run_rq1_ranking.py --seeds 10
Write-Host ">> RQ2  Ablation (break the gauge)"
python experiments/run_rq2_ablation.py --seeds 8
Write-Host ">> RQ3  Correlated errors and M_eff"
python experiments/run_rq3_correlation.py --seeds 5
Write-Host ">> RQ4  Equivalence kernel precision/recall"
python experiments/run_rq4_kernel.py --seeds 8
Write-Host ">> RQ5  Scaling with pool size"
python experiments/run_rq5_scaling.py --seeds 5
Write-Host ">> done. See results/*.json"
