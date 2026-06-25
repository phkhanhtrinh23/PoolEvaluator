# PoolEval — Label-Free Joint Evaluation of a Text-to-SQL Model Pool

Reference implementation for the paper **"Evaluating a Pool of Text-to-SQL Models
on Unlabeled Data via Anchored Latent-Correctness Inference" (PoolEval)**.

A team that runs Text-to-SQL rarely owns one model — it keeps a *pool* of versions,
fine-tunes, and vendor APIs, and must decide, on each new **unlabeled** private
database, *which model to deploy and how accurate each is*. Existing label-free
evaluators score **one model at a time** and lean on a shift calibration that decays
under the very shift that makes evaluation necessary. **PoolEval** evaluates the
*whole pool jointly*: it treats the unlabeled questions as items and the models as
graders of a latent correct answer, so their **mutual agreement** becomes evidence
no single-model estimator can see. Executing the queries on the live database
grounds that agreement and supplies an independent verifier; each model's existing
calibration anchors the absolute level; and a provenance-grouped error model plus a
graded equivalence kernel keep near-clones and brittle exact matching from
corrupting the estimate.

> **Note on data.** The paper evaluates real model *zoos* on Spider / BIRD /
> Spider 2.0; those checkpoints and databases are not redistributable. This repo
> ships a faithful **simulator** of the evaluation problem (latent correctness,
> provenance-correlated errors, seen priors, an execution verifier, a graded
> equivalence kernel) so the estimator, the baselines, and every experiment run
> end-to-end on a commodity CPU in seconds. All reported numbers below come from
> that simulator; plugging in real pool outputs only requires replacing
> `pooleval/data/simulator.py` with a loader that returns the same `PoolRun`.

---

## Method library

PoolEval realizes one coupling — the pool scored against a shared latent answer —
anchored by a seen prior and an execution verifier, refined by a graded kernel and
precision fusion. The components and the baselines we compare against:

| Method | Role | Paper § | Code |
| --- | --- | --- | --- |
| **PoolEval (ours)** | **anchored, correlation-aware latent-correctness EM over the pool** | **§4** | **[`pooleval/`](pooleval/)** |
| ├ Graded execution-equivalence kernel (L0/L1/L2) | ground agreement in graded equivalence (R4) | §4.1 | [`pooleval/kernel.py`](pooleval/kernel.py) |
| ├ Correlation-aware latent correctness (IRT + provenance loadings) | couple the pool, discount near-clones (R1,R3) | §4.2 | [`pooleval/latent.py`](pooleval/latent.py) |
| ├ Seen-prior + execution-verifier anchors | break the gauge (R2) | §4.3 | [`pooleval/latent.py`](pooleval/latent.py) |
| ├ Shift-adaptive precision fusion | weight prior vs agreement by reliability | §4.4 | [`pooleval/latent.py`](pooleval/latent.py) |
| └ Estimator (ranking, intervals, M_eff, collusion flag) | outputs | §4.5 | [`pooleval/inference.py`](pooleval/inference.py) |
| B1 Independent (Garg et al., 2022) | single-model label-free eval, run once per model | §5 | [`baselines/independent/`](baselines/independent/) |
| B2 Majority / self-consistency (Wang et al., 2023) | accuracy = agreement with execution-majority | §5 | [`baselines/majority/`](baselines/majority/) |
| B3 Dawid–Skene (1979) | latent-truth crowd model, no prior, independent errors | §5 | [`baselines/dawid_skene/`](baselines/dawid_skene/) |
| B4 Agreement-on-the-line (Baek et al., 2022) | tuned agreement→accuracy linear trend | §5 | [`baselines/agreement_line/`](baselines/agreement_line/) |

---

## Install

```bash
cd majority_voting_model_eval_code
python -m pip install -r requirements.txt   # numpy, scipy, scikit-learn, pyyaml
# optional: pip install -e .
```

## Quickstart

```python
from pooleval import Config, PoolEval, simulate, metrics

cfg  = Config(seed=0)                  # default diverse M=12 pool, 7 provenance groups
run  = simulate(cfg)                   # a pool run (gold withheld; used only to score)
out  = PoolEval(cfg).evaluate(run)     # joint latent-correctness inference

print(metrics.all_metrics(out["acc"], run.true_acc))   # MAE, Flip, Kendall, Top-1/3
print("ranking :", out["ranking"])     # best-first model order
print("M_eff   :", round(out["Meff"], 1), "of", cfg.M) # effective independent models
print("collusion flag:", out["collusion"])
```

## Reproduce the main experiments

```bash
bash scripts/reproduce_main.sh          # or: powershell scripts/reproduce_main.ps1
# or individually:
python experiments/run_rq1_ranking.py     --seeds 10   # Table: ranking a pool
python experiments/run_rq2_ablation.py    --seeds 8    # break the gauge (ablation)
python experiments/run_rq3_correlation.py --seeds 5    # correlated errors & M_eff
python experiments/run_rq4_kernel.py      --seeds 8    # equivalence-kernel P/R
python experiments/run_rq5_scaling.py     --seeds 5    # scaling with pool size
python experiments/run_all.py             --seeds 8    # everything -> results/*.json
```

---

## Results (this repository's simulator)

**RQ1 — Ranking a diverse `M=12` pool** (`--seeds 10`; lower MAE/Flip better, higher
Kendall/Top-k better). PoolEval is best on accuracy MAE and on the
ranking-difference metrics (flip rate, Kendall-τ) it is designed to improve:

| Method | MAE ↓ | Flip ↓ | Kendall-τ ↑ | Top-1 ↑ | Top-3 ↑ |
| --- | --- | --- | --- | --- | --- |
| B1 Independent | 7.20 | 0.25 | 0.49 | 0.20 | 0.60 |
| B2 Majority / self-cons. | 3.73 | 0.09 | 0.81 | 0.50 | 0.83 |
| B3 Dawid–Skene | 3.69 | 0.10 | 0.79 | 0.50 | 0.83 |
| B4 Agreement-on-the-line | 3.66 | 0.11 | 0.78 | 0.50 | 0.80 |
| **PoolEval (ours)** | **3.24** | **0.07** | **0.86** | 0.50 | 0.83 |

**RQ2 — Breaking the gauge (ablation, `--seeds 8`).** Removing any component raises
accuracy MAE; the kernel and the anchors are the load-bearing ones:

| Variant | MAE ↓ | Flip ↓ |
| --- | --- | --- |
| **PoolEval (full)** | **3.19** | **0.06** |
| − seen prior (R2) | 3.54 | 0.04 |
| − execution verifier (R2) | 4.48 | 0.10 |
| − correlation model (R3) | 3.82 | 0.08 |
| − graded kernel → exact match (R4) | 8.37 | 0.10 |
| − all (≈ B3) | 7.62 | 0.12 |

**RQ3 — Correlated errors & `M_eff` (`--seeds 5`).** As within-pool correlation
grows, PoolEval stays accurate (and below B3), and the reported effective-independent
count falls from ~`M` toward the number of independent provenance groups:

| collusion | B3 MAE | PoolEval MAE | M_eff (of 12) |
| --- | --- | --- | --- |
| 0.00 | 3.60 | 3.73 | 11.8 |
| 0.40 | 3.95 | 3.42 | 11.3 |
| 0.80 | 4.08 | 3.00 | 9.8 |
| 0.95 | 3.86 | 2.45 | 9.4 |

**RQ4 — Equivalence kernel precision/recall vs gold (`--seeds 8`).** L0 exact match
has high precision but poor recall; L1 canonicalization recovers recall; L2
multi-instance raises precision:

| Level | Precision | Recall |
| --- | --- | --- |
| L0 exact | 0.999 | 0.549 |
| L1 canonical | 0.995 | 0.870 |
| L2 multi-instance | 1.000 | 0.940 |

**RQ5 — Scaling with pool size (`--seeds 5`).** PoolEval's ranking stays sharp as the
pool grows; the independent baseline scores each model in isolation and does not
improve:

| M | B1 Flip | PoolEval Flip | B1 Kendall | PoolEval Kendall |
| --- | --- | --- | --- | --- |
| 5 | 0.18 | 0.08 | 0.64 | 0.84 |
| 12 | 0.26 | 0.06 | 0.47 | 0.87 |
| 20 | 0.28 | 0.07 | 0.44 | 0.86 |

*(Numbers are deterministic per seed; minor differences across `--seeds` reflect the
coarse, binary Top-1 metric on near-tied pools.)*

---

## How the code maps to the paper

| Paper | Code |
| --- | --- |
| Objective (Eq. agreement+verifier+anchor) | [`pooleval/latent.py`](pooleval/latent.py) `run_em` |
| §4.1 Graded execution-equivalence kernel (L0–L2) | [`pooleval/kernel.py`](pooleval/kernel.py) |
| §4.2 IRT coupling + provenance-group loadings, M_eff | [`pooleval/latent.py`](pooleval/latent.py) |
| §4.3 Seen prior + execution verifier | `Config.use_prior/use_verifier`, simulator anchors |
| §4.4 Shift-adaptive precision (inverse-variance) fusion | `latent.py` M-step (`fusion="precision"`) |
| §4.5 Inference outputs (ranking, intervals, M_eff, flag) | [`pooleval/inference.py`](pooleval/inference.py) |
| Prop. 4.1 (joint estimation ↓ ranking-difference variance) | RQ1/RQ5 flip-rate & Kendall reductions |
| §5 Baselines B1–B4 | [`baselines/`](baselines/) |
| §6 RQ1–RQ5 | [`experiments/`](experiments/) |

## Repository layout

```
pooleval/        core: config, simulator, kernel, latent EM, fusion, inference, metrics
baselines/       B1 Independent, B2 Majority, B3 Dawid–Skene, B4 Agreement-on-the-line
experiments/     run_rq1..run_rq5, run_all (write results/*.json)
configs/         default.yaml (mirrors pooleval/config.py)
scripts/         reproduce_main.sh / .ps1
tests/           smoke tests (pipeline runs; PoolEval has lowest flip rate)
```

## Tests

```bash
python tests/test_smoke.py        # or: pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
