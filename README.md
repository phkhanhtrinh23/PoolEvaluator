# PoolEval-SQL — Label-Free Joint Evaluation and Ranking of Text-to-SQL Model Pools

Reference implementation for the paper **"Label-Free Joint Evaluation and Ranking of
Text-to-SQL Model Pools"** (the framework is named **PoolEval-SQL**; this repo ships
it as the `pooleval` package with the `PoolEval` estimator class).

A team that runs Text-to-SQL rarely owns one model — it keeps a *pool* of versions,
fine-tunes, and vendor APIs, and must decide, on each new **unlabeled** private
database, *which model to deploy and how accurate each is*. Existing label-free
evaluators score **one model at a time** and lean on a shift calibration that decays
under the very shift that makes evaluation necessary. **PoolEval-SQL** evaluates the
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

PoolEval-SQL realizes one coupling — the pool scored against a shared latent answer —
anchored by a seen prior and an execution verifier, refined by a graded kernel and
precision fusion. The components and the baselines we compare against:

| Method | Role | Paper § | Code |
| --- | --- | --- | --- |
| **PoolEval-SQL (ours)** | **anchored, correlation-aware latent-correctness EM over the pool** | **§4** | **[`pooleval/`](pooleval/)** |
| ├ Graded execution-equivalence kernel (LA0/LA1/LA2) | ground agreement in graded equivalence (R4) | §4.1 | [`pooleval/kernel.py`](pooleval/kernel.py) |
| ├ Correlation-aware latent correctness (IRT + provenance loadings) | couple the pool, discount near-clones (R1,R3) | §4.2 | [`pooleval/latent.py`](pooleval/latent.py) |
| ├ Seen-prior + execution-verifier anchors | break the gauge (R2) | §4.3 | [`pooleval/latent.py`](pooleval/latent.py) |
| ├ Shift-adaptive precision fusion | weight prior vs agreement by reliability | §4.4 | [`pooleval/latent.py`](pooleval/latent.py) |
| └ Estimator (ranking, intervals, M_eff, collusion flag) | outputs | §4.5 | [`pooleval/inference.py`](pooleval/inference.py) |
| B1 Independent (Garg et al., 2022) | single-model label-free eval, run once per model | §6 | [`baselines/independent/`](baselines/independent/) |
| B2 Majority / self-consistency (Wang et al., 2023) | accuracy = agreement with execution-majority | §6 | [`baselines/majority/`](baselines/majority/) |
| B3 Dawid–Skene (1979) | latent-truth crowd model, no prior, independent errors | §6 | [`baselines/dawid_skene/`](baselines/dawid_skene/) |
| B4 Agreement-on-the-line (Baek et al., 2022) | tuned agreement→accuracy linear trend | §6 | [`baselines/agreement_line/`](baselines/agreement_line/) |
| B5 LLM-as-judge (preference judge, no execution) | simulator proxy for judge-style scoring | §6 | [`baselines/llm_judge/`](baselines/llm_judge/) |

---

## How the method works (the EM in [`pooleval/latent.py`](pooleval/latent.py))

**The one-sentence picture.** On a new database we never see the gold SQL, so *the
correct answer to each question is a hidden variable*. PoolEval-SQL treats the pool as
a crowd of graders and runs **expectation–maximization (EM)** to jointly (a) infer each
question's hidden correct answer and (b) score each model against those inferred
answers — bootstrapping one from the other until they stop changing. It is an
**inference procedure run fresh on a single unlabeled target**: there is no gradient
descent and no weights learned across datasets.

### What is estimated vs. what is held fixed

EM solves for four things and treats everything else as fixed data or a fixed anchor.

| Quantity (code name) | Meaning | Status in EM |
| --- | --- | --- |
| `latent_hat[i]` (`z_i`) | the hidden correct answer (result class) of question *i* | **latent variable** — re-inferred every E-step |
| `a` (`a_m`) | per-model accuracy / IRT "ability" — **the deliverable** | **estimated** — updated every M-step |
| `b` (`b_i`) | per-question difficulty | **estimated** — updated every M-step |
| `u` (`u_g`) | per-provenance-group shared-error loading | **estimated** — updated every M-step |
| `obs[m,i]` | observed result-equivalence classes from the kernel | **fixed data** |
| `run.prior` (`π_m`), `run.prior_sigma` (`σ_m`) | each model's seen single-model calibration | **fixed anchor** |
| `run.verifier_guess` (`v_i`) | the execution verifier's guess of the true answer | **fixed anchor** |
| `run.group` (`G(m)`) | provenance tag (which base model it derives from) | **fixed input** |

Nothing on the "fixed" rows is trained here. In particular the seen priors `π_m` were
fit **earlier**, by an external single-model calibrator on a *labeled source* domain;
they enter EM only as the anchor that pins the accuracy level.

### What "maximization" maximizes

EM maximizes the log-posterior of the whole problem (paper Eq. *objective*, §III) over
the parameters `a, b, u` and the latent answers `z`:

```
maximize  Σ_{m,i} log p(result_{m,i} | z_i, a_m, b_i)   agreement likelihood: every model scored
                                                         against the SAME latent answer, with a
                                                         provenance-group correlated-error discount
        + Σ_i     log p(v_i | z_i)                       verifier: the database is an independent grader
        + Σ_m     log π_m(a_m)                            seen prior: anchors each model's absolute level
```

The first term is the whole point of evaluating the pool *jointly*: a single-model
evaluator never sees that two models agreed, but here every pairwise agreement is
evidence about the shared `z_i`, hence about both models' accuracies. The verifier and
prior terms pin the level (see *why the anchors matter* below).

### The two alternating steps

Each iteration of the `for it in range(n_iters)` loop does:

- **E-step — "given the current scores, what was each question's true answer?"**
  Holding `a, b, u` fixed, for every item it builds a **reliability-weighted,
  group-discounted, verifier-nudged vote** over the candidate answers and softmaxes it
  into a posterior `p(z_i = class)`; `latent_hat[i]` is the arg-max.
  - `w = clip(a)` — each model votes with weight equal to its current estimated accuracy
    (a model believed to be 0.8-accurate counts far more than a 0.3-accurate one).
  - `disc = 1/(1 + u_g·(n−1))` — if `n` models from the *same provenance group* vote the
    same class, their weights are shrunk so a colluding clique of near-clones counts as
    ≈ **one** independent vote, not `n`.
  - the verifier adds `verifier_strength` to whichever class it points at.

- **M-step — "given those answers, how accurate is each model?"**
  Holding the latent answers fixed, each model's raw accuracy is
  `a_agree[m] = mean_i (obs[m,i] == latent_hat[i])` — the fraction of questions where it
  matched the inferred truth — then pulled toward the seen prior by **inverse-variance
  (precision) fusion**: an uncertain prior yields to the consensus, a near-clone pool
  yields to the prior. Difficulties `b_i` are read from how many models got item *i*
  right; loadings `u_g` from how much a group agrees *on wrong answers* beyond the
  cross-group baseline (`_estimate_loadings`).

The two steps reinforce each other — better accuracies sharpen the weighted vote, which
sharpens the latent answers, which sharpen the accuracies — and the loop runs until
`a, b, u` stop changing (`delta < tol`), typically a handful of iterations. The
accuracy update deliberately matches against the **hard** (arg-max) latent estimate:
soft averaging would shrink every model toward 0.5 and lose the absolute level.

### Why the anchors and the group discount are load-bearing

Agreement *alone* is **gauge-ambiguous**: a pool that is all-correct and a pool that
all-agrees-on-the-same-wrong-answer produce identical agreement, so the absolute
accuracy level is unidentifiable. The **seen prior** and the **execution verifier** (an
independent channel — a query that fails to run or returns an implausible result is
wrong no matter who produced it) break that tie. Separately, models sharing a base
model fail *together*, so a naive majority over-credits a colluding clique; the
**group loadings `u_g`** down-weight within-group agreement and feed the reported
**effective independent-model count** `M_eff = M / (1 + c̄·(M−1))`. The ablations (RQ2)
switch each of these off (`use_prior`, `use_verifier`, `use_correlation`) to show the
specific failure it prevents.

### Nothing is trained across datasets

`run_em` is called once per target `PoolRun` and estimates `a, b, u` from that target's
unlabeled outputs alone. The **only** cross-dataset learning is the optional warm-start
(RQ7, [`run_rq7_warmstart.py`](experiments/run_rq7_warmstart.py)), which meta-learns a
better *initial* `a` so EM converges in fewer iterations — the answer at convergence is
unchanged.

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

## Reproduce the paper experiments

```bash
bash scripts/reproduce_main.sh          # or: powershell scripts/reproduce_main.ps1
# or individually:
python experiments/run_rq1_ranking.py     --seeds 10   # Table: ranking a pool
python experiments/run_rq2_ablation.py    --seeds 8    # break the gauge (ablation)
python experiments/run_rq3_correlation.py --seeds 5    # correlated errors & M_eff
python experiments/run_rq4_kernel.py      --seeds 8    # equivalence-kernel P/R
python experiments/run_rq5_scaling.py     --seeds 5    # scaling with pool size
python experiments/run_rq6_reliability_efficiency.py --seeds 5 # coverage, ECE, cost, budget
python experiments/run_rq7_warmstart.py    --seeds 5    # warm-start optimization
python experiments/run_rq8_drift.py        --seeds 5    # workload drift
python experiments/run_all.py             --seeds 8    # everything -> results/*.json
```

---

## Extension: Active PoolEval-SQL (label-efficient, judge-in-the-loop)

Pure PoolEval-SQL is label-free but can only ever pick a latent answer from the classes
the pool **produced**. When every model is wrong — especially when a near-clone clique
agrees on the *same* wrong SQL and a plausible result fools the verifier — the correct
answer is **absent from the candidate set**, and consensus confidently credits a wrong
class (the **candidate-coverage** limitation). **Active PoolEval-SQL** keeps the
framework unchanged and spends a strong label-free judge (**gpt-5-mini**: reads the
question, executes the SQL, aligns result-to-question — no gold) *only* on the most
ambiguous items, chosen by **greedy submodular maximization** (≥ (1−1/e)·OPT,
Nemhauser–Wolsey–Fisher 1978), then pins each verdict as a hard EM constraint and
re-solves with incremental EM.

```bash
python experiments/run_active.py --seeds 20     # candidate-gap DGP + budget sweep + abstention
python -m zoo.run_active --budget 12            # REAL Spider zoo + gpt-5-mini judge
python -m zoo.run_active --budget 12 --mock     # same pipeline, oracle judge, no API cost
```

On a controlled candidate-gap DGP where pure PoolEval-SQL inflates a colluding clique
(true acc **0.357 → 0.582**) and picks it #1, the provenance-aware submodular strategy
recovers the correct deployment decision **85 % of the time at a 10 % label budget** (vs
30 % random, and **15 % for uncertainty sampling** — the dangerous items look *confident*,
so entropy-driven acquisition is the *worst*). Full write-up and tables:
**[`experiments/ACTIVE_POOLEVAL.md`](experiments/ACTIVE_POOLEVAL.md)**.
Code: [`pooleval/active.py`](pooleval/active.py), [`zoo/judge.py`](zoo/judge.py).

**Real model zoos across 5 datasets.** The gpt-5-mini judge runs on real pools for
**Spider, SQLFlow, BIRD, BIRD-MiniDev, and Spider 2.0-lite** (10 models, 12-question
budget). With that tiny budget, Active PoolEval **lowers ranking error (MAE) on every
dataset**, the judge's *"none"* rate **tracks difficulty** (0 on Spider → 4 on
BIRD-MiniDev), and the largest ranking win lands on the **hardest** benchmark —
**Spider 2.0-lite** (pool EX just 5 %), where it improves Kendall **0.79 → 0.85** and
fixes the Top-1 deployment choice. Table, definitions, analysis, and reproduction
commands: **[`experiments/MULTI_DATASET_RESULTS.md`](experiments/MULTI_DATASET_RESULTS.md)**.

Deeper studies — **prior ablation** (the seen prior is load-bearing on real data;
removing it roughly doubles MAE on hard sets), **judge reliability** (gpt-5-mini is a
reliable *"none"*-detector but a weak *picker* — which explains the small BIRD dip),
**bootstrap CIs** (the MAE gain is significant on 4/5 datasets), an **oracle budget
sweep** (more budget monotonically lowers MAE), and a **synthesis-judge ablation** (an
honest negative — letting the judge *write* SQL doesn't help on hard data): see
**[`experiments/ANALYSIS.md`](experiments/ANALYSIS.md)**.

**Where does the prior come from?** The seen prior is the anchor that fixes the gauge,
but in the runs above it is measured on a *labeled split of the target benchmark* — the
one thing a deployed operator lacks. [`synsql/`](synsql/README.md) tests replacing it
with a prior estimated on **SynSQL-2.5M** subsets *retrieved* to match the unlabeled
target (2.54M records, 16,583 live DBs, ~1K-item subsets, 8,000 executed generations).
Result, in short: **retrieval alignment is uninformative** (corr(distance, prior error)
≈ 0 on all five targets), but the label-free corpus prior recovers the pool's *relative*
ability almost exactly — after removing a single shared offset its MAE is **2.98–3.89
vs the target-labeled prior's 2.80**. The entire gap is one scalar: the gauge. Buying
just that back costs **~10–40 target labels** instead of a 120-item labeled split,
which composes directly with the judge-in-the-loop budget above. A zero-cost
source x target matrix over all five benchmarks shows the level gap is a property of
being **out-of-domain**, not of SynSQL being synthetic — Spider→BIRD is off by 40.5 and
Spider→Spider 2.0-local by 72.3, both worse than SynSQL — while centered MAE stays in
1.8–4.4 for *every* source. Full write-up:
**[`experiments/SYNSQL_PRIOR.md`](experiments/SYNSQL_PRIOR.md)**.

---

## Results (this repository's simulator)

**RQ1 — Ranking a diverse `M=12` pool** (`--seeds 10`; lower MAE/Flip better, higher
Kendall/Top-k better). PoolEval-SQL is best on accuracy MAE and on the
ranking-difference metrics (flip rate, Kendall-τ) it is designed to improve; the
preference-based judge (B5) trails the execution-grounded methods, as in the paper:

| Method | MAE ↓ | Flip ↓ | Kendall-τ ↑ | Top-1 ↑ | Top-3 ↑ |
| --- | --- | --- | --- | --- | --- |
| B1 Independent | 7.20 | 0.25 | 0.49 | 0.20 | 0.60 |
| B2 Majority / self-cons. | 3.73 | 0.09 | 0.81 | 0.50 | 0.83 |
| B3 Dawid–Skene | 3.69 | 0.10 | 0.79 | 0.50 | 0.83 |
| B4 Agreement-on-the-line | 3.66 | 0.11 | 0.78 | 0.50 | 0.80 |
| B5 LLM-as-judge | 6.50 | 0.25 | 0.48 | 0.50 | 0.57 |
| **PoolEval-SQL (ours)** | **3.24** | **0.07** | **0.86** | 0.50 | 0.83 |

**RQ2 — Breaking the gauge (ablation, `--seeds 8`).** Removing any component raises
accuracy MAE; the kernel and the anchors are the load-bearing ones:

| Variant | MAE ↓ | Flip ↓ |
| --- | --- | --- |
| **PoolEval-SQL (full)** | **3.19** | **0.06** |
| − seen prior (R2) | 3.54 | 0.04 |
| − execution verifier (R2) | 4.48 | 0.10 |
| − correlation model (R3) | 3.82 | 0.08 |
| − graded kernel → exact match (R4) | 8.37 | 0.10 |
| − all (≈ B3) | 7.62 | 0.12 |

**RQ3 — Correlated errors & `M_eff` (`--seeds 5`).** As within-pool correlation
grows, PoolEval-SQL stays accurate (and below B3), and the reported effective-independent
count falls from ~`M` toward the number of independent provenance groups:

| collusion | B3 MAE | PoolEval-SQL MAE | M_eff (of 12) |
| --- | --- | --- | --- |
| 0.00 | 3.60 | 3.73 | 11.8 |
| 0.40 | 3.95 | 3.42 | 11.3 |
| 0.80 | 4.08 | 3.00 | 9.8 |
| 0.95 | 3.86 | 2.45 | 9.4 |

**RQ4 — Equivalence kernel precision/recall vs gold (`--seeds 8`).** LA0 exact match
has high precision but poor recall; LA1 canonicalization recovers recall; LA2
multi-instance raises precision:

| Level | Precision | Recall |
| --- | --- | --- |
| LA0 exact | 0.999 | 0.549 |
| LA1 canonical | 0.995 | 0.870 |
| LA2 multi-instance | 1.000 | 0.940 |

**RQ5 — Scaling with pool size (`--seeds 5`).** PoolEval-SQL's ranking stays sharp as the
pool grows; the independent baseline scores each model in isolation and does not
improve:

| M | B1 Flip | PoolEval-SQL Flip | B1 Kendall | PoolEval-SQL Kendall |
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
| §4.1 Graded execution-equivalence kernel (LA0–LA2) | [`pooleval/kernel.py`](pooleval/kernel.py) |
| §4.2 IRT coupling + provenance-group loadings, M_eff | [`pooleval/latent.py`](pooleval/latent.py) |
| §4.3 Seen prior + execution verifier | `Config.use_prior/use_verifier`, simulator anchors |
| §4.4 Shift-adaptive precision (inverse-variance) fusion | `latent.py` M-step (`fusion="precision"`) |
| §4.5 Inference outputs (ranking, intervals, M_eff, flag) | [`pooleval/inference.py`](pooleval/inference.py) |
| Prop. 4.1 (joint estimation ↓ ranking-difference variance) | RQ1/RQ5 flip-rate & Kendall reductions |
| Conformal intervals calibrated on a held-out slice (§4.5) | [`experiments/run_rq6_reliability_efficiency.py`](experiments/run_rq6_reliability_efficiency.py) `calibrate_scales` |
| §6 Baselines B1–B5 | [`baselines/`](baselines/) |
| §6 RQ1–RQ8 | [`experiments/`](experiments/) |

## Repository layout

```
pooleval/        core: config, simulator, kernel, latent EM, fusion, inference, metrics, active (Active PoolEval-SQL)
baselines/       B1 Independent, B2 Majority, B3 Dawid–Skene, B4 Agreement-on-the-line, B5 LLM-as-judge (preference proxy)
experiments/     run_rq1..run_rq8, run_all; run_active (candidate-coverage + judge budget); ACTIVE_POOLEVAL.md
zoo/             real multi-dataset pipeline; datasets.py (Spider/BIRD/SQLFlow/Spider2 loaders), judge.py (gpt-5-mini RealJudge), run_multi.py, run_active.py
synsql/          SynSQL-2.5M as a retrievable prior corpus: ingest, ~1K-subset partitionings, MMD retrieval, subset probing, gauge label-budget, cross-benchmark prior matrix
configs/         default.yaml (mirrors pooleval/config.py)
scripts/         reproduce_main.sh / .ps1
tests/           test_smoke (pipeline; PoolEval-SQL lowest flip), test_active (constraint EM, judge, submodular, recovery)
```

## Tests

```bash
python tests/test_smoke.py && python tests/test_active.py     # or: pytest -q
```

## Experimental closed-form formulation

The full beginner-friendly derivation of the collision-aware E-step, auxiliary
objective $Q$, and numerical beta M-step is in
[`docs/collision_em_beta_derivation.md`](docs/collision_em_beta_derivation.md).

The alternative pseudo-label agreement EM in `new_formulation/` is implemented
alongside the original estimator in [`pooleval/new_formulation.py`](pooleval/new_formulation.py).
It uses old PoolEval's kernel/prior/provenance/verifier score to fix a pseudo-label,
then applies the document's closed-form EM updates to the binary agreement matrix.
The paired RQ1--RQ8 comparison and full interpretation are in
[`experiments/NEW_FORMULATION_RESULTS.md`](experiments/NEW_FORMULATION_RESULTS.md).
The evaluation on saved real Spider, BIRD, SQLFlow, and BIRD-MiniDev model-zoo
artifacts is reported in
[`experiments/NEW_FORMULATION_REAL_RESULTS.md`](experiments/NEW_FORMULATION_REAL_RESULTS.md).
The collision-aware case-3 extension and its leave-one-dataset-out real evaluation
are reported in
[`experiments/COLLISION_FORMULATION_REAL_RESULTS.md`](experiments/COLLISION_FORMULATION_REAL_RESULTS.md).
The replay of saved real `gpt-5-mini` pseudo-label corrections after case-3 EM is
reported in
[`experiments/COLLISION_ACTIVE_REAL_RESULTS.md`](experiments/COLLISION_ACTIVE_REAL_RESULTS.md).
The complete rerun is reported in
[`experiments/COLLISION_RERUN_2026_08_14.md`](experiments/COLLISION_RERUN_2026_08_14.md).

```bash
python experiments/run_new_formulation_comparison.py --seeds 8
python -m zoo.new_formulation_real --bootstrap 500
python -m zoo.collision_formulation_real --bootstrap 500
python -m zoo.collision_active_real --bootstrap 400
```

## Beyond Text2SQL: image and node classification

`pooleval/domains/` ports the estimator to image classification (MNIST -> USPS /
SVHN, MetaEvaluator's architectures) and node classification (ACMv9 / Citationv1 /
DBLPv7, GNNEvaluator's GNNs) on real trained pools. The EM core is unchanged --
only the observation kernel, prior, and verifier are domain-specific. Results,
the three anchor defects the ports exposed, and the `verifier_mode="learned"`
mitigation are in [`docs/domain_ports.md`](docs/domain_ports.md).

```bash
python experiments/run_domain_graph.py
python experiments/run_domain_vision.py
python experiments/run_domain_diagnostics.py --kind graph
```

[`docs/multiclass_ds.md`](docs/multiclass_ds.md) works through what the right
estimator is once the answer space is a closed set of K classes: why the
collision rate `gamma` is not needed there, the closed-form multiclass EM, a
measurement showing Dawid--Skene's uniform-error assumption is violated by
1.9x-6.4x on real pools, and the confusion-matrix fix that turns MNIST -> SVHN
from a total failure (rho -0.886) into a solved case (rho +0.829).

It also reports which estimator wins in which regime -- small target set, many
classes, imbalanced classes -- with the crossover set by observations per free
parameter (PoolEval's parameter count is independent of K; full DS's grows as
K^2).

```bash
python experiments/run_ds_assumption.py
python experiments/run_regime_scenarios.py
python experiments/run_gamma_ablation.py
```

## License

MIT — see [LICENSE](LICENSE).
