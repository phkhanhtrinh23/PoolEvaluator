# Real Text-to-SQL Zoo on Spider — Measured Results (2026-07-19)

The paper's numbers all come from a simulator whose data-generating process equals the
estimator's own model (see the review). This is the **first real-data run**: a pool of
real LLMs generating SQL, executed on the live Spider databases, evaluated by the
shipped PoolEval-SQL estimator and baselines against true execution accuracy.

- **Pipeline / how to reproduce:** [`zoo/README.md`](../zoo/README.md);
  `python -m zoo.run --n-target 150 --n-source 120` then `python -m zoo.bootstrap`.
- **Artifacts:** `zoo_artifacts/poolrun_dev.npz`, `results/zoo_spider_main.json`,
  `results/zoo_spider_bootstrap.json`.

## Setup

- **Benchmark:** Spider dev (target), 150 items spread across databases; Spider train
  (source, 120 items) supplies each model's **real seen prior** (its EX on the labeled
  source domain, transferred to the unlabeled dev target). Train/dev share no databases.
- **Pool:** M=10 real models over **7 provenance groups**, three groups with a
  within-group near-clone pair (same base, different prompt) — true independent ≈7 < M.
  OpenAI-only (the Together key is 403 on this machine); see deviations in the README.
- **Estimand:** execution accuracy (EX). Gold used **only** to score.

## Measured per-model accuracy (the inflation, in one table)

| member | group | true EX | seen prior (src) | PoolEval-SQL pred |
| --- | --- | --- | --- | --- |
| gpt4o-schema | gpt-4o | 0.760 | 0.808 | 0.944 |
| gpt4o-minimal | gpt-4o | 0.760 | 0.775 | 0.891 |
| gpt4omini-schema | gpt-4o-mini | 0.727 | 0.767 | 0.889 |
| gpt4omini-fewshot | gpt-4o-mini | 0.733 | 0.800 | 0.847 |
| gpt41 | gpt-4.1 | 0.767 | 0.767 | 0.897 |
| gpt41-mini | gpt-4.1-mini | **0.800** | 0.775 | 0.914 |
| gpt41-nano | gpt-4.1-nano | 0.733 | 0.725 | 0.861 |
| gpt4-turbo | gpt-4-turbo | 0.760 | 0.792 | 0.880 |
| gpt35-schema | gpt-3.5-turbo | 0.693 | 0.792 | 0.812 |
| gpt35-fewshot | gpt-3.5-turbo | 0.693 | 0.733 | 0.830 |
| **mean** | | **0.743** | **0.773** | **0.876** |

The pool's mean pairwise agreement is **0.864** while it is only **0.743** correct: the
models **agree on wrong answers 12 points more often than they are right**, because a
single-vendor pool shares failure modes. The latent-consensus reads that shared error as
correctness, so PoolEval-SQL's estimate is inflated to **0.876** (+13 pts over truth).
The accurate prior (0.773, +3 pts) is diluted by the precision fusion, which over-weights
the low-variance (but biased) agreement estimate.

## Ranking vs baselines (item-bootstrap, 300 resamples; mean [95% CI])

| Method | MAE ↓ | Flip ↓ | Kendall ↑ | Top-1 ↑ |
| --- | --- | --- | --- | --- |
| B1 Independent (prior) | **4.77** [2.4, 9.7] | 0.37 [0.27, 0.47] | 0.16 [-0.05, 0.36] | 0.01 |
| B2 Majority / self-cons. | 16.95 [11.9, 22.6] | **0.14** [0.02, 0.28] | **0.67** [0.34, 0.89] | **0.44** |
| B3 Dawid–Skene | 16.94 [11.9, 22.6] | 0.15 [0.03, 0.33] | 0.64 [0.26, 0.90] | 0.46 |
| B4 Agreement-on-line | 4.52 [1.7, 9.7] | 0.28 [0.09, 0.74] | 0.40 [-0.55, 0.79] | 0.06 |
| **PoolEval-SQL (ours)** | 13.74 [8.5, 19.4] | 0.18 [0.04, 0.34] | 0.60 [0.25, 0.86] | 0.13 |

## Findings (honest)

1. **PoolEval-SQL does not win on the real zoo.** It is *not* best on a single metric.
   The paper's headline — "best on every metric of every benchmark" — does not hold on
   real data.

2. **It loses badly on absolute accuracy (MAE).** At 13.7 MAE it is beaten ~3× by the
   trivial prior baseline B1 (4.8) and by B4 (4.5), with non-overlapping CIs. The anchor
   + verifier + correlation machinery makes the estimate *worse* than just reporting the
   seen prior. (It does beat the un-anchored B2/B3 at ~17, so the anchor helps *some* —
   just not enough to be useful.)

3. **On ranking it is competitive but not best.** Plain majority/self-consistency (B2)
   matches or beats it on every ranking metric — Flip 0.14 vs 0.18, Kendall 0.67 vs 0.60,
   Top-1 0.44 vs 0.13. CIs are wide (a near-saturated pool with tiny EX spread), so no
   method is *significantly* best on ranking, but PoolEval-SQL is clearly not ahead.

4. **The failure mechanism is exactly `misspec-cross-group` from NE-1.** The real error
   correlation is *cross-group* (all OpenAI bases share failure modes), not within the
   labeled provenance groups. So the within-group discount does nothing: reported
   **M_eff = 8.8 of 10** — the estimator thinks the pool is nearly independent when it is
   effectively far less. The synthetic NE-1 prediction is confirmed on real data.

5. **The MAE-vs-ranking tension is real and visible.** B1 has the best MAE but the worst
   ranking (a good absolute level, near-random order on a tied pool); B2 has the best
   ranking but the worst MAE (right order, inflated level). PoolEval-SQL is supposed to
   get the best of both via fusion; instead it gets a mediocre version of each.

## Important caveats (what this run does and does not show)

- **This is Spider = the paper's *weakest* claimed regime**: low train→dev shift and a
  near-saturated pool (true EX 0.69–0.80). The paper claims its advantage is "largest
  under strong shift and diverse pools" and "vanishes" under low shift / homogeneity — so
  a single low-shift result is not a full refutation. But it *does* refute the
  "best on every benchmark" claim and confirms the method can actively hurt.
- **Single vendor, single seed.** The pool is less provenance-diverse than the paper's
  seven independent bases (Together was unavailable). A truly multi-vendor pool would
  have more cross-group independence and might help the joint estimator — but would also
  need the *correct* group tags to beat the cross-group correlation problem.
- **The real test of the paper's central claim needs a harder benchmark** (BIRD /
  Spider 2.0, higher shift, wider accuracy spread). BIRD's data is unfetched here; the
  pipeline is ready to run it once available.

## What to change in the paper (from this run)

- Drop "best on every metric of every benchmark." On real Spider it is not.
- Report **absolute-accuracy honestly**: the joint estimate is *positively biased* when
  the pool shares failure modes; the fusion needs a bias term or a much stronger anchor
  weight, or the method should report **ranking only** and not per-model EX.
- Fix M_eff / the correlation model to detect **cross-group** correlation (agreement
  clustering), not only operator-tagged within-group correlation — otherwise a
  single-vendor pool is scored as if independent.
- Add real-zoo results (this run) and, critically, a **BIRD/Spider 2.0 high-shift** run
  before making the "helps most under shift" claim.
