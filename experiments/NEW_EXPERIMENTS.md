# New Experiments (independent review, 2026-07-18)

These experiments were added during an independent review of the paper
*"Label-Free Joint Evaluation and Ranking of Text-to-SQL Model Pools"* (PoolEval-SQL).
They are **not** part of the original paper. All code is in
[`experiments/run_ne_robustness.py`](run_ne_robustness.py); results are written to
`results/ne_misspec.json` and `results/ne_label_budget.json`.

```bash
python experiments/run_ne_robustness.py --seeds 20
```

They target two weaknesses of the paper's empirical evaluation:

1. **Circularity.** The shipped simulator (`pooleval/data/simulator.py`) draws data
   from *exactly* the generative model PoolEval-SQL assumes — IRT ability/difficulty,
   provenance-group shared errors, an *independent* execution verifier, and a Gaussian
   seen prior. When the estimator's model equals the data-generating process (DGP),
   the estimator is guaranteed to look best; the reported margins are therefore partly
   an artifact of the match, not evidence of robustness. **NE-1** breaks each
   assumption one at a time and re-measures.

2. **A missing baseline.** The paper never compares against the obvious practical
   alternative a team actually has: *label a handful of items by hand and rank by the
   measured accuracy.* **NE-2** adds that oracle-sampling baseline (B6) and reports the
   **label-equivalent budget** — how many gold labels buy a ranking as good as
   PoolEval-SQL's label-free estimate.

Every run reuses the shipped estimator and baselines unchanged; only the DGP and the
extra baseline are new. Metrics: **MAE** (accuracy error ×100, ↓), **Flip** (pairwise
ranking-flip rate, ↓), **Kendall-τ** (↑), **Top-1** (↑). 20 seeds each.

---

## NE-1 — Misspecification robustness

For each DGP we compare PoolEval-SQL against the five shipped baselines (B1–B5). The
first row (`matched`) is the paper's own simulator, for reference; the rest each break
one assumption.

| DGP | what it breaks | PoolEval MAE | best baseline MAE | PoolEval Flip | best baseline Flip |
| --- | --- | --- | --- | --- | --- |
| matched (paper sim) | nothing (DGP = estimator model) | **3.81** | 3.59 (B4) | **0.07** | 0.08 (B2/B3) |
| misspec-non-IRT | no shared item-difficulty axis | **1.41** | 3.28 (B2) | **0.02** | 0.02 (B2/B3) |
| misspec-cross-group | errors shared *across* provenance groups (tags anti-informative) | 6.26 | **3.95 (B4)** | **0.09** | 0.13 |
| misspec-corr-verifier | verifier fails where the pool fails (not independent) | 4.18 | **3.59 (B4)** | **0.07** | 0.08 |
| misspec-heavy-gauge | 60% of items are "everyone agrees but wrong" | 8.14 | **4.33 (B4)** | **0.11** | 0.13 |

**Findings.**

- **The ranking (Flip / Kendall) advantage is fairly robust.** PoolEval-SQL keeps the
  lowest flip rate in every DGP, including the three that break its assumptions. This
  supports the paper's *ranking-stability* claim (Prop. 1) more honestly than the
  matched-DGP-only evidence in the paper, because it now holds under misspecification.

- **The absolute-accuracy (MAE) advantage is brittle and can invert.** In three of the
  four misspecified DGPs the simple **Agreement-on-the-line baseline (B4) has clearly
  lower MAE than PoolEval-SQL** (3.95 vs 6.26; 3.59 vs 4.18; 4.33 vs 8.14). The
  provenance-correlation model and the seen-prior/verifier anchor — the paper's
  headline contributions for *absolute level* — actively hurt when their structural
  assumptions are wrong. The paper never surfaces this because it only tests the
  matched DGP.

- **The provenance model is the biggest liability under misspecification.** When the
  real error correlation is *cross-group* rather than within-group (a realistic case:
  two different base models both mis-handle the same dirty value), discounting
  within-group agreement is the wrong move and MAE nearly doubles vs B4.

- **The independent-verifier assumption is load-bearing and idealized.** A real
  execution verifier (does it run? plausible arity/row count?) *cannot* catch a query
  that executes but returns the wrong rows — exactly the errors the pool also makes —
  so its errors are correlated with the pool, not independent as the simulator assumes.
  Under that realistic model (`misspec-corr-verifier`) PoolEval's Top-1 drops from 0.60
  to 0.40 and MAE rises 3.81 → 4.18.

**Takeaway for the paper.** Reframe the contribution as *ranking stabilization* (robust)
and be explicit that *absolute-accuracy* estimation depends on the anchor/correlation
model being well specified. Add a misspecification table like this one so the claims
are defensible beyond the self-matching simulator.

---

## NE-2 — Label-equivalent budget (missing B6 baseline)

B6 labels `k` randomly chosen items by hand and estimates each model's accuracy as its
fraction correct on those `k` (averaged over 5 random subsets per seed). Sweep `k` and
compare to PoolEval-SQL, which uses **zero** labels. `N = 400` items total.

| method | labels used | MAE ↓ | Flip ↓ | Kendall ↑ |
| --- | --- | --- | --- | --- |
| **PoolEval-SQL (ours)** | **0** | **3.20** | **0.068** | **0.86** |
| B6 oracle-label, k=5 | 5 | 17.83 | 0.210 | 0.34 |
| B6 oracle-label, k=10 | 10 | 12.94 | 0.213 | 0.43 |
| B6 oracle-label, k=20 | 20 | 8.68 | 0.180 | 0.56 |
| B6 oracle-label, k=40 | 40 | 5.94 | 0.159 | 0.63 |
| B6 oracle-label, k=80 | 80 | 4.01 | 0.110 | 0.75 |
| B6 oracle-label, k=160 | 160 | 2.45 | 0.070 | 0.84 |

**Findings.**

- **PoolEval-SQL's label-free ranking is worth roughly 160 hand labels out of 400
  (~40% of the workload).** B6 does not reach PoolEval's flip rate (0.068) until
  `k ≈ 160`; below `k = 80` a hand-labeled sample ranks the pool *worse* than the
  label-free estimator. This is a concrete, honest "value of the method" number that
  the paper completely lacks.

- **B6 overtakes on MAE well before it overtakes on ranking.** By `k = 80` the
  hand-labeled estimate already has lower MAE (4.01 vs 3.20 is close; by k=160 it is
  2.45 « 3.20), yet it still ranks the pool worse than PoolEval-SQL until k≈160. This
  reinforces NE-1: the method's real strength is *ranking*, not *absolute accuracy* —
  a modest label budget beats it on MAE much sooner than on flip rate.

**Takeaway for the paper.** A practitioner's true alternative is not only "run a
single-model evaluator" (B1) but "label 20–80 items." Adding B6 and the
label-equivalent-budget curve makes the cost/benefit case the paper is really trying to
make, and it is defensible.

---

## Caveat

Like the paper's own experiments, these run on the **simulator**, so they still cannot
substitute for a real Text-to-SQL model-zoo evaluation on Spider / BIRD / Spider 2.0.
Their purpose is narrower: to show that conclusions drawn *from the simulator alone* are
fragile (NE-1) and to add the practical baseline the paper omits (NE-2). The single most
important missing experiment remains a real-data run — see the review notes.
