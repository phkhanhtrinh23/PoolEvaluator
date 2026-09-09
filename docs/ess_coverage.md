# Testing the coverage bound and the two effective sample sizes

Two derivations were added to the paper:

* `coverage_hoeffding_derivation.md` — dataset coverage is monotone submodular, greedy
  gets `alpha_K >= 1 - 1/e` of optimal coverage, and under two extra assumptions
  (smooth expected correctness, conditionally independent outcomes) coverage plus a
  weighted Hoeffding term bound the error of a **target-matched** calibration
  estimator.
* `MTM08_analysis.md` — the Morita–Thall–Müller curvature-matched effective sample
  size of `Beta(a, b)` is `a + b`, so PoolEval's `Beta(1 + s·pi, 1 + s·(1-pi))` anchor
  carries `ESS = s + 2`, with `s = a0 · n0` a power-prior discount.

Both are now executable: [`pooleval/theory.py`](../pooleval/theory.py) implements
them, [`tests/test_theory.py`](../tests/test_theory.py) verifies the algebra against
the numbers stated in the proofs, and
[`experiments/run_ess_coverage.py`](../experiments/run_ess_coverage.py) runs the
empirical questions the proofs cannot answer.

```
python -m pytest tests/test_theory.py -q      # 14 passed
python experiments/run_ess_coverage.py        # -> results/ess_coverage.json
```

Everything below is measured on the **real** SynSQL calibration probes (800 gradable
items over 32 candidate subsets, 10 real models, real execution) against the five
real Text2SQL targets already in the repo.

---

## The two things both called "effective sample size"

They are different numbers and the code keeps them apart:

| | definition | lives in | what it counts |
|---|---|---|---|
| `n_eff` | `1 / sum_z w_z^2` | the Hoeffding term | how many independent observations a *weighted average* is worth |
| `s + 2` | `a0·n0 + 2` (MTM08) | the MAP update | how many observations a *Beta prior* is worth, by curvature matching |

The `+2` is the uniform `Beta(1,1)` baseline. It contributes curvature but contributes
**zero exponents** to the MAP objective, which is why the prior's mixing weight is
`s / (N + s)` and not `s / (N + s + 2)`. The paper calls `s` the effective sample
size; the MTM08 ESS is `s + 2`. Harmless, but now stated.

## What the tests verify

Every algebraic claim, re-derived numerically rather than quoted:

| claim | check |
|---|---|
| `Beta(3,7)` has ESS 10 | curvature `delta(m)` reproduces the paper's table at `m = 0, 5, 10, 15` and its argmin is 10 |
| `Beta(a,b)` has ESS `a+b` | `delta(a+b) = 0` for arbitrary `a, b` |
| power prior gives `10·a0 + 2` | `Beta(1+3a0, 1+7a0)` for every `a0` |
| ours gives `s + 2`, mode at `pi`, mean ≠ `pi` | §4.3 |
| MAP is `(1-r)·mu_hat + r·pi`, `r = s/(N+s)` | matches `(T + s·pi)/(N + s)` exactly |
| `\|\|phi(x)-phi(z)\|\|² = 4(1-S)` | exact, and `mean d² = 4(1-c)` |
| coverage monotone + submodular | 200 random pairs of nested collections |
| greedy ≥ `alpha_K` × optimum | 25 random instances vs brute force, `K = 2, 3` |
| `1 ≤ n_eff ≤ r ≤ min(n, N)` | 50 random assignments |
| weighted Hoeffding covers at `1-delta` | 20 000 Monte-Carlo draws |
| §16.3 counterexample | reproduces 0.82 / 0.50 / 0.32 / 0.40 exactly |
| §15, §19.12, §19.13 numerics | 0.2158, 5.1234, 0.9544 vs 1.0074, 128.0861 |

**14 passed.** The mathematics is sound. The rest of this document is about whether
it *helps*.

---

## E1 — the geometry of the selection Stage 1 actually makes

`c` is normalized coverage, `n_eff` the concentration ESS, `TV` the weight-mismatch
term the uniform estimator pays, `r_pos` the number of calibration items that are
somebody's nearest neighbour.

| target | N | n | c | raw cos | n_eff | r_pos | TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| spider | 150 | 125 | 0.5699 | 0.140 | 21.59 | 31 | 0.760 |
| bird | 150 | 125 | 0.5567 | 0.113 | 13.47 | 22 | 0.835 |
| bird_minidev | 150 | 125 | 0.5549 | 0.110 | 12.47 | 21 | 0.840 |
| sqlflow | 150 | 125 | 0.5923 | 0.185 | 21.43 | 27 | 0.676 |
| spider2local | 24 | 125 | 0.6063 | 0.213 | 11.08 | 12 | 0.872 |

Because TF-IDF features are non-negative, `S = (1+cos)/2` floors at 0.5, so `c = 0.57`
means a raw best-match cosine of **0.14**. The calibration corpus is nearly orthogonal
to the targets. Of 125 selected labeled items, only 12–31 are ever used as a match.

## E2 — is the target-matched estimator better than the uniform one?

Section 16 argues the current uniform average `pi_j = (1/n)·sum_z Y_j(z)` is the wrong
estimator and the matched `theta_hat_j = sum_z w_z Y_j(z)` is the right one. Same
labeled items, different weights, so reweighting is free.

| target | prior MAE uniform | matched | PoolEval MAE uniform | matched |
|---|---:|---:|---:|---:|
| spider | **22.11** | 36.73 | 8.64 | **5.85** |
| bird | **16.73** | 22.00 | **23.04** | 25.42 |
| bird_minidev | **26.85** | 36.27 | **27.28** | 31.96 |
| sqlflow | 9.20 | **8.93** | 14.24 | **12.61** |
| spider2local | 48.92 | **35.42** | 39.85 | **30.84** |
| **mean delta** | | **+3.11** | | **−1.27** |

**Matching makes the prior worse and the downstream estimate marginally better.** The
mechanism is visible in the bound itself. Matching does not change coverage at all —
both estimators use the same selected items — so it buys **zero** transfer improvement
and pays a pure concentration cost:

| | Hoeffding term |
|---|---:|
| uniform, `n = 125` | 0.121 |
| matched, `n_eff = 11–22` | 0.292 – 0.408 |

Matching multiplies the sampling term by **2.4–3.4×**. The only thing it buys back is
the TV mismatch term, and that term is a worst case that never binds:

| target | measured \|uniform − matched\| | TV bound | looseness |
|---|---:|---:|---:|
| spider | 0.219 | 0.760 | 3.5× |
| bird | 0.135 | 0.835 | 6.2× |
| bird_minidev | 0.196 | 0.840 | 4.3× |
| sqlflow | 0.128 | 0.676 | 5.3× |
| spider2local | 0.226 | 0.872 | 3.9× |

So the bound and the data disagree about which estimator to use:

| target | B uniform | B matched | bound prefers | err uniform | err matched | data prefers |
|---|---:|---:|---|---:|---:|---|
| spider | 1.354 | 0.765 | matched | 0.221 | 0.367 | uniform |
| bird | 1.436 | 0.850 | matched | 0.167 | 0.220 | uniform |
| bird_minidev | 1.443 | 0.866 | matched | 0.269 | 0.363 | uniform |
| sqlflow | 1.258 | 0.754 | matched | 0.092 | 0.089 | matched |
| spider2local | 1.446 | 0.861 | matched | 0.489 | 0.354 | matched |

**The bound prefers matched on 5/5; the data prefers it on 2/5.** Switching estimator
on the strength of the bound is not justified by these measurements.

## E3 — the bound holds, and is nearly vacuous

`L_hat` is a kNN-smoothed 95th-percentile Lipschitz proxy over the calibration items
(0.359–0.364 across models). `delta = 0.05`.

| target | median B | median \|err\| | holds | n_eff needed for B≤1 | for B≤0.2 | n_eff achieved |
|---|---:|---:|---:|---:|---:|---:|
| spider | 0.765 | 0.373 | 100% | 6.6 | ∞ | 21.59 |
| bird | 0.850 | 0.210 | 100% | 6.8 | ∞ | 13.47 |
| bird_minidev | 0.866 | 0.377 | 100% | 6.9 | ∞ | 12.47 |
| sqlflow | 0.754 | 0.080 | 100% | 6.5 | ∞ | 21.43 |
| spider2local | 0.861 | 0.333 | 100% | 6.8 | ∞ | 11.08 |

**50/50 model×target pairs satisfy the bound.** They satisfy it the way "accuracy ≤ 1"
satisfies it. The Lipschitz constant that would make the bound *bind* is a median of
0.000–0.062 against `L_hat = 0.361` — the bound has an order of magnitude of slack.

Which term is to blame is unambiguous:

| target | transfer `2L√(1−c)` | Hoeffding | B |
|---|---:|---:|---:|
| spider | 0.473 | 0.292 | 0.765 |
| bird | 0.480 | 0.370 | 0.850 |
| bird_minidev | 0.481 | 0.385 | 0.866 |
| sqlflow | 0.461 | 0.293 | 0.754 |
| spider2local | 0.453 | 0.408 | 0.861 |

The **sample-size condition of §19.8 is comfortably met** (need ~6.6, have 11–22). The
**coverage condition is what fails**: at `L = 0.36`, reaching `B ≤ 0.2` requires
`c > 0.923`, and 0.57 is achieved. No amount of extra labeling fixes this; only a
calibration corpus that is actually near the target does.

## E4 — is similarity the right power-prior discount?

MTM08 makes `a0 = clip(f_cov/N, 0, 1)` algebraically legitimate as a power-prior
exponent. It does not make it *correct*. Sweeping it:

| target | `a0 = c` | best `a0` (PoolEval) | best `a0` (collision EM) | PE at `c` | PE best | CE at `c` | CE best |
|---|---:|---:|---:|---:|---:|---:|---:|
| spider | 0.570 | 1.000 | 0.500 | 9.08 | 5.85 | 1.98 | 1.88 |
| bird | 0.557 | 1.000 | 1.000 | 26.20 | 25.42 | 29.11 | 27.90 |
| bird_minidev | 0.555 | 0.000 | 0.100 | 30.51 | 27.53 | 37.66 | 36.98 |
| sqlflow | 0.592 | 1.000 | 1.000 | 14.05 | 12.61 | 16.53 | 14.74 |
| spider2local | 0.606 | 0.000 | 0.000 | 29.18 | 18.75 | 34.24 | 23.80 |

Two problems.

1. **`a0 = c` barely varies.** Across five targets whose true pool accuracy spans
   0.05–0.74, the coverage discount moves only over `[0.555, 0.606]`. It is effectively
   the constant 0.57 and carries almost no target-specific information.
2. **The optimum is at a boundary.** `a0 = 1` on 3/5 targets and `a0 = 0` on 2/5 for
   PoolEval. The middle is never right; `a0 = c` sits in the middle by construction and
   costs 0.8–10.4 MAE points.

This is exactly the qualification §4.2 states: *"interpreting a number as a valid
exponent does not prove it is the correct discount."* Measured, it is not.

## E5 — the factor `J` in `s_beta`

`s_beta = a0·J·n0` expresses the pool-level anchor in model-item pair units. Holding
the prior location fixed:

| target | `s_beta = a0·n0` (item) | `= a0·J·n0` (pair) | `= 0` | winner |
|---|---:|---:|---:|---|
| spider | 1.98 | **1.81** | 2.03 | pair |
| bird | 29.11 | 33.11 | **25.64** | zero |
| bird_minidev | 37.66 | 40.40 | **35.57** | zero |
| sqlflow | 16.53 | 20.07 | **13.85** | zero |
| spider2local | 34.24 | **33.98** | 35.05 | pair |

**Item-scaling never wins, pair-scaling wins twice by <0.3 points, and dropping the
pool-level anchor entirely wins three times by 1.5–7.5 points.** Pair-scaling costs
+7.5 MAE on bird. §5 argues `J` is nominal because one pseudo-label correctness event
per item does not become `J` independent observations when repeated across models; the
data agrees.

## E6 — the bound is not monotone in the dataset budget

Section 14 warns that raising coverage can concentrate the matching weights and worsen
the sampling term. Sweeping `K = 1..10`:

| target | `c` monotone ↑ | `n_eff` monotone ↑ | `B` monotone ↓ | `n_eff` range |
|---|---|---|---|---|
| spider | yes | yes | yes | 9.0 – 22.3 |
| bird | yes | **no** | **no** | 8.5 – 14.6 |
| bird_minidev | yes | **no** | **no** | 8.1 – 12.9 |
| sqlflow | yes | **no** | yes | 7.5 – 31.5 |
| spider2local | yes | **no** | **no** | 7.6 – 15.2 |

Coverage is monotone on 5/5, as proved. `n_eff` is non-monotone on 4/5 and the bound
is non-monotone on 3/5. **A bigger labeling budget can make the certificate worse.**

## E7 — where `alpha_K` actually sits

Greedy coverage vs the brute-force optimum over all `C(32, K)` collections, and vs the
distance-ranked top-k that Stage 1 currently uses:

| target | K | `alpha_K` | greedy/opt | distance-topk/opt | prior MAE greedy | distance |
|---|---:|---:|---:|---:|---:|---:|
| spider | 2 | 0.750 | 1.0000 | 0.9851 | 44.27 | 50.67 |
| spider | 3 | 0.704 | 1.0000 | 0.9930 | 46.13 | 39.53 |
| bird | 2 | 0.750 | 0.9998 | 0.9701 | 9.27 | 23.60 |
| bird | 3 | 0.704 | 0.9999 | 0.9732 | 7.93 | 18.40 |
| bird_minidev | 2 | 0.750 | 1.0000 | 0.9472 | 14.40 | 28.93 |
| bird_minidev | 3 | 0.704 | 1.0000 | 0.9292 | 37.80 | 23.53 |
| sqlflow | 2 | 0.750 | 1.0000 | 0.9638 | 13.40 | 8.00 |
| sqlflow | 3 | 0.704 | 0.9986 | 0.9597 | 10.00 | 5.27 |
| spider2local | 2 | 0.750 | 0.9983 | 0.9670 | 62.08 | 21.25 |
| spider2local | 3 | 0.704 | 1.0000 | 0.9583 | 39.17 | 18.33 |

Greedy lands at **0.9983–1.0000 of optimal** against a guarantee of 0.704–0.750. The
`1 - 1/e` factor is real but enormously conservative; quoting it understates greedy by
30 percentage points. Distance-ranked top-k reaches 0.929–0.993 of optimal coverage —
Stage 1 is an accidental coverage maximizer.

But **more coverage is not less error**: greedy has strictly higher coverage than
distance-topk in all 10 cells and lower prior MAE in only 5 of them, with swings up to
41 points. Coverage maximization and error minimization are different objectives, as
§14 says.

## E8 — the assumption the whole transfer term rests on

Section 8.1 assumes `|p_j(x) - p_j(z)| <= L_j·||phi(x) - phi(z)||`. Slicing all
target×calibration pairs into distance deciles and comparing each side's mean
correctness:

| target | distance range observed | spearman(d, gap) | gap in the *nearest* decile |
|---|---|---:|---:|
| spider | 1.366 – 1.413 | −0.018 | 0.164 |
| bird | 1.367 – 1.412 | +0.636 | 0.142 |
| bird_minidev | 1.359 – 1.411 | +0.309 | 0.239 |
| sqlflow | 1.332 – 1.412 | +0.152 | 0.103 |
| spider2local | 1.333 – 1.410 | +0.915 | 0.479 |

The maximum possible distance between unit vectors is `sqrt(2) = 1.4142`. **The entire
observed range spans about 3% of it** — every SynSQL probe is nearly orthogonal to
every target item. Fitted slopes across the deciles range from −0.54 to +2.14 with R²
from 0.04 to 0.80: the Lipschitz constant is **not identifiable from this data**, and
extrapolating the fit back to `d = 0` is a 30× extrapolation with no support.

The consequential number is the last column. Even the *nearest* decile of pairs already
shows a 0.10–0.48 accuracy gap. Section 8.1 anticipates this: *"Domain differences that
invalidate this implication require an additional discrepancy term."* That term is
missing from the bound, and on this corpus it is the dominant one.

## E9 — the in-distribution control

E1–E8 all run at `c ≈ 0.57`, far below anything the bound needs, so a valid-but-vacuous
bound proves nothing about the theory. Control: split each target benchmark in half and
calibrate on its own other half. Identical machinery; only the preconditions change.

| target | c | n_eff | prior MAE uniform | matched | median B | median \|err\| |
|---|---:|---:|---:|---:|---:|---:|
| spider | 0.9653 | 20.60 | **3.47** | 11.47 | 0.431 | 0.113 |
| bird | 0.9725 | 13.42 | **7.33** | 19.47 | 0.505 | 0.153 |
| bird_minidev | 0.9752 | 18.81 | 5.87 | **4.80** | 0.425 | 0.040 |
| sqlflow | 0.8127 | 27.71 | 10.13 | **5.47** | 0.489 | 0.047 |
| spider2local | 0.7340 | 4.80 | **5.00** | 15.83 | 0.620 | 0.125 |
| **mean delta** | | | | **+5.05** | | |

Three things follow.

1. **Coverage really is the binding term.** It rises 0.57 → 0.73–0.98, the error falls
   0.08–0.38 → 0.040–0.153, and the bound tightens 0.75–0.87 → 0.43–0.62. The theory's
   causal story is right.
2. **The two terms swap dominance.** Out of distribution: transfer 0.45–0.48 vs
   Hoeffding 0.29–0.41. In distribution: transfer 0.00–0.23 vs Hoeffding 0.26–0.62.
   In the regime the bound is designed for, the *sampling* term is what limits it, and
   the only way to shrink that is more labels or less weight concentration.
3. **Matching still loses, by more (+5.05 pts).** So its failure in E2 is not an
   artifact of the domain gap. It is intrinsic: matching concentrates 150 target items
   onto 12–31 distinct labeled ones and pays for it in `n_eff`.

---

## What this means for the paper

**Keep, with a correction.**

* State `ESS = s + 2`, not `s`. It is a one-word fix and a reviewer will find it.
* Report `s`, `a0`, `ESS = s + 2`, and the MAP weight `r = s/(N+s)` per model. At the
  current settings `r` runs 0.32 at the coverage discount — the anchor supplies about
  a third of the posterior accuracy, which is a fact worth stating rather than burying.
* The greedy `1 - 1/e` guarantee is correct and worth citing, but measured greedy is at
  0.998–1.000 of optimal. Present `alpha_K` as a floor, not as a description.

**Do not adopt, on this evidence.**

* **The Section 16 switch to target-matching weights.** The bound prefers it 5/5, the
  data prefers it 2/5, and the mechanism is understood: matching buys no coverage and
  costs 2.4–3.4× in the sampling term. Revisit only if coverage rises far enough that
  the transfer term dominates — and E9 shows that even then it loses.
* **`a0 = f_cov/N` as the discount.** Algebraically a valid power-prior exponent,
  empirically near-constant at 0.57 across five targets and never optimal. Either
  calibrate `a0` on held-out targets or drop the coverage link and treat it as a
  hyper-parameter honestly.
* **The factor `J` in `s_beta`.** Never wins; `s_beta = 0` wins on 3/5.

**Report as a limitation.**

The bound holds on 50/50 model×target pairs and is loose by roughly an order of
magnitude, because `c = 0.57` puts the transfer term at 0.45–0.48 on its own. Reaching
a useful `B <= 0.2` needs `c > 0.923` at the measured `L`. The honest claim is that the
bound *certifies nothing yet* on SynSQL-retrieved calibration data, and that the
smoothness assumption it rests on cannot even be tested there, because every probe item
is nearly orthogonal to every target item (all pairwise distances within 3% of the
`sqrt(2)` maximum). That is a statement about the retrieval corpus, not about the
theorem.
