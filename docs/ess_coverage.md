# Does the new theory reduce the error?

Two derivations were added to the paper:

* **`coverage_hoeffding_derivation.md`** — dataset coverage is monotone submodular,
  greedy gets `α_K ≥ 1 − 1/e` of optimal coverage, and coverage plus a weighted
  Hoeffding term bound the error of a **target-matched** calibration estimator.
* **`MTM08_analysis.md`** — a Beta prior's curvature-matched effective sample size is
  the sum of its parameters, so PoolEval's anchor carries `ESS = s + 2` with
  `s = a₀·n₀` a power-prior discount.

They are now code: [`pooleval/theory.py`](../pooleval/theory.py) (primitives),
[`tests/test_theory.py`](../tests/test_theory.py) (14 tests, verifies every
algebraic claim), [`run_theory_ablation.py`](../experiments/run_theory_ablation.py)
(head-to-head), [`run_ess_coverage.py`](../experiments/run_ess_coverage.py)
(diagnostics).

Everything below is measured on **real** data: 800 labeled SynSQL probe items, 10 real
models, real SQL execution, five real Text2SQL targets. MAE is in accuracy points.

---

# 1. The bottom line

| | prior MAE | **PoolEval MAE** | collision-EM MAE |
|---|---:|---:|---:|
| Baseline — no new theory | 24.76 | 22.61 | 25.60 |
| **Best new-theory configuration** | **22.57** | **18.86** | **21.18** |
| **improvement** | **−2.19** | **−3.75** | **−4.42** |

The best configuration is **greedy coverage selection + target-matched weights +
item-scaled `s_β`**, at *full* prior strength. It wins on **4 of 5 targets**.

**The one piece of the new theory that does not work is the coverage-derived discount
`a₀ = f_cov/N`.** Adding it costs 1.24 MAE points. Everything else helps.

---

# 2. Head-to-head, one switch at a time

Four independent switches, each defined by the derivations:

| switch | baseline (today) | new theory |
|---|---|---|
| selection | rank subsets by centroid distance, take top-k | greedy coverage (submodular) |
| weights | uniform average over selected items | target-matched `w_z` |
| prior strength | `s = n₀` (i.e. `a₀ = 1`) | `s = c·n₀` (coverage discount) |
| `s_β` | `a₀·J·n₀` (pair-scaled) | `a₀·n₀` (drop the nominal `J`) |

## Adding one switch at a time

PoolEval MAE, lower is better. Best per column in bold.

| configuration | spider | bird | bird_minidev | sqlflow | spider2local | **MEAN** |
|---|---:|---:|---:|---:|---:|---:|
| Baseline — no new theory | 8.64 | 23.04 | **27.28** | 14.24 | 39.85 | 22.61 |
| + greedy coverage selection | 8.76 | 24.46 | 27.64 | 15.26 | 39.36 | 23.10 |
| + target-matched weights | 5.85 | 25.42 | 31.96 | **12.61** | 30.84 | 21.34 |
| + coverage discount `a₀ = c` | 11.60 | 24.68 | 27.37 | 15.28 | 36.70 | 23.13 |
| + item-scaled `s_β` (drop `J`) | 8.64 | 23.04 | **27.28** | 14.24 | 39.85 | 22.61 |
| All new theory | 9.67 | 19.68 | 30.04 | 14.25 | **26.86** | 20.10 |

Individually, only **matched weights** helps (−1.27). Greedy alone and the discount
alone both *hurt*. But all four together give −2.51, more than any single switch —
so there is an interaction. The reverse ablation finds it.

## Removing one switch from the full set

| configuration | spider | bird | bird_minidev | sqlflow | spider2local | **MEAN** |
|---|---:|---:|---:|---:|---:|---:|
| All new theory | 9.67 | 19.68 | 30.04 | 14.25 | **26.86** | 20.10 |
| − greedy selection | 9.08 | 26.20 | 30.51 | 14.05 | 29.18 | 21.80 |
| − matched weights | 11.60 | 25.67 | 27.62 | 15.97 | 36.34 | 23.44 |
| **− coverage discount** | **5.76** | **16.55** | 30.92 | 12.95 | 28.13 | **18.86** |
| − item-scaled `s_β` | 9.67 | 19.68 | 30.04 | 14.25 | **26.86** | 20.10 |

**Removing the coverage discount *improves* the result by 1.24 points.** Removing
greedy or matched makes it worse. So the winning package is greedy + matched, and the
discount is dead weight.

**Greedy and matching only work together.** Greedy alone is +0.49 (worse); matched
alone is −1.27; both together are −3.75. That makes sense — greedy changes *which*
items you hold but you still average them uniformly, and matching reweights items that
were chosen by a criterion other than coverage. Only together do they form the
estimator the theorem actually describes.

---

# 3. Verdict per switch

| switch | effect on PoolEval MAE | verdict |
|---|---:|---|
| Greedy coverage selection | −1.70 (in the full set) | **adopt** — but only with matched weights |
| Target-matched weights | −3.34 (in the full set) | **adopt** — the single biggest win |
| Coverage discount `a₀ = c` | **+1.24** | **reject** — see §5.4 |
| Item-scaled `s_β` (drop `J`) | 0.00 on PoolEval, **−1.90** on collision EM | **adopt** — free, and helps where it applies |

`s_β` only enters the collision EM, which is why its PoolEval column is exactly zero.

---

# 4. Is the win robust?

The mean over five targets can be carried by one of them, so: how often does each
configuration actually win, and what happens if the smallest and most extreme target
(spider2local — 24 items, true accuracy 0.05) is dropped?

| configuration | mean MAE | vs baseline | targets won | mean excl. spider2local |
|---|---:|---:|:---:|---:|
| Baseline — no new theory | 22.61 | — | 0/5 | 18.30 |
| + greedy coverage selection | 23.10 | +0.49 | 1/5 | 19.03 |
| + target-matched weights | 21.34 | −1.27 | 3/5 | 18.96 |
| + coverage discount `a₀ = c` | 23.13 | +0.52 | 1/5 | 19.73 |
| + item-scaled `s_β` | 22.61 | +0.00 | 0/5 | 18.30 |
| All new theory | 20.10 | −2.51 | 2/5 | 18.41 |
| **All new theory − discount** | **18.86** | **−3.75** | **4/5** | **16.55** |

The winner survives both checks: **4/5 targets** and still best (**16.55** vs 18.30)
with spider2local removed. "All new theory" *with* the discount does not — it wins
only 2/5 and is a wash (18.41 vs 18.30) once spider2local is dropped. Its apparent
advantage is one target.

---

# 5. Why — the diagnostics

## 5.1 Why the bound holds and still tells you nothing

This section assumes no prior familiarity with the bound. Read it before the tables.

### The bound, in one line

For model $j$, the derivation ends at a single inequality that is true with
probability at least $1-\delta$:

$$
\bigl|\underbrace{\hat\theta_j}_{\text{our estimate}}-\underbrace{\theta_j}_{\text{the truth}}\bigr|
\;\le\;
B_j,
\qquad
B_j=
\underbrace{2L_j\sqrt{1-c}}_{\textbf{transfer term}}
\;+\;
\underbrace{\sqrt{\frac{\log(2/\delta)}{2\,n_{\mathrm{eff}}}}}_{\textbf{sampling term}} .
$$

Each half prices one distinct worry:

| term | the worry it prices | shrinks when |
|---|---|---|
| $2L_j\sqrt{1-c}$ | *my labeled items are not the items I want to score* | coverage $c\to1$ |
| $\sqrt{\log(2/\delta)/(2n_{\mathrm{eff}})}$ | *even the right items are a finite random sample* | $n_{\mathrm{eff}}$ grows |

$c\in[0,1]$ is normalized coverage, $n_{\mathrm{eff}}$ the concentration effective
sample size, $L_j$ the smoothness constant, $\delta=0.05$.

### Two different questions

These are not the same question, and the answers are opposite:

1. **Is the bound true?** Does the real error ever exceed $B_j$? — **No, never. 50/50.**
2. **Is the bound useful?** Does knowing $B_j$ tell you anything you did not
   already know? — **No.**

A forecast of *"tomorrow's temperature will be between $-50^\circ$C and
$+50^\circ$C"* is never wrong and never useful. That is the situation here, and the
word for it is **vacuous**, not merely loose.

### Worked example: spider, every number

Measured inputs: $L_j=0.361$, $c=0.5699$, $n_{\mathrm{eff}}=21.59$, $\delta=0.05$.

**Transfer term.**

$$
2L_j\sqrt{1-c}
=2(0.361)\sqrt{1-0.5699}
=0.722\times\underbrace{\sqrt{0.4301}}_{=\,0.6558}
=\mathbf{0.473}
$$

**Sampling term.** With $\log(2/0.05)=\log 40=3.689$,

$$
\sqrt{\frac{\log(2/\delta)}{2\,n_{\mathrm{eff}}}}
=\sqrt{\frac{3.689}{2\times21.59}}
=\sqrt{0.08543}
=\mathbf{0.292}
$$

**Total.**

$$
B_j=0.473+0.292=\mathbf{0.765}
$$

### What $B_j = 0.765$ actually buys you

A bound on the error is a confidence interval on the truth:

$$
\hat\theta_j-B_j\;\le\;\theta_j\;\le\;\hat\theta_j+B_j .
$$

Substituting the real estimates for spider's first three models:

| model | estimate $\hat\theta_j$ | $B_j$ | interval | after clipping to $[0,1]$ | true accuracy |
|---:|---:|---:|---|---|---:|
| 0 | 0.427 | 0.765 | $[-0.338,\,+1.192]$ | $[0.00,\,1.00]$ | 0.760 |
| 1 | 0.420 | 0.763 | $[-0.343,\,+1.183]$ | $[0.00,\,1.00]$ | 0.760 |
| 2 | 0.293 | 0.767 | $[-0.474,\,+1.061]$ | $[0.00,\,1.00]$ | 0.727 |

**Accuracy is already a number in $[0,1]$.** The interval is 1.53 wide, so after
clipping it covers **100% of $[0,1]$ for all ten models**. The theorem's output is
*"this model's accuracy is somewhere between 0% and 100%"* — which was true before
any of it was proved. That is what "the bound certifies nothing" means.

### Why coverage $0.57$ is not "57% of the way there"

Coverage uses the shifted similarity

$$
S(x,z)=\frac{1+\phi(x)^\top\phi(z)}{2},
\qquad
c=\frac1N\sum_{x}\max_z S(x,z).
$$

TF-IDF features are **non-negative**, so $\phi(x)^\top\phi(z)\ge0$ and therefore
$S\ge\tfrac12$ *always* — two completely unrelated items still score $0.5$. So $c$ is
not a percentage of anything; the usable range is $[0.5,1]$, and $c=0.57$ sits almost
at its floor:

| raw cosine $\phi^\top\phi$ | $c$ | $\sqrt{1-c}$ | transfer term |
|---:|---:|---:|---:|
| 0.00 (unrelated) | 0.500 | 0.707 | 0.510 |
| **0.14 (what we have)** | **0.570** | **0.656** | **0.473** |
| 0.50 | 0.750 | 0.500 | 0.361 |
| 0.85 | 0.925 | 0.274 | 0.198 |
| 1.00 (identical) | 1.000 | 0.000 | 0.000 |

Going from *totally unrelated* ($0.510$) to *what we actually achieved* ($0.473$)
buys **0.037**. The square root is the reason: it is flat near $c=0.5$ and only bites
near $c=1$.

### How much coverage would be enough?

Demand $B_j\le\varepsilon$. The transfer term alone must fit inside $\varepsilon$:

$$
2L_j\sqrt{1-c}<\varepsilon
\;\Longleftrightarrow\;
\sqrt{1-c}<\frac{\varepsilon}{2L_j}
\;\Longleftrightarrow\;
\boxed{\;c>1-\frac{\varepsilon^{2}}{4L_j^{2}}\;}
$$

At the measured $L_j=0.361$:

| target $\varepsilon$ | required $c$ | have $c=0.570$? |
|---:|---:|:---:|
| 0.5 | 0.519 | yes |
| 0.3 | 0.827 | **no** |
| **0.2** | **0.923** | **no** |
| 0.1 | 0.981 | **no** |

This is the origin of the number 0.92. And it is a *necessary* condition only: at
exactly $c=0.923$ the transfer term consumes the entire budget, leaving nothing for
sampling, so $n_{\mathrm{eff}}$ would have to be infinite. Real usefulness needs
coverage comfortably above it.

### The diagnosis, and the proof that it is the corpus and not the theorem

The sample-size half is **fine** — §19.8 needs $n_{\mathrm{eff}}\approx6.6$ and we
have $11$–$22$. The coverage half is what fails, by a wide margin. No amount of extra
labeling repairs this; only a calibration corpus that is genuinely near the target
does.

§5.9 is the control that proves the point. Calibrating each target on its **own other
half** raises $c$ to $0.73$–$0.98$, and every quantity moves exactly as the formula
says it must:

| | SynSQL corpus | target's own other half |
|---|---:|---:|
| coverage $c$ | 0.570 | **0.965** |
| transfer term | 0.473 | **0.132** |
| sampling term | 0.292 | 0.299 |
| bound $B$ | 0.765 | **0.431** |
| real error | 0.373 | **0.113** |
| share of $[0,1]$ the interval covers | 100% | **86%** |

The transfer term collapses by $3.6\times$ while the sampling term barely moves, and
the two halves swap which one dominates. **The theorem is behaving correctly. SynSQL
is simply too far from these targets for it to say anything.**

## 5.2 The same picture across all five targets

§5.1 worked spider in full. The other four behave identically. $\delta=0.05$;
$\hat L$ is a kNN-smoothed Lipschitz proxy (0.359–0.364 across models).

| target | bound $B$ | real error | holds? | ratio | interval width $2B$ | share of $[0,1]$ covered |
|---|---:|---:|:---:|---:|---:|---:|
| spider | 0.765 | 0.373 | yes | 2.0× | 1.53 | 100% |
| bird | 0.850 | 0.210 | yes | 4.0× | 1.70 | 100% |
| bird_minidev | 0.866 | 0.377 | yes | 2.3× | 1.73 | 100% |
| sqlflow | 0.754 | **0.080** | yes | 9.4× | 1.51 | 100% |
| spider2local | 0.861 | 0.333 | yes | 2.6× | 1.72 | 100% |

**50 of 50 model×target pairs satisfy the bound**, and on every one of them the
resulting interval is wider than $[0,1]$ itself. The bound is 2.0–9.4× the real error
(median 2.6×), and the Lipschitz constant that would make it *bind* is 0.000–0.062
against $\hat L=0.361$ — 6× too small or worse.

## 5.3 Coverage is the binding constraint, not sample size

| target | coverage `c` | raw cosine | `n_eff` | transfer term | Hoeffding term |
|---|---:|---:|---:|---:|---:|
| spider | 0.570 | 0.140 | 21.59 | 0.473 | 0.292 |
| bird | 0.557 | 0.113 | 13.47 | 0.480 | 0.370 |
| bird_minidev | 0.555 | 0.110 | 12.47 | 0.481 | 0.385 |
| sqlflow | **0.592** | **0.185** | 21.43 | **0.461** | 0.293 |
| spider2local | 0.606 | 0.213 | 11.08 | 0.453 | 0.408 |

The sample-size condition is comfortably met (needs ≈6.6, has 11–22). The **coverage
condition fails**: `B ≤ 0.2` requires `c > 0.923` and we have 0.57. Because TF-IDF
features are non-negative, `S = (1+cos)/2` floors at 0.5, so `c = 0.57` means a raw
best-match cosine of **0.14** — the calibration corpus is nearly orthogonal to the
targets. More labels cannot fix this; only a nearer corpus can.

## 5.4 The coverage discount carries almost no information

| target | `a₀ = c` | best `a₀` | MAE at `a₀ = c` | MAE at best | cost |
|---|---:|---:|---:|---:|---:|
| spider | 0.570 | 1.000 | 9.08 | **5.85** | +3.23 |
| bird | 0.557 | 1.000 | 26.20 | **25.42** | +0.78 |
| bird_minidev | 0.555 | 0.000 | 30.51 | **27.53** | +2.98 |
| sqlflow | 0.592 | 1.000 | 14.05 | **12.61** | +1.44 |
| spider2local | 0.606 | 0.000 | 29.18 | **18.75** | +10.43 |

Across five targets whose true pool accuracy spans 0.05–0.74, `a₀ = c` moves only over
**[0.555, 0.606]** — effectively the constant 0.57. And the optimum is always at a
*boundary* (0 or 1), never in the middle where `a₀ = c` sits by construction.

MTM08 makes `a₀ = f_cov/N` a legitimate power-prior *exponent*. It does not make it the
right one. This is exactly the qualification in §4.2 of the note, now measured.

## 5.5 The factor `J` is not supported

Collision-EM MAE, prior location held fixed:

| target | `s_β = a₀·n₀` (item) | `s_β = a₀·J·n₀` (pair) | `s_β = 0` |
|---|---:|---:|---:|
| spider | 1.98 | **1.81** | 2.03 |
| bird | 29.11 | 33.11 | **25.64** |
| bird_minidev | 37.66 | 40.40 | **35.57** |
| sqlflow | 16.53 | 20.07 | **13.85** |
| spider2local | 34.24 | **33.98** | 35.05 |

Pair-scaling wins twice by <0.3 points and loses badly on bird (+7.5). Dropping the
pool-level anchor entirely wins 3/5. One pseudo-label event per item does not become
`J` independent observations when repeated across models — §5 of the note is right.

## 5.6 A bigger budget can make the certificate worse

Sweeping the dataset budget `K = 1…10`:

| target | `c` rises monotonically? | `n_eff` rises monotonically? | bound falls monotonically? |
|---|:---:|:---:|:---:|
| spider | yes | yes | yes |
| bird | yes | **no** | **no** |
| bird_minidev | yes | **no** | **no** |
| sqlflow | yes | **no** | yes |
| spider2local | yes | **no** | **no** |

Coverage is monotone on 5/5, as proved. `n_eff` is not on 4/5, and the bound is not on
3/5 — §14's warning, measured.

## 5.7 `α_K` is very conservative

| K | guarantee `α_K` | greedy actually achieves | distance top-k achieves |
|---:|---:|---:|---:|
| 2 | 0.750 | **0.9983 – 1.0000** | 0.947 – 0.985 |
| 3 | 0.704 | **0.9986 – 1.0000** | 0.929 – 0.993 |

Greedy is at 99.8–100% of the brute-force optimum against a 70–75% guarantee. Quote
`1 − 1/e` as a floor, not as a description.

## 5.8 The smoothness assumption cannot be tested on this corpus

Every target×calibration pair, sliced into distance deciles:

| target | distance range observed | max possible | correlation(d, gap) | gap in *nearest* decile |
|---|---|---:|---:|---:|
| spider | 1.366 – 1.413 | 1.4142 | −0.018 | 0.164 |
| bird | 1.367 – 1.412 | 1.4142 | +0.636 | 0.142 |
| bird_minidev | 1.359 – 1.411 | 1.4142 | +0.309 | 0.239 |
| sqlflow | 1.332 – 1.412 | 1.4142 | +0.152 | **0.103** |
| spider2local | 1.333 – 1.410 | 1.4142 | +0.915 | 0.479 |

**The entire observed range spans about 3% of the maximum.** Every SynSQL probe is
nearly orthogonal to every target item, so the Lipschitz slope is unidentifiable
(fitted slopes run −0.54 to +2.14, R² 0.04–0.80). And even the *nearest* decile already
shows a 0.10–0.48 accuracy gap — the extra discrepancy term §8.1 warns about, and which
the bound does not have.

## 5.9 In-distribution control

Everything above runs at `c ≈ 0.57`, far below what the bound needs, so a
valid-but-vacuous bound proves nothing either way. Control: calibrate each target on
its **own other half**. Same machinery, but now the preconditions can hold.

| target | `c` | `n_eff` | real error | bound `B` |
|---|---:|---:|---:|---:|
| spider | 0.965 | 20.60 | 0.113 | 0.431 |
| bird | 0.973 | 13.42 | 0.153 | 0.505 |
| bird_minidev | **0.975** | 18.81 | **0.040** | **0.425** |
| sqlflow | 0.813 | **27.71** | 0.047 | 0.489 |
| spider2local | 0.734 | 4.80 | 0.125 | 0.620 |

Coverage rises 0.57 → 0.73–0.98, the real error collapses 0.08–0.38 → 0.04–0.15, and
the bound tightens 0.75–0.87 → 0.43–0.62. **The theory's causal story is right.** And
the two terms swap roles: out of distribution the transfer term dominates (0.45–0.48 vs
0.29–0.41); in distribution the Hoeffding term does (0.00–0.23 vs 0.26–0.62).

---

## 5.10 Controlled budget sweep across B = 1

**Reference correction after code audit:** the crossing below concerns the two-term bound for expected target accuracy, whereas reported MAE is measured against observed finite-benchmark accuracy (`zoo/build.py` computes the mean of binary execution outcomes). It is not a crossing of a bound aligned with that MAE reference. The script now also records `realized_bounds` using the existing `pooleval.theory.bound_realized()` function, which adds target-outcome concentration and divides the failure probability between calibration and target outcomes. Under the stochastic assumptions, this is the formula corresponding to the observed benchmark reference. On bird_minidev its maximum across models is **1.2763, 1.1717, 1.0945, 1.0338, and 1.0195** at K = 1, 2, 3, 5, and 10. No tested budget makes all of these bounds at most one. Thus the earlier crossing does not establish the requested threshold comparison for observed-accuracy MAE.

The script additionally records simultaneous realized-accuracy bounds over all ten models and ten budgets within each target, allocating delta / 100 per comparison. These are larger still. Taking the maximum of ten pointwise bounds does not itself provide 95% simultaneous coverage. All versions remain diagnostics because the smoothness proxy and conditional outcome independence are unverified. With fixed cached SQL predictions there are no repeated independent executions in this experiment, so counting observed errors below bounds cannot validate a failure-probability claim. The MAE values below are unchanged.

To test the two-term theorem itself, a separate controlled-outcome experiment is needed. Keep embeddings and input-only selection fixed, specify known correctness probabilities with a provable Lipschitz constant, repeatedly draw independent calibration outcomes, and compare the matched estimate with the known expected target accuracy. For example, on unit embeddings use p(v) = 1/2 + a u^T phi(v), with a fixed unit vector u and 0 <= a <= 1/2. Then p is in [0,1] and its Lipschitz constant is at most a by Cauchy-Schwarz. Generate independent Bernoulli outcomes from these probabilities. Report MAE against the known expected accuracy and the frequency of bound violations over independent repetitions, with Monte Carlo uncertainty. This would test the implementation under the theorem's assumptions, not certify those assumptions for real Text2SQL. It is run in §5.11 below, and choosing the effective sample size by the Section 19 threshold is tested in §5.12. Real-data MAE and downstream EM remain separate empirical evaluations (§5.13).

This additional experiment varies only the dataset budget K from 1 to 10 along a fixed coverage-greedy ordering. It keeps the calibration corpus, target items, matched-weight estimator, embedding geometry, PoolEvaluator configuration, and delta = 0.05 unchanged. Crucially, downstream prior strength is fixed at s = 125, the K = 5 calibration count, at every budget. Sigma is recomputed from each prior location to preserve this same strength. Thus increasing K does not also strengthen the EM anchor. Target labels are used only for evaluation, never to choose the ordering, budget, or threshold comparison. The matched weights necessarily change when the selected data changes.

Reproduce with `.venv/bin/python experiments/run_bound_threshold.py`. Full per-model bounds, absolute errors, geometry, and all 50 target-budget configurations are saved in `results/bound_threshold.json`. These are new cached-data evaluations, not new model calls. The K = 5 PoolEvaluator MAEs reproduce the previous greedy/matched/full-strength ablation on every target.

The experiment freezes the earlier per-model Lipschitz proxies, estimated from calibration data, without retuning them to obtain a crossing. Their range is 0.3591-0.3640. Consequently, B is a diagnostic calculation under assumed smoothness, not a certified confidence bound. Define B_max as the maximum calculated bound across the ten models. B_max <= 1 means every model passes the numerical threshold. The ESS requirement below uses the largest frozen proxy so it applies to all ten models.

### Actual crossing on bird_minidev

| K | labeled calibration items | coverage c | matched ESS | required ESS for all models | B range across models | prior MAE | PoolEvaluator MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | 0.5500 | 4.9234 | 7.0459 | 1.0938-1.1004 | 9.13 | 18.97 |
| 2 | 50 | 0.5686 | 6.6805 | 6.7736 | 0.9972-1.0036 | 14.40 | 21.46 |
| 3 | 75 | 0.5824 | 8.6538 | 6.5772 | 0.9258-0.9321 | 37.80 | 33.03 |
| 5 | 125 | 0.5934 | 10.8801 | 6.4247 | 0.8697-0.8759 | 34.53 | 30.92 |
| 10 | 250 | 0.6054 | 11.2500 | 6.2622 | 0.8560-0.8622 | 24.40 | 26.07 |

MAE is in accuracy percentage points, lower is better. At K = 1, all ten bounds exceed one. At K = 2, seven are at most one and three exceed one. At K = 3, all ten are below one. The largest transfer term A is 0.4884, 0.4782, and 0.4704 at K = 1, 2, and 3, respectively. A < 1 throughout. The sampling term falls from 0.6121 to 0.5254 to 0.4617, and the ESS requirement becomes satisfied for all models at K = 3.

Despite this, the adjacent all-model threshold crossing from K = 2 to K = 3 increases prior MAE by **23.40 points** and PoolEvaluator MAE by **11.58 points**. The unambiguous all-above versus all-below comparison, K = 1 versus K = 3, also worsens both MAEs: **9.13 to 37.80** for the prior and **18.97 to 33.03** for PoolEvaluator.

For an exact per-model threshold comparison, take each model's adjacent budgets where its own B moves from >1 to <=1. Seven models cross at K = 1 to 2 and three at K = 2 to 3. Averaging these paired absolute-error changes gives **+7.87 points for the matched prior** and **+3.88 points for PoolEvaluator**. Seven models worsen and three improve for both estimators. These ten models share data, so this is a descriptive paired comparison, not ten independent replications.

### Other targets and scope

| target | B_max at K = 1 | B_max at K = 10 | prior MAE, K = 1 to 10 | PoolEvaluator MAE, K = 1 to 10 |
|---|---:|---:|---:|---:|
| spider | 0.9094 | 0.7584 | 53.87 to 26.27 | 6.83 to 7.43 |
| bird | 0.9573 | 0.8524 | 28.13 to 12.73 | 28.51 to 21.20 |
| sqlflow | 0.8289 | 0.7034 | 14.67 to 6.27 | 16.01 to 10.89 |
| spider2local | 0.9697 | 0.8018 | 65.42 to 35.00 | 55.51 to 29.94 |

All four other targets remain below one at every tested budget, so they cannot supply a threshold-crossing comparison in this sweep. They are not omitted because of their MAE outcomes. We did not change L or delta to manufacture additional crossings.

**Conclusion:** satisfying the calculated ESS condition and bringing B below one did not improve MAE in the observed crossing. An error upper bound is not the error itself, and reducing an upper bound does not require the realized error to decrease. Changing K also changes the selected items, coverage, and weights, so this controlled budget ablation tests the proposed association but does not identify an independent causal effect of the threshold. The bound pertains to the Stage 1 matched estimator under its assumptions, not to downstream EM accuracy. This finite-data diagnostic neither proves nor refutes the theorem, particularly because the global cross-domain smoothness constant is not certified. It does not support presenting B <= 1 as a sufficient condition for better MAE.

## 5.11 Controlled validation: the two-term theorem is right

§5.10 tested the bound on real outcomes and found it uninformative because coverage is low — a fact about the corpus, not the theorem. The controlled experiment §5.10 called for is now `run_bound_control.py`: real embedding geometry, but synthetic expected correctness `p_j(v) = 1/2 + a * cos(phi(v), phi(anchor_j))`, so `L = a` is a provable Lipschitz constant by Cauchy-Schwarz, with independent Bernoulli outcomes. Both assumptions of the theorem therefore hold by construction. 5000 repetitions, 10 synthetic models, delta = 0.05; the budget `K = 1..10` moves `n_eff` while the assumptions stay fixed.

| L = a | worst violation rate (all K, targets, models) | bound K=1 -> K=10 (n_eff 9.2 -> 17.6) | MAE pts K=1 -> K=10 | transfer floor 2L√(1-c) | bound ever <= 1? |
|---|---|---:|---:|---:|:---:|
| 0.00 | 0.0024 pool / 0.0072 pointwise | 0.600 -> 0.431 | 14.31 -> 10.02 | 0.000 | yes, all K |
| 0.25 | 0.0000 | 0.931 -> 0.745 | 14.36 -> 10.05 | 0.321 | yes, all K |
| 0.50 | 0.0000 | 1.263 -> 1.059 | 14.44 -> 10.09 | 0.641 | never |

Three facts. First, **the bound is never violated**: the worst pooled violation is 0.24% at `L = 0` (union bound across ten models) and exactly 0% at `L >= 0.25`, all far under delta = 5%. Second, **the deterministic transfer inequality `|theta_tilde - theta| <= 2L sqrt(1-c)` held on every configuration** (asserted in code), so the constructed `L` is genuinely valid. Third, **raising the effective sample size shrinks the sampling term and the error together**: `n_eff` 9.2 -> 17.6 tightens the bound and drops MAE about 4 points, with violations pinned at zero. The transfer term is an irreducible floor the sampling term sits on top of — at `L = 0.5` the bound never falls below one no matter how large `n_eff` grows.

## 5.12 Choosing ESS by the Section 19 threshold

Section 19.13 gives the exact rule: `B <= eps` iff `A < eps` and `n_eff >= n* = log(2/delta) / (2 (eps - A)^2)`. `run_ess_threshold.py` tests *choosing* the effective sample size by this rule, in a fully synthetic setting where coverage — hence the transfer term `A = 2L sqrt(1-c)` — is fixed and uniform, while `n_eff` is dialed directly as the number of equally weighted anchors, with a provable `L`. delta = 0.05, 4000 repetitions, 8 models.

| regime | L | c | A | eps | n* | crossing behavior | worst violation |
|---|---:|---:|---:|---:|---:|---|---:|
| general eps, feasible | 0.20 | 0.750 | 0.200 | 0.50 | 20.5 | `B <= eps` flips True exactly at n_eff = 21 | 0.0000 |
| **B in [0,1], A < 1** | 0.50 | 0.750 | 0.500 | 1.00 | 7.4 | `B <= 1` flips True exactly at n_eff = 8 = ceil(n*); n=7 -> B=1.013 | 0.0000 |
| **B in [0,1], A >= 1** | 0.90 | 0.600 | 1.138 | 1.00 | inf | A<1 fails; B > 1 at every n_eff up to 640 | 0.0000 |
| general eps, infeasible | 0.50 | 0.570 | 0.656 | 0.50 | inf | B floored at A; never <= eps up to n_eff = 320 | 0.0000 |

**The `B in [0,1]` condition specifically (eps = 1).** Section 19.8 says `B <= 1` iff `A < 1` and `n_eff >= log(2/delta) / (2 (1-A)^2)`. Both halves are confirmed. With `A = 0.5` the required `n* = log(40)/(2*0.25) = 7.38`, and the bound enters `[0,1]` at **exactly `n_eff = 8 = ceil(7.38)`** — n=7 gives B=1.013 (>1), n=8 gives B=0.980 (<=1) — one integer above the threshold, as the necessary-and-sufficient statement predicts. With `A = 1.138 >= 1` the necessary condition fails, `n*` is infinite, and `B` stays above one at every effective sample size up to 640 (the transfer term alone already exceeds one). So no amount of ESS brings `B` into `[0,1]` unless `A < 1` first.

The threshold is therefore **sharp** (the crossing lands on the exact integer above `n*`) and **valid** (zero violations in 4000 repetitions at every `n_eff`). The same pattern holds for a general tolerance `eps < 1`: at `eps = 0.5, A = 0.2` the crossing is exactly `n_eff = 21` above `n* = 20.5`; and when `A >= eps` the certificate is stuck at its transfer floor while the *estimator* MAE keeps falling (12.91 -> 2.23 points) — improving ESS keeps helping the realized error but cannot make the *certificate* reach `eps`.

**What we care about is MAE, and MAE tracks ESS, not the `[0,1]` threshold.** The bound is a loose certificate — the realized MAE runs 5-75x below `B`, because the estimator incurs only a small transfer bias where the bound charges the full worst-case transfer term. The MAE against known expected target accuracy, across the ESS sweep of the two `B in [0,1]` regimes:

When `A < 1`, the `n_eff` threshold for `B <= 1` comes from solving the bound directly:

$$
B = A + \sqrt{\frac{\log(2/\delta)}{2\,n_{\mathrm{eff}}}} \le 1
\;\Longrightarrow\;
\sqrt{\frac{\log(2/\delta)}{2\,n_{\mathrm{eff}}}} \le 1-A
\;\Longrightarrow\;
n_{\mathrm{eff}} \ge n^{*} = \frac{\log(2/\delta)}{2\,(1-A)^2}.
$$

The step that squares both sides is valid only because `A < 1` makes `1 - A` positive — that is exactly why `A < 1` is the necessary condition. If `A >= 1` the strictly positive sampling term sits on top of an already `>= 1` transfer term, so `B > 1` at every `n_eff` and `n* = inf`. For a stricter tolerance `eps < 1`, replace `1` by `eps` throughout: `A < eps`, then `n_eff >= log(2/delta) / (2 (eps - A)^2)`.

`n*` depends only on `A` and `delta`, so it is constant within a regime, and `n_eff >= ceil(n*)` is the integer requirement. It is solved per regime below and shown as its own column:

| regime | n_eff | n* for B<=1 | MAE (pts) | bound (pts) | bound / MAE | B <= 1 ? |
|---|---:|---:|---:|---:|---:|:---:|
| A = 0.500 (A < 1) | 5 | 7.38 (need 8) | 17.81 | 110.7 | 6.2 | no |
| | 7 | 7.38 (need 8) | 14.92 | 101.3 | 6.8 | no |
| | 8 | 7.38 (need 8) | 13.85 | 98.0 | 7.1 | **yes** |
| | 12 | 7.38 (need 8) | 11.59 | 89.2 | 7.7 | yes |
| A = 1.138 (A >= 1) | 10 | inf | 12.97 | 156.8 | 12.1 | no |
| | 40 | inf | 6.23 | 135.3 | 21.7 | no |
| | 160 | inf | 3.15 | 124.6 | 39.5 | no |
| | 640 | inf | 1.59 | 119.2 | 75.1 | no |

The `B <= 1 ?` column is exactly `n_eff >= ceil(n*)`: it flips to yes at `n_eff = 8` in the first regime and is never satisfiable in the second (`A >= 1` makes `n*` infinite). Reading the MAE column against it: crossing `B = 1` (n_eff 7 -> 8) is a non-event, a smooth 14.92 -> 13.85 step; and when `A >= 1` so `B` never enters `[0,1]`, raising ESS still drives MAE to 1.59 points — an essentially exact estimate under a permanently vacuous certificate. The MAE follows `MAE ~ 40 / sqrt(n_eff)` in every regime (a clean halving per 4x in n_eff), so choosing ESS by the Section 19 threshold just picks a point on that `1/sqrt(n_eff)` curve; the point where the loose certificate dips below one has no special status for the error. This is the controlled-setting version of §5.10's real-data finding that bringing `B` below one did not improve MAE.

Choosing ESS by the Section 19 threshold therefore does exactly what the theory predicts: it is the sampling-side lever, sharp and valid, and it can only be pulled after the transfer condition `A < eps` (for `B in [0,1]`, `A < 1`) already holds. On the real corpus that condition is the wall (§5.3): coverage 0.57 keeps `A` near 0.47, and the sampling side is already satisfied there (achieved `n_eff` 11-28 versus a requirement of about 6), so choosing ESS by the threshold buys nothing further on SynSQL — only a nearer corpus (higher coverage) would.

## 5.13 The certificate is not the estimator, and does not extend to EM

`report_mae_comparison.py` fixes the scope. The two-term theory is a certificate and diagnostics layer; it does not change the point estimator, so certificate-only MAE equals the pre-theory PoolEvaluator MAE exactly.

| method | Text2SQL | Image | Node |
|---|---:|---:|---:|
| before theory (PoolEvaluator) | 13.91 | 39.88 | 15.31 |
| after theory, certificate only | 13.91 | 39.88 | 15.31 |
| change | 0.00 | 0.00 | 0.00 |
| separate ESS-anchored estimator | 11.25 | — | — |

A separate estimator that *uses* the prior reaches 11.25 on Text2SQL, but the matched-weight theorem does not bound that EM output. Bounding the final EM estimate needs the separate stability argument in `stage1_accuracy_bound_and_em_gap_proof.md` (§17-§20), not this theorem.

# 6. What to change in the paper

**Adopt.**

* Greedy coverage selection **together with** target-matched weights — −3.75 MAE,
  wins 4/5 targets, robust to dropping the extreme target.
* Item-scaled `s_β`. Drop the factor `J`. Free, and −1.90 on the collision EM.
* Say `ESS = s + 2`, not `s`. The `+2` is the uniform `Beta(1,1)` baseline. It
  contributes curvature but zero exponents to the MAP objective, which is why the
  prior's mixing weight is `s/(N+s)` and not `s/(N+s+2)`.
* Report `s`, `a₀`, `ESS = s+2` and the MAP weight `r = s/(N+s)` per model. At current
  settings `r ≈ 0.32` — the anchor supplies about a third of the posterior accuracy.

**Reject.**

* `a₀ = f_cov/N` as the discount. Near-constant at 0.57 across five very different
  targets, never optimal, and costs +1.24 MAE. Keep `a₀ = 1` or calibrate `a₀` on
  held-out targets and say so.

**State as a limitation.**

The bound holds on 50/50 pairs and is 2.0–9.4× the real error, which makes it vacuous
rather than merely loose: it promises "within 0.75–0.87" about a quantity that already
lives in `[0, 1]`. The cause is entirely the transfer term, which `c = 0.57` puts at
0.45–0.48 before the sampling term is even added. Reaching a useful `B ≤ 0.2` needs
`c > 0.923`; `B ≤ 0.3` needs `c > 0.827`. The smoothness assumption it rests on cannot be tested on this
corpus at all, because every probe item is nearly orthogonal to every target item.
That is a statement about the SynSQL retrieval corpus, not about the theorem — §5.9
shows the bound behaves exactly as the theory predicts once coverage is real.

---

## Reproduce

```bash
python -m pytest tests/test_theory.py -q       # 14 passed
python experiments/run_theory_ablation.py      # sections 1-4
python experiments/run_ess_coverage.py         # section 5
python experiments/run_bound_threshold.py      # section 5.10 (real-data B=1 crossing)
python experiments/run_bound_control.py        # section 5.11 (controlled, provable L)
python experiments/run_ess_threshold.py        # section 5.12 (ESS chosen by threshold)
python experiments/report_mae_comparison.py --results-root results   # section 5.13
```

Outputs: `results/theory_ablation.json`, `results/ess_coverage.json`, `results/bound_threshold.json`, `results/bound_control.json`, `results/ess_threshold.json`, `results/mae_comparison.json`.
