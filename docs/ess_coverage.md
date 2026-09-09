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
```

Outputs: `results/theory_ablation.json`, `results/ess_coverage.json`.
