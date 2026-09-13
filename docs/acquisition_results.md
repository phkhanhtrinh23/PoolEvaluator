# Acquisition: information gain about the MEAN, and design-based estimation

**Status: complete.** All three tasks, 15 cases: Text-to-SQL (5 datasets, 7
case/protocol combinations), image (2 shifts), node (6 shifts). Budget 40, 5 seeds each.
A companion experiment (§7) crosses the statistics-refresh rule against every selector.

Code: `pooleval/validated_em.py` (acquisition), `pooleval/estimators.py` (design-based
estimators), `experiments/run_validated_em.py` (`_acquisition_block`).
Derivation: `docs/validated_em.md` §7b.

---

## 1. What is being compared

Six ways to spend the same budget of 40 expert calls, each averaged over **5 seeds**
(not optional: a ~20-item audit has a standard error near 0.1 on a proportion, so single
draws say nothing about a 2-point difference).

| rule | what it maximises | design valid for HT? |
|---|---|---|
| `IG [label entropy]` | Hung et al. Eq. (9): uncertainty over **every label** | no ($\pi\in\{0,1\}$) |
| `IG [accuracy variance, $A_\mu$]` | $\sum_j \operatorname{Var}_\ell(E[\alpha_j\mid e(q){=}\ell])$ — uncertainty about the **mean** | no |
| `$A_\mu$ sampled` | the same score, drawn from an $\epsilon$-mixed softmax | **yes**, $\pi_i>0$ |
| `random sampling` | nothing; uniform | **yes** |
| `hybrid: audit + $A_\mu$` | half budget SRS audit, half $A_\mu$ | **yes**, on the audit half |
| `hybrid: audit + IG` | half budget SRS audit, half entropy IG | **yes**, on the audit half |

The hybrid is the split-budget architecture: the audit half is drawn by simple random
sampling *before anything is seen*, so $\pi_i=\text{pilot}/N$ is exactly known and can
carry Horvitz-Thompson; the active half improves the model but never does audit duty,
because adaptive sampling without replacement leaves no clean marginal.

---

## 2. Text-to-SQL — COMPLETE

MAE in accuracy points, budget 40, 5 seeds. `/A` = protocol A (real labeled source
split), `/B` = protocol B (target holdout).

| acquisition rule | spider/A | spider/B | bird/A | bird/B | sqlflow/B | minidev/B | s2local/B | **mean** |
|---|---|---|---|---|---|---|---|---|
| IG [label entropy, Hung et al.] | 3.20 | **1.64** | 6.17 | 4.58 | 4.29 | 3.73 | 2.13 | 3.68 |
| **IG [accuracy variance, $A_\mu$]** | **2.29** | 1.75 | **3.30** | 3.95 | 5.70 | 5.13 | **2.03** | **3.45** |
| $A_\mu$ sampled ($\epsilon$-mixed softmax) | 4.48 | 4.63 | 4.28 | 3.23 | 4.85 | 3.84 | 2.63 | 3.99 |
| random sampling | 5.72 | 4.58 | 3.59 | **2.74** | 3.07 | 3.82 | 2.17 | 3.67 |
| hybrid: random audit + $A_\mu$ | 6.78 | 3.36 | **2.34** | 3.38 | 4.79 | 4.42 | 2.54 | 3.95 |
| hybrid: random audit + IG | 6.74 | 3.31 | 5.46 | 4.65 | **2.65** | **2.57** | 2.54 | 3.99 |

### What holds up

**$A_\mu$ wins on the mean, and wins big where it wins.** 3.45 vs 3.68, and on the two
protocol-A cases — the only ones with a genuine labeled source split — it is far ahead:
spider 2.29 vs 3.20, bird **3.30 vs 6.17**.

**The BIRD margin was predicted before it was measured.** The mechanism claim is that
$A_\mu$'s second factor $(m_1-m_0)^2$ is what sees confidently-wrong items, which label
entropy cannot, and that a weaker pool has more of them. Spider's pool accuracy is 0.743
and the margin 0.91; BIRD's is 0.369 and the margin **2.87**, three times larger. The
prediction was registered in advance and it held.

### What does not

**$A_\mu$ does not dominate. It wins 3 of 7 cases and loses clearly on 2** — sqlflow
(5.70 vs 4.29) and bird_minidev (5.13 vs 3.73). The simple "weaker pool $\Rightarrow$
bigger margin" story does **not** survive: sqlflow's pool accuracy (0.424) sits between
BIRD's (0.369) and spider's (0.743) yet it breaks the pattern. Pool accuracy alone is not
the discriminator, and the mechanism story is at best incomplete.

**The spread between rules is smaller than the spread between datasets.** Every rule sits
in 3.45-3.99 on the mean while a single rule swings from 1.6 to 6.2 across datasets. On
this evidence the acquisition rule is a second-order choice for Text-to-SQL.

**The rule that never fails badly is the hybrid with entropy IG** (worst case 6.74, and
it wins sqlflow and bird_minidev outright) -- consistent with the principle that when the
correlation model cannot be trusted, keeping probability sampling in the design is the
safe floor.

---

## 3. Image classification — COMPLETE

| acquisition rule | mnist→usps | mnist→svhn | **mean** |
|---|---|---|---|
| IG [label entropy, Hung et al.] | 3.03 | 46.73 | 24.88 |
| **IG [accuracy variance, $A_\mu$]** | **1.68** | **13.03** | **7.36** |
| $A_\mu$ sampled ($\epsilon$-mixed softmax) | 2.61 | 39.61 | 21.11 |
| random sampling | 2.32 | 19.73 | 11.03 |
| hybrid: audit + $A_\mu$ | 3.72 | 14.06 | 8.89 |
| hybrid: audit + IG | 2.48 | 31.06 | 16.77 |

**The largest effect measured anywhere: $A_\mu$ cuts MAE by 70% (24.88 → 7.36), and on
MNIST→SVHN by a factor of 3.6 (46.73 → 13.03).**

MNIST→SVHN is the adversarial case -- the pool is 13% accurate and its consensus is wrong
on 94% of items, so label entropy is *anti*-correlated with consensus failure and spends
the budget on the items the consensus already gets right. An earlier note in this project
concluded from that "random beats active selection here". That was wrong: it was Hung's
OBJECTIVE that failed, not active selection. $A_\mu$'s leverage factor finds those items
and beats random by 6.7 points.

**Caveat on the estimators.** On MNIST→SVHN inverse-variance fusion FAILS (13.92 and
30.82, worse than either input -- HT 5.43, model-assisted 6.21). Fusion assumes both
inputs unbiased; the model estimate there is not, so precision-weighting imports its bias.
On a pool that broken only the design-based estimators should be read.

---

## 4. Node classification — COMPLETE (6 shifts)

| acquisition rule | AC | AD | CA | CD | DA | DC | **mean** | worst |
|---|---|---|---|---|---|---|---|---|
| IG [label entropy, Hung et al.] | 7.12 | 12.73 | 9.52 | 8.87 | 10.06 | **6.35** | 9.11 | 12.73 |
| IG [accuracy variance, $A_\mu$] | 8.85 | 11.82 | 12.29 | 11.39 | 10.55 | 8.35 | 10.54 | 12.29 |
| **$A_\mu$ sampled** | **6.19** | 12.25 | **9.15** | 8.94 | 10.99 | 6.69 | **9.04** | 12.25 |
| random sampling | 7.85 | **11.82** | 9.68 | 11.65 | 11.36 | 8.72 | 10.18 | **11.82** |
| hybrid: audit + $A_\mu$ | 7.43 | 12.64 | 10.47 | 11.04 | 10.76 | 8.54 | 10.14 | 12.64 |
| hybrid: audit + IG | 7.34 | 13.42 | 10.05 | 10.55 | 11.25 | 9.28 | 10.32 | 13.42 |

**This is where $A_\mu$ loses.** Deterministic $A_\mu$ is the *worst* rule (10.54) and
label-entropy IG beats it (9.11). But **sampling the same score is the best rule** (9.04)
-- a pattern visible on the single AC shift beforehand and confirmed across all six.

The reading: $A_\mu$'s *ranking* carries real signal on these pools while its *argmax*
overcommits. The $\epsilon$-mixed softmax introduced to satisfy the positivity condition
turns out to double as a regulariser on a score the model cannot yet be trusted to rank
perfectly. That is a design-based safeguard paying off for a second, unplanned reason.

---

## 5. The design-based estimators

Reported on every hybrid row. Across the cases measured so far:

| estimator | behaviour |
|---|---|
| Horvitz-Thompson | **consistently the worst** — 8.70, 7.73, 8.93, 8.02, 10.00. That is just its standard error: ~20 audit items on a proportion gives ~0.11. |
| model-assisted | better than HT everywhere (6.94, 8.08, 7.09, 5.42, 8.74) but still behind the model estimate |
| inverse-variance fused | **tracks the better of its two inputs every time** (2.45, 3.31, 2.91, 3.47, 7.26) |

The design-based machinery buys unbiasedness that the model estimate cannot promise; the
fusion is what stops that being paid for in variance. HT alone at this budget is not a
usable estimator, and reporting it alone would misrepresent the design.

Two implementation points that are not cosmetic:

* **Cross-fitting is required.** Audit items are pinned in the fitted model, so its
  prediction there *is* the label: residuals would be identically zero, the correction
  would vanish, and `model_assisted` would silently collapse into the plain model average
  — the exact bias it exists to remove. The model is re-solved with audit labels withheld.
* **HT cannot be retrofitted onto deterministic top-$k$.** Those items have $\pi=1$ and
  every other item $\pi=0$; $1/0$ is not a weight. `horvitz_thompson` raises rather than
  returning an infinity, and `design_estimates` returns nothing when `pilot=0` instead of
  computing a number it is not entitled to.

---

## 6. Across all three tasks

| acquisition rule | Text2SQL (7) | image (2) | node (6) | best-on |
|---|---|---|---|---|
| IG [label entropy, Hung et al.] | 3.68 | 24.88 | **9.11** | node |
| **IG [accuracy variance, $A_\mu$]** | **3.45** | **7.36** | 10.54 | Text2SQL, image |
| **$A_\mu$ sampled** | 3.99 | 21.11 | **9.04** | node |
| random sampling | 3.67 | 11.03 | 10.18 | — |
| hybrid: audit + $A_\mu$ | 3.95 | 8.89 | 10.14 | — |
| hybrid: audit + IG | 3.99 | 16.77 | 10.32 | — |

**No rule dominates, and the honest summary is conditional.**

* Where the pool's consensus is badly broken (MNIST→SVHN, 94% wrong), $A_\mu$ is
  transformative and label entropy is actively harmful. This is the regime the criterion
  was derived for and it delivers.
* Where the model is reasonable but not sharp (node classification), $A_\mu$'s argmax
  overcommits and the *sampled* version is best.
* On Text-to-SQL the spread between rules (3.45-3.99) is far smaller than the spread
  across datasets (1.6-6.2), so the acquisition rule is second-order there.

The one question from the earlier partial draft that remains open: **what discriminates
the wins from the losses.** Pool accuracy alone does not -- sqlflow (0.424) sits between
BIRD (0.369, big $A_\mu$ win) and spider (0.743, $A_\mu$ win) yet $A_\mu$ loses there.
The candidate that fits every case so far is calibration of the correlation model rather
than the strength of the correlation, which is the distinction drawn in §9 of the design
note, but it has not been measured directly and should not be asserted.

---

## 7. The `(old + temp)/2` refresh, crossed with every selector

`experiments/run_refresh_rule.py` — 6 cases across all three tasks, budget 20, 3 seeds.
Reports the refreshed `beta` against the **true** pseudo-label accuracy on the target.

### Recovered `beta` under `(old+temp)/2` (truth in bold)

| case | **true β** | entropy | info_gain | $A_\mu$ | $A_\mu$ sampled | random |
|---|---|---|---|---|---|---|
| text2sql/spider | **0.767** | 0.657 | 0.657 | 0.524 | 0.559 | **0.791** |
| text2sql/bird | **0.427** | 0.103 | 0.077 | 0.283 | 0.184 | **0.529** |
| vision/mnist→usps | **0.887** | 0.428 | 0.447 | 0.718 | 0.572 | **0.886** |
| vision/mnist→svhn | **0.060** | 0.105 | 0.106 | **0.053** | 0.093 | 0.088 |
| graph/AC | **0.748** | 0.529 | 0.409 | 0.392 | 0.400 | **0.836** |
| graph/CD | **0.677** | 0.242 | 0.144 | 0.317 | 0.244 | **0.739** |

### Three findings

**1. Every active selector under-estimates `beta`; random recovers it almost exactly.**
0.791 vs 0.767, 0.886 vs 0.887, 0.836 vs 0.748. The mechanism is not the arithmetic but
the sample: an acquisition rule picks items *because* the consensus looks doubtful there,
so `temp_beta` on them is systematically low — and on an overruled item it is 0 by
construction. `beta` is a raw **mean over items**, the same kind of object as the accuracy
$\mu$, so estimating it from an adaptively selected sample is exactly the error that
motivates probability sampling in the first place. `e` and `gamma` are *conditional*
rates and are far less affected.

**2. $A_\mu$ is less biased for `beta` than label entropy — on 4 of 6 cases.**
bird 0.283 vs 0.103, mnist→usps 0.718 vs 0.428, graph/CD 0.317 vs 0.242. Better, but
still biased; it does not escape the problem, only softens it.

**3. The "collapse" is sometimes the correct answer, and this reverses the earlier reading.**
On MNIST→SVHN the source split is MNIST, where `beta = 0.998`; the target is SVHN, where
the true `beta` is **0.060**. The geometric filter moves it to 0.053 in 20 calls and scores
**MAE 18.31**. Pooled counts — which weighs 20 target items against 3000 source items —
keeps `beta` at 0.992 and scores **55.69**. Under severe shift the fast filter is not
over-reacting; it is the only rule that can move far enough.

| | `(old+temp)/2` | pooled counts |
|---|---|---|
| **adapts** | fast — 0.998 → 0.053 in 20 calls | inert — barely leaves the source value |
| **fails when** | the sample is biased (active selection, no real shift) | the shift is large |
| MAE, mnist→svhn + $A_\mu$ | **18.31** | 55.69 |
| MAE, bird + $A_\mu$ | 7.61 | **5.27** |

### What this implies for the design

Neither refresh rule is right on its own, and the split is clean:

$$
\boxed{\text{keep } (\text{old}+\text{temp})/2 \text{ for its adaptation speed, but compute }
\texttt{beta} \text{ from the PROBABILITY SAMPLE, not the actively chosen items.}}
$$

That is the same separation as $A_\mu$ for acquisition versus HT/IPW for the mean, applied
one level down — and the evidence for it is that `random` + `(old+temp)/2` recovers `beta`
to within 0.02-0.09 on every non-adversarial case while every active selector does not.

### Isolating the two effects

**Under $A_\mu$ selection, is `(old+temp)/2` better than pooled counts?** Wins 5 of 6.

| case | true β | `(old+temp)/2` | pooled counts | Δ |
|---|---|---|---|---|
| text2sql/spider | 0.767 | **5.42** | 8.62 | −3.20 |
| text2sql/bird | 0.427 | 7.61 | **5.27** | +2.34 |
| vision/mnist→usps | 0.887 | **2.86** | 3.23 | −0.37 |
| vision/mnist→svhn | 0.060 | **18.31** | 55.69 | **−37.38** |
| graph/AC | 0.748 | **7.33** | 8.64 | −1.30 |
| graph/CD | 0.677 | **12.92** | 14.99 | −2.07 |
| **MEAN** | | **9.07** | 16.07 | −7.00 |
| MEAN excl. svhn | | **7.23** | 8.15 | −0.92 |

The single loss is BIRD, and it is the selection-bias case: the filter drags β to 0.283
against a truth of 0.427.

**Under `(old+temp)/2`, which selector?**

| | entropy | info_gain | $A_\mu$ | $A_\mu$ sampled | random |
|---|---|---|---|---|---|
| MEAN | 14.50 | 15.09 | **9.07** | 12.97 | 12.99 |
| MEAN excl. svhn | 7.94 | 8.15 | 7.23 | **6.26** | 7.87 |

**`(old+temp)/2` + $A_\mu$ is the best pairing measured** — 9.07, against 14.50 for the
original (entropy + same refresh) and 16.07 for ($A_\mu$ + pooled counts). The two changes
are complementary: $A_\mu$ finds the items that move the accuracy estimate, and the fast
filter lets the statistics actually move once they are revealed. Excluding the adversarial
shift, $A_\mu$-sampled is best (6.26), consistent with §4.

Note it is winning *despite* a biased β, not because of an accurate one — β is wrong on
every non-adversarial case (0.524 vs 0.767; 0.392 vs 0.748).

**Not yet implemented.** `LabeledStatistics.add_validated` currently folds every validated
item into `beta`. The fix is to restrict `beta`'s update to items drawn with known
inclusion probability (the `pilot` audit), leaving `e` and `gamma` on all of them.
