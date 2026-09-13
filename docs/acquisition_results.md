# Acquisition: information gain about the MEAN, and design-based estimation

**Status: partial.** Text-to-SQL is complete (5 datasets, 7 case/protocol combinations).
Image is 1 of 2 shifts, node 1 of 6; both are still running and the tables below are
marked accordingly. Analysis will be revised against the full table — nothing here is
final, and the pattern that currently looks strongest is also the one with the fewest
cases behind it.

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

## 3. Image classification — PARTIAL (1 of 2 shifts)

| acquisition rule | mnist→usps | mnist→svhn |
|---|---|---|
| IG [label entropy, Hung et al.] | 3.03 | *pending* |
| **IG [accuracy variance, $A_\mu$]** | **1.68** | *pending* |
| $A_\mu$ sampled | 2.61 (sd 0.55) | *pending* |
| random sampling | 2.32 (sd 0.55) | *pending* |
| hybrid: audit + $A_\mu$ | 3.72 → fused 3.47 | *pending* |
| hybrid: audit + IG | 2.48 → fused 2.46 | *pending* |

$A_\mu$ cuts MAE by **45%** (3.03 → 1.68), the largest margin measured anywhere so far.
The missing shift is the adversarial one — MNIST→SVHN has a pool that is 13% accurate
with a consensus wrong on 94% of items — so this row will very likely move the
conclusion and should not be read alone.

---

## 4. Node classification — PARTIAL (1 of 6 shifts)

| acquisition rule | AC | AD, CA, CD, DA, DC |
|---|---|---|
| IG [label entropy, Hung et al.] | 7.12 | *pending* |
| IG [accuracy variance, $A_\mu$] | 8.85 | *pending* |
| **$A_\mu$ sampled** | **6.19** (sd 0.80) | *pending* |
| random sampling | 7.85 (sd 1.06) | *pending* |
| hybrid: audit + $A_\mu$ | 7.43 → fused 7.26 | *pending* |
| hybrid: audit + IG | 7.34 → fused 7.20 | *pending* |

The one case so far where **deterministic $A_\mu$ is the worst rule (8.85) while the
SAME score sampled is the best (6.19)**. If that survives the remaining five shifts it is
the most interesting result in this table: it would say the $A_\mu$ ranking carries real
signal while its argmax overcommits, and that the $\epsilon$-mixed softmax introduced for
the positivity condition doubles as a regulariser on a score the model cannot yet be
trusted to rank perfectly. One shift is not enough to claim it.

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

## 6. To be revised when the full table lands

1. Whether $A_\mu$'s image win survives MNIST→SVHN, the adversarial shift.
2. Whether sampled-$A_\mu$ beating deterministic-$A_\mu$ holds across the six node shifts.
3. What actually discriminates the cases where $A_\mu$ wins from where it loses, given
   that pool accuracy alone demonstrably does not.
