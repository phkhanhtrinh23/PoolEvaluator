# Acquisition: information gain about the MEAN, and design-based estimation

**Status: complete, re-run under the current default.** All three tasks, 15 cases: Text-to-SQL (5 datasets, 7
case/protocol combinations), image (2 shifts), node (6 shifts). Budget 40, 5 seeds each.
A companion experiment (§7) crosses the statistics-refresh rule against every selector.

> **Re-run note.** Every table below was regenerated after `gamma_mode` changed from
> `both_wrong` to `model_wrong` (see `docs/three_betas.md`), which removes a biased
> conversion from the E-step and moves the underlying estimator substantially --- e.g.
> `validated EM (no judge)` on text2sql/A went 7.91 → 3.87. Numbers here are therefore not
> comparable with any earlier draft; §8 records what the change did to this comparison.

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

MAE in accuracy points, budget 40, 5 seeds. `/A` = protocol A (real labeled source split),
`/B` = protocol B (target holdout).

| acquisition rule | spider/A | bird/A | spider/B | bird/B | sqlflow/B | bird_minidev/B | spider2local/B | **mean** |
|---|---|---|---|---|---|---|---|---|
| IG [label entropy, Hung et al.] | **2.56** | 5.76 | 3.66 | 3.81 | 5.07 | 3.80 | 2.12 | 3.83 |
| IG [accuracy variance, $A_\mu$] | 2.66 | 3.96 | 3.38 | 2.82 | 8.34 | 6.13 | **2.02** | 4.19 |
| $A_\mu$ sampled ($\epsilon$-mixed softmax) | 3.08 | **2.54** | **2.75** | 2.37 | 7.02 | 4.44 | 2.16 | 3.48 |
| random sampling | 3.43 | 3.10 | 3.57 | **2.05** | 5.83 | 5.73 | 2.18 | 3.70 |
| hybrid: random audit + $A_\mu$ | 3.90 | 3.27 | 2.77 | 2.65 | 7.74 | 5.68 | 2.22 | 4.03 |
| hybrid: random audit + IG | 3.68 | 3.50 | 2.91 | 3.12 | **3.98** | **2.99** | 2.22 | **3.20** |

### What holds up

**$A_\mu$ is strongest where the labeled source split is real.** On the two protocol-A
cases it beats label-entropy IG 3.31 vs 4.16, and on BIRD/A 3.96 vs 5.76.

**Randomising the score is now the best Text2SQL rule on protocol A** ($A_\mu$ sampled,
2.81) --- it was 4.38 under the old `gamma_mode`. With a better-calibrated $\gamma$ the
ranking $A_\mu$ produces is worth sampling from rather than maximising.

### What does not

**$A_\mu$ loses protocol B** (4.54 vs 3.69), driven by sqlflow (8.34) and bird_minidev
(6.13). The best protocol-B rule is `hybrid: random audit + IG` at **3.05** --- the only
rule that is never bad on any Text2SQL case.

**The spread between rules stays smaller than the spread between datasets.** Rules span
3.31--4.54 on protocol B while a single rule spans 2.0--8.3 across datasets, so the
acquisition rule remains a second-order choice here.

---

## 3. Image classification — COMPLETE

| acquisition rule | mnist->usps | mnist->svhn | **mean** |
|---|---|---|---|
| IG [label entropy, Hung et al.] | **4.00** | 45.40 | 24.70 |
| IG [accuracy variance, $A_\mu$] | 6.15 | **13.09** | **9.62** |
| $A_\mu$ sampled ($\epsilon$-mixed softmax) | 5.38 | 36.74 | 21.06 |
| random sampling | 6.76 | 15.90 | 11.33 |
| hybrid: random audit + $A_\mu$ | 7.17 | 14.00 | 10.58 |
| hybrid: random audit + IG | 5.22 | 14.31 | 9.77 |

**$A_\mu$ cuts MAE by 61% against label-entropy IG (24.70 → 9.62)**, and on MNIST→SVHN by
a factor of 3.5 (45.40 → 13.09).

MNIST→SVHN is the adversarial case --- the pool is 13% accurate and its consensus wrong on
94% of items, so label entropy is *anti*-correlated with consensus failure and spends the
budget on items the consensus already gets right. An earlier draft concluded from this that
"random beats active selection here". That was wrong: it was Hung's OBJECTIVE that failed,
not active selection. $A_\mu$ beats random by 1.7 points and label entropy by 15.1.

**Caveat on the estimators.** On MNIST→SVHN inverse-variance fusion fails --- worse than
either input --- because fusion assumes both inputs unbiased and the model estimate there is
not. On a pool that broken only the design-based estimators should be read.

---

## 4. Node classification — COMPLETE (6 shifts)

| acquisition rule | AC | AD | CA | CD | DA | DC | **mean** |
|---|---|---|---|---|---|---|---|
| IG [label entropy, Hung et al.] | 5.89 | 11.66 | 7.69 | 8.44 | 8.59 | **5.48** | 7.96 |
| IG [accuracy variance, $A_\mu$] | 8.22 | 10.97 | 8.00 | 11.74 | 15.37 | 9.05 | 10.56 |
| $A_\mu$ sampled ($\epsilon$-mixed softmax) | 7.19 | 11.11 | **7.38** | 8.73 | 10.22 | 6.60 | 8.54 |
| random sampling | 8.24 | **8.08** | 7.54 | 7.55 | **6.48** | 10.07 | 7.99 |
| hybrid: random audit + $A_\mu$ | 11.51 | 10.23 | 9.98 | 9.91 | 11.60 | 10.16 | 10.56 |
| hybrid: random audit + IG | **5.45** | 9.15 | 8.53 | **6.56** | 10.80 | 6.86 | **7.89** |

**This is where deterministic $A_\mu$ loses** --- 10.56 against 7.96 for label-entropy IG,
worse on 5 of 6 shifts. But **sampling the same score recovers most of the gap** (8.54), and
`hybrid: random audit + IG` is best overall (7.89).

The reading: $A_\mu$'s *ranking* carries signal on these pools while its *argmax*
overcommits. The $\epsilon$-mixed softmax introduced to satisfy the positivity condition
doubles as a regulariser on a score the model cannot yet be trusted to rank perfectly.

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

| acquisition rule | Text2SQL/A (2) | Text2SQL/B (5) | image (2) | node (6) | best-on |
|---|---|---|---|---|---|
| IG [label entropy, Hung et al.] | 4.16 | 3.69 | 24.70 | 7.96 | — |
| IG [accuracy variance, $A_\mu$] | 3.31 | 4.54 | **9.62** | 10.56 | image (2) |
| $A_\mu$ sampled | **2.81** | 3.75 | 21.06 | 8.54 | Text2SQL/A (2) |
| random sampling | 3.27 | 3.87 | 11.33 | 7.99 | — |
| hybrid: audit + $A_\mu$ | 3.58 | 4.21 | 10.58 | 10.56 | — |
| hybrid: audit + IG | 3.59 | **3.05** | 9.77 | **7.89** | Text2SQL/B (5), node (6) |

**No rule dominates, and the winner changes per block.** Four rules take a block each.

* Where the pool's consensus is badly broken (MNIST→SVHN, 94% wrong), $A_\mu$ is
  transformative --- 9.62 against 24.70 for label entropy --- and label entropy is actively
  harmful. This is the regime the criterion was derived for and it delivers.
* Where the model is reasonable but not sharp (node classification, Text2SQL protocol B),
  $A_\mu$'s argmax overcommits and either the **sampled** variant or the **hybrid with
  label-entropy IG** is better.
* On Text-to-SQL the spread between rules stays smaller than the spread across datasets, so
  the acquisition rule remains second-order there.

**Averaging across blocks is dominated by the image block**, whose scale (9.6--24.7) is an
order of magnitude above the others (2.8--4.5). A cross-block mean would rank rules mostly
by how they handle MNIST→SVHN, so it is deliberately not reported.

**The most consistent rule is `hybrid: audit + IG`** --- best on two blocks, never worse
than 9.77, and the only rule that is never bad on any Text2SQL case. That is the design
keeping a known-$\pi$ audit in reserve, which is what makes it robust when the correlation
model cannot be trusted.

---

## 7. The `(old + temp)/2` refresh, crossed with every selector

`experiments/run_refresh_rule.py` — 6 cases across all three tasks, budget 20, 3 seeds.
Reports the refreshed measured pseudo-label accuracy (`LabeledStatistics.pseudo_accuracy`,
written $\hat\beta_{\text{meas}}$ --- **not** the M-step $\beta$, see `docs/three_betas.md`)
against the true value on the target.

### Recovered $\hat\beta_{\text{meas}}$ under `(old+temp)/2` (truth in bold)

| case | **true β** | entropy | info_gain | $A_\mu$ | $A_\mu$ sampled | random |
|---|---|---|---|---|---|---|
| text2sql/spider | **0.767** | 0.657 | 0.657 | 0.524 | 0.559 | **0.791** |
| text2sql/bird | **0.427** | 0.103 | 0.077 | 0.283 | 0.184 | **0.529** |
| vision/mnist→usps | **0.887** | 0.428 | 0.447 | 0.718 | 0.572 | **0.886** |
| vision/mnist→svhn | **0.060** | 0.105 | 0.106 | **0.053** | 0.093 | 0.088 |
| graph/AC | **0.748** | 0.529 | 0.409 | 0.392 | 0.400 | **0.836** |
| graph/CD | **0.677** | 0.242 | 0.144 | 0.317 | 0.244 | **0.739** |

### Three findings

**1. Every active selector under-estimates $\hat\beta_{\text{meas}}$; random recovers it almost exactly.**
0.791 vs 0.767, 0.886 vs 0.887, 0.836 vs 0.748. The mechanism is not the arithmetic but
the sample: an acquisition rule picks items *because* the consensus looks doubtful there,
so `temp_beta` on them is systematically low — and on an overruled item it is 0 by
construction. $\hat\beta_{\text{meas}}$ is a raw **mean over items**, the same kind of object as the
accuracy $\mu$, so estimating it from an adaptively selected sample is exactly the error that
motivates probability sampling in the first place. `e` and `gamma` are *conditional*
rates and are far less affected.

**2. $A_\mu$ is less biased for $\hat\beta_{\text{meas}}$ than label entropy — on 4 of 6 cases.**
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
item into `pseudo_accuracy`. The fix is to restrict `beta`'s update to items drawn with known
inclusion probability (the `pilot` audit), leaving `e` and `gamma` on all of them.

---

## 8. What the `gamma_mode` change did to this comparison

Every table above was regenerated after `gamma_mode` changed from `both_wrong` to
`model_wrong`. That change is about the E-step, not about acquisition, but it moves the
model the acquisition rules are scoring, so the comparison shifts with it.

| acquisition rule | Text2SQL/A | Text2SQL/B | image | node |
|---|---|---|---|---|
| IG [label entropy] | 4.68 → 4.16 (-0.52) | 3.27 → 3.69 (+0.42) | 24.88 → 24.70 (-0.18) | 9.11 → 7.96 (-1.15) |
| IG [$A_\mu$] | 2.79 → 3.31 (+0.52) | 3.71 → 4.54 (+0.83) | 7.36 → 9.62 (+2.26) | 10.54 → 10.56 (+0.02) |
| $A_\mu$ sampled | 4.38 → 2.81 (-1.57) | 3.84 → 3.75 (-0.09) | 21.11 → 21.06 (-0.04) | 9.04 → 8.54 (-0.50) |
| random | 4.66 → 3.27 (-1.39) | 3.28 → 3.87 (+0.59) | 11.03 → 11.33 (+0.30) | 10.18 → 7.99 (-2.19) |

**The gap between $A_\mu$ and label-entropy IG narrowed where the model got better
calibrated.** On Text2SQL/A the lead went from 1.89 points to 0.85; on image from 17.5 to
15.1. This was predicted before the re-run and it held: the two fixes were partly addressing
the same underlying miscalibration, so they do not simply add.

$A_\mu$'s leverage factor $(m_1-m_0)^2$ finds items where the model would be moved a lot by
the truth --- which includes items where the model is *wrong* about $\gamma$. Removing the
biased conversion removes some of that wrongness, so there is less for $A_\mu$ to exploit.
The criterion is therefore worth most exactly where the model is least trustworthy, which is
consistent with it remaining transformative on MNIST→SVHN (the one block where the estimator
still fails badly) and second-order on Text2SQL.

**Randomised selection gained where deterministic $A_\mu$ lost.** $A_\mu$ sampled improved
on Text2SQL/A (4.38 → 2.81) and node (9.04 → 8.54) while deterministic $A_\mu$ got worse on
both. With a better-calibrated model the ranking is more trustworthy than the argmax ---
which is the same conclusion §4 reaches from the node shifts alone.

---

## 9. Which `gamma_mode` for the headline?

§8 shows deterministic $A_\mu$ is the only rule that got *worse* when `gamma_mode` flipped
to `model_wrong` --- on all four blocks. Since the judge loop with $A_\mu$ is the headline
configuration, that comparison was run on its own: 15 cases (every case available), both
modes, both $A_\mu$ variants, budgets 10/20/40, 5 seeds.

| selector | mean $\Delta$ (model_wrong − both_wrong) | model_wrong better in |
|---|---|---|
| $A_\mu$ (deterministic) | **+0.93** | 20/45 cells |
| $A_\mu$ sampled | **−0.03** | 27/45 cells |

`both_wrong` wins for the argmax by 0.93 and the sampled variant is indifferent. The margin
is concentrated in the vision block, does not shrink with budget (contradicting the proposed
mechanism), and buys that 0.93 at the cost of a conversion measured to be biased by
0.28--0.47 on all ten labeled splits. Full analysis: `docs/gamma_mode_headline.md`.
