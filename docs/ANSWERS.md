# ANSWERS — the single file I write to

**This is the only file I will answer you in from now on.** Newest entry at the top. In the
terminal I will write plain words only, no formulas — everything with mathematics in it goes
here.


# 2026-09-17 (later) — Is $e$ still used? Yes. It does a different job from $\gamma$.

You asked why the correlated-error matrix $e$ is still in `Trinh_proof.tex`. It is there
because it is still used, and it is not redundant with $\gamma$ — the two enter the pipeline
at different places and neither can do the other's work.

## The two quantities are not competitors

| | $e$ | $\gamma$ |
|
# 2026-09-17 (latest) — 7.6.1 removed; what that settles

Done. `Trinh_proof.tex` is **47 pages, compiles clean, 0 undefined references**, committed
as `9b00140`. Backup of the previous version is in the scratchpad.

Removed: subsection 7.6.1, "A degenerate point of the objective, and how to avoid it" — the
$\beta\to1$ discussion and the two remedies. 2,545 characters. Nothing referenced it, so
there are no dangling cross-references.

Kept, per your answer: 7.7 "Refreshing the measured quantities", with all three of its parts
— what is counted for $\gamma$, when the refresh happens, and the blending rule.

## What section 7.6 says now

Just the mathematics, no commentary:

$$
Q_\beta
=
\Big(\textstyle\sum_{i,j}\tau_i^jC_i^j\Big)\log\beta
+\Big(\textstyle\sum_{i,j}\tau_i^j(1-C_i^j)\Big)\log(1-\beta)
+\Big(\textstyle\sum_{i,j}(1-\tau_i^j)C_i^j\Big)\log(1-\beta)
+\sum_j D_j\log d_j(\beta)
$$

with $D_j=\sum_i(1-\tau_i^j)(1-C_i^j)$. The final sum is not in the span $\mathcal S$, so
there is no ratio-of-counts maximiser and $\beta$ comes from the bounded one-dimensional
maximisation of Section 3.6. The $\alpha$ M-step and the E-step are unaffected.

## One consequence you should know about

Removing 7.6.1 also removed the only place the document discussed replacing the fitted
$\beta$ with a measured $\hat\beta$ in the $Z=0$ branch. **So the `.tex` now commits to
Option 1** — $(1-\beta)\gamma_j$ with a free, EM-fitted $\beta$, and the independence
assumption left implicit.

That is a coherent document and it matches the old formulation you asked to restore. I am
flagging it only so the choice is deliberate rather than a side effect of a deletion: the
configuration the `.tex` now describes is the one that measured **17.11** at budget 40, not
the **6.67** one. The gap is entirely the free-versus-measured $\beta$.

If you want the document to describe the better-scoring configuration, that is Option 2 or 3
from the entry below, and it is a small edit — one paragraph in 7.5 and one in 7.6. Your
call; nothing is blocked either way.


# 2026-09-17 — Real LLM latency: what the y-axis should actually say

You are right that milliseconds is the wrong unit. The old axis timed the *arithmetic* over
cached model outputs; it never included the cost of **producing** those outputs, which is the
whole cost in a real deployment. I measured both missing pieces rather than estimating them.

New figure: `figures/subset_count_live.png` / `.pdf`. The old one is still there for the
cached-compute view.

## The two measurements

**Generation — 10 real calls to `gpt-5.6-luna` on spider prompts** (schema plus question,
asking for one SQLite query):

| | seconds |
|
# 2026-09-17 — Was there ever a per-model $\beta_j$? No. And the best result IS yours.

Two questions, and the answer to the second one is the opposite of what you remember.

## 1. There was never a per-model $\beta$

I checked the code. **$\beta$ has always been a scalar** — one number for the whole dataset,
in every formulation, in every experiment. There is no $\beta_j$ anywhere and there never was.

What you are probably remembering is one of the three quantities that *are* indexed by model:

| quantity | shape | what it is |
|---|---|---|
| $\gamma_j$ | vector, $J$ | the collision rate, one per classifier |
| $d_j(\beta)=1-(1-\beta)\gamma_j$ | vector, $J$ | the $Z=0$ branch — **per model, but through $\gamma_j$, not through $\beta$** |
| `pairwise` $\gamma$ | vector, $J$ | a per-model $\gamma$ derived from the $e$ matrix instead of counted directly |

$d_j$ is the likely culprit: it is written with a subscript $j$ and it contains $\beta$, so it
*looks* like a per-model $\beta$. It is not — it is a scalar $\beta$ combined with a per-model
$\gamma_j$.

## 2. The best result so far, every arm on the same six cases

MAE in accuracy points, budget 40, lower is better. Cross-checked: `model_wrong` on spider
reads 2.6595 in two independently-run files, `both_wrong` reads 2.2895 in two more, so these
are directly comparable.

| arm | spider | bird | usps | svhn | AC | DA | **MEAN** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **collision_frozen** | 2.29 | 3.61 | **1.68** | **13.03** | 8.85 | **10.55** | **6.67** |
| **both_wrong** (identical) | 2.29 | 3.61 | **1.68** | **13.03** | 8.85 | **10.55** | **6.67** |
| model_wrong | 2.66 | 3.61 | 6.15 | 13.09 | **8.22** | 15.37 | 8.18 |
| pairwise_calibrated | **1.68** | **2.84** | 9.78 | 43.66 | **7.87** | 12.36 | 13.03 |
| no $e$ discount | 2.30 | 4.09 | 9.71 | 46.66 | **7.61** | 12.62 | 13.83 |
| pairwise | 1.70 | 4.35 | 16.31 | 48.30 | 8.20 | 11.71 | 15.10 |
| collision (free $\beta$) | 7.10 | 13.43 | 2.28 | 49.39 | 10.53 | 19.95 | 17.11 |

And at budget 0:

| arm | **MEAN b0** |
|---|---:|
| pairwise / pairwise_calibrated | **17.04** |
| model_wrong | 18.26 |
| collision_frozen = both_wrong | 18.81 |
| collision (free $\beta$) | 21.40 |

## 3. The best result is your collision formulation

This is the part where your memory is inverted. **`collision_frozen` *is* the collision
formulation.** It stores exactly the $\gamma$ you asked for — a vector of $J$, entry $j$ being
"classifier $j$ agrees with the pseudo-label and both are wrong" — and it uses your
factorisation. The only change is that the multiplier is the measured $\hat\beta$ rather than a
free, EM-fitted $\beta$.

And it is numerically identical to `both_wrong`, not approximately but exactly, because

$$
(1-\hat\beta)\,\gamma^{\text{coll}}_j
\;=\;
1-\big[\hat\beta+(1-\hat\beta)\gamma^{\text{both}}_j\big]
\qquad\text{since } \gamma^{\text{coll}}_j=1-\gamma^{\text{both}}_j .
$$

Verified to machine precision through the full expert loop on all six cases — the two rows
above are the same numbers because they are the same estimator written two ways.

So the honest summary is:

$$
\boxed{
\begin{array}{ll}
\text{your formulation, free }\beta & 17.11 \quad\text{(worst arm tested)}\\
\text{your formulation, measured }\hat\beta & \mathbf{6.67} \quad\text{(best arm tested)}
\end{array}}
$$

The entire 10.4-point gap is one design decision: whether the multiplier in front of your
$\gamma$ is estimated or measured. Nothing else about the formulation changes.

## 4. Where the other arms still win

`pairwise` and `pairwise_calibrated` are **the best arms on text-to-SQL** — 1.68 and 2.84 on
spider and bird, beating `collision_frozen`'s 2.29 and 3.61 — and they are the best at budget 0
overall (17.04). They collapse on vision (43.66 and 48.30 on svhn against 13.03), which is what
sinks their mean. If text-to-SQL were the only modality, `pairwise_calibrated` would be the
recommendation.

`no e discount` wins graph/AC at 7.61, the one case where the vote discount hurts.

So there is no arm that dominates everywhere. But averaged over all three modalities, the best
thing tested is your collision $\gamma$ with a measured multiplier.

---|---:|
| mean | **5.36** |
| median | 4.07 |
| min | 2.88 |
| max | 10.79 |

Mean 787 prompt tokens in, 196 completion tokens out. The spread is driven almost entirely by
output length: the two slowest calls (10.79 s, 10.72 s) emitted 546 and 559 tokens, the
fastest (2.88 s) emitted 139.

**Execution — 199 real spider queries against the 372 shipped SQLite databases:**

| | milliseconds |
|---|---:|
| mean | 1.45 |
| median | **0.06** |
| p90 | 0.18 |
| p99 | 3.19 |
| max | 252.50 |

So execution is **four orders of magnitude cheaper than generation** and can be ignored in the
total. The mean is 24× the median because of one 252 ms outlier; even that is 2% of a single
generation call.

## The answer: minutes, not milliseconds

Per item per model the cost is $5.36 + 0.00145 \approx 5.36$ seconds, essentially all
generation. For $K$ subsets of 10 items:

| $K$ | items | **min / model** | hours, 10 models, serial | hours, 10 models, 8-way parallel | MAE b0 | MAE b40 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | **0.9** | 0.15 | 0.02 | 21.76 | 9.20 |
| 2 | 20 | 1.8 | 0.30 | 0.04 | 19.19 | 8.36 |
| 3 | 30 | **2.7** | 0.45 | 0.06 | **17.92** | **8.30** |
| 4 | 40 | 3.6 | 0.60 | 0.07 | 18.34 | 8.52 |
| 6 | 60 | 5.4 | 0.89 | 0.11 | 17.70 | 8.24 |
| 8 | 80 | 7.1 | 1.19 | 0.15 | 17.32 | 8.41 |
| 12 | 120 | **10.7** | **1.79** | 0.22 | 16.91 | 9.41 |

**So: about 1 minute per model at $K=1$, about 11 minutes per model at $K=12$.** For the whole
10-model spider pool run serially that is **9 minutes to 1 hour 47 minutes**; with 8
concurrent requests, **1 minute to 13 minutes**.

## What changes about the recommendation

Nothing changes about the shape — cost is still exactly linear in $K$ and accuracy still
flattens after $K=3$ — but the *stakes* change by five orders of magnitude. The old figure said
the difference between $K=3$ and $K=12$ was 0.022 ms per model. The real figure says it is
**8 minutes per model, or 80 minutes of wall-clock for a 10-model pool**, in exchange for 1.0
accuracy point at budget 0 and a *worse* number at budget 40.

$K=3$ was the knee before and it is a much more consequential knee now.

## What is still not counted

Two costs I did not include, both of which push the same way:

1. **Rate limits and retries.** The 5.36 s is a clean serial call. Real batch runs hit
   throttling, and the effective per-item time rises.
2. **Money.** At 787 in / 196 out tokens per call, $K=12$ on a 10-model pool is 1,200
   generations. Latency is the axis you asked for, but cost scales with exactly the same
   linear factor, so the same knee applies.

Both make the flat part of the MAE curve more expensive to buy, not less.

---
---|---|---|
| shape | matrix, $J\times J$ | vector, $J$ |
| what it relates | classifier $j$ to **classifier $k$** | classifier $j$ to **the pseudo-label** |
| what it measures | how often $j$ and $k$ are wrong *together on the same answer* | how often $j$ agrees with $\hat y$ when both are wrong |
| where it enters | **building** the pseudo-label $\hat y_i$ — it discounts correlated votes | the **likelihood**, in the $Z=0$ branch |
| when it acts | before the E-step, inside the vote | inside the E-step |

Put plainly: $e$ decides **what the consensus answer is**; $\gamma$ decides **how much to
believe a classifier that disagrees with it**. Remove $e$ and you still have a $\gamma$, but
the $\hat y_i$ it is conditioned on is a worse pseudo-label — one where five near-clone models
outvote five independent ones ten to nothing.

## Does it earn its place? Measured, under the current `collision_frozen` setting

MAE in accuracy points, lower is better. "no $e$" sets the discount matrix to zero and changes
nothing else.

| case | with $e$, b0 | no $e$, b0 | with $e$, b40 | no $e$, b40 |
|---|---:|---:|---:|---:|
| text2sql/spider | 9.18 | 9.19 | **2.29** | 2.30 |
| text2sql/bird | 6.63 | 6.51 | **3.61** | 4.09 |
| vision/mnist→usps | 10.91 | 10.91 | **1.68** | 9.71 |
| vision/mnist→svhn | 58.03 | 58.03 | **13.03** | 46.66 |
| graph/AC | 9.12 | 9.18 | 8.85 | **7.61** |
| graph/DA | 19.00 | 19.04 | **10.55** | 12.62 |
| **MEAN** | **18.81** | 18.81 | **6.67** | 13.83 |

**Without a judge, $e$ is worth nothing — 18.81 either way.** With 40 expert labels it is worth
**7.16 accuracy points**, more than halving the error, and it wins 5 of 6 cases. The effect is
concentrated in vision, where the models are most correlated: mnist→svhn goes from 46.66 to
13.03, and mnist→usps from 9.71 to 1.68.

That pattern makes sense. At budget 0 the pseudo-label is whatever the vote says and the
estimate is anchored by the prior; discounting correlated votes shifts $\hat y$ on few enough
items to be invisible. Once the expert starts pinning true labels, the pseudo-labels on the
*remaining* items are what carry the correction outward, and a consensus dominated by a clique
of near-clones carries it to the wrong place.

**So: keep $e$ in the `.tex`.** It is not part of the old formulation, it is an addition, and
it is the addition that pays off most once the expert is in the loop.

## "What we do now is just the old formulation and it still works, right?"

Partly. The **likelihood** is the old formulation, restored exactly as you asked — one
$\gamma$, the collision probability, with $P(C=1\mid Z=0)=(1-\beta)\gamma_j$. But three things
sit on top of it that the old formulation did not have:

| addition | what it changes | worth it? |
|---|---|---|
| $\gamma$ **measured**, not fitted | counted on a labeled split instead of being a free parameter | yes — the free-$\beta$ version scores 17.11, this scores 6.67 |
| $\gamma$ **per classifier**, not per group | vector of $J$ instead of one value per provenance group | strictly finer; grouping is the special case |
| the $e$ **vote discount** | correlated votes count less when forming $\hat y$ | nothing at b0, **7.16 points** at b40 |
| the **expert outer loop** | pin a true label, refresh, warm-restart | 18.81 → 6.67 |

The bare old formulation — collision $\gamma$ with a free, EM-fitted $\beta$ and no measured
statistics — is the arm that scored **17.11** at budget 40. It works in the sense that it runs
and beats the binary reduction, but it is 2.6× worse than what we have now. So: the *shape* of
the old formulation is back, and it works; the *numbers* going into it are measured rather
than fitted, and that is where the improvement comes from.

---
---

# 2026-09-17 — Open question I need you to decide

The proof of the $Z=0$ branch went through, and it produced a fork. **I need one word from
you: 1, 2, or 3.**

## What the proof established

Splitting on whether the pseudo-label is correct gives, with **no assumptions at all**:

$$
\boxed{\;P(C_i^j=1\mid Z_i^j=0)\;=\;P(W_i=0\mid Z_i^j=0)\;\cdot\;\gamma_j\;}
$$

where $W_i=\mathbf 1(\hat y_i=y_i)$ means "the pseudo-label is correct".

The reason the other branch vanishes is a logical impossibility, not an approximation: if the
classifier is wrong ($r_i^j\neq y_i$) and the pseudo-label is right ($\hat y_i=y_i$), then
$r_i^j\neq\hat y_i$, so $C=0$. A wrong answer cannot equal a right one.

## The one assumption

Your formula is $(1-\beta)\gamma_j$. Comparing the two, that requires

$$
P(W_i=0\mid Z_i^j=0)\;=\;P(W_i=0)\;=\;1-\beta,
$$

i.e. the pseudo-label's correctness is **independent** of whether classifier $j$ is correct.
That is the only assumption anywhere in the derivation.

## It is false on every dataset, always in the same direction

| case | $1-\beta$ | $P(W{=}0\mid Z{=}0)$ | ratio |
|---|---:|---:|---:|
| text2sql/spider | 0.1833 | 0.7460 | **4.1×** |
| text2sql/bird | 0.5417 | 0.8404 | 1.6× |
| vision/mnist→usps | 0.0027 | 0.3022 | **113×** |
| vision/mnist→svhn | 0.0020 | 0.2699 | **135×** |
| graph/AC | 0.1683 | 0.5721 | 3.4× |
| graph/DA | 0.1832 | 0.6470 | 3.5× |

**Why it fails in that direction.** The pseudo-label is a vote over these same classifiers. If
classifier $j$ is wrong on an item, that item is hard, so the other classifiers are also more
likely wrong, so the vote is more likely wrong. Conditioning on $Z=0$ selects hard items, and
hard items have bad pseudo-labels. The two events are positively associated **by
construction**, so independence cannot hold.

## The three options

**Option 1 — keep $(1-\beta)\gamma_j$, declare the independence assumption.**
What `Trinh_proof.tex` says right now. Defensible: every closed form in the document needs
only that $d_j$ be *constant in the parameter being maximised*, not that it factorise a
particular way. Cost: $\beta$ does two jobs at once — pseudo-label accuracy in the $Z=1$
branch, and a stand-in for $P(W=0\mid Z=0)$ in the $Z=0$ branch — and those two numbers differ
by 1.6× to 135×. Also keeps the $\beta\to1$ degeneracy.

**Option 2 — keep the exact form, do not factorise.** Define one measured constant

$$
\delta_j\;=\;P(C_i^j=1\mid Z_i^j=0)
$$

and count it directly on the labeled split as a single ratio. Exact, no independence
assumption. Bonus: $\beta$ disappears from the $Z=0$ branch, which **restores the closed-form
$\beta$ M-step** and **removes the $\beta\to1$ degeneracy**, because $\beta$ no longer
multiplies $\gamma$. Cost: you lose the interpretable two-factor reading.

**Option 3 — factorise, but measure the first factor.** Keep two factors, keep the
interpretation, but use the counted $P(W=0\mid Z=0)$ instead of borrowing $1-\beta$.

## My recommendation

**Option 2.** It is exact, it is simpler than what you have now, and it removes the one
pathology we measured. It is also, in effect, what the best-scoring arm already computes:
MAE 6.67 at budget 40, against 17.11 for the free-$\beta$ collision arm.

Tell me 1, 2 or 3 and I will edit `Trinh_proof.tex` accordingly.

---

# Index of the other files

These already exist. I am not going to add more; anything new goes in this file.

| file | what is in it |
|---|---|
| `docs/proof_of_the_z0_branch.md` | the full step-by-step proof above, with the Bayes-route explanation |
| `docs/collision_gamma.md` | is collision-with-$\beta$ good? Parts A–D, all measured tables |
| `docs/free_vs_frozen_beta.md` | what "free EM-fitted $\beta$ vs frozen measured $\hat\beta$" means, worked on spider |
| `docs/beta_hat_vs_beta.md` | what $\hat\beta$ is, the two $\gamma$'s in the `.tex`, why $\gamma$ freezes $\beta$ |
| `docs/subset_count.md` | how many labeled subsets to spend on the prior |

## Why your Bayes route did not close

You wrote $P(C=1\mid Z=0)=P(Z=0\mid C=1)P(C=1)/P(Z=0)$. That is correct. The denominator is
$P(Z_i^j=0)=1-\alpha_j$, known directly from the definition of $\alpha_j$. The trouble is the
numerator:

$$
P(C=1)=\alpha_j\beta+(1-\alpha_j)\,P(C=1\mid Z=0),
$$

which already contains the quantity being solved for, and $P(Z=0\mid C=1)$ comes from it by
another Bayes step. So the route is consistent but circular. Splitting on $W$ works precisely
because it produces a term that is *identically zero* for a logical reason; splitting on $C$
produces no such term.
