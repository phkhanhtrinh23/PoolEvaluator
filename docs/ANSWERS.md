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
