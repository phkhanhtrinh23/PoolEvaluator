# ANSWERS — the single file I write to

**This is the only file I will answer you in from now on.** Newest entry at the top. In the
terminal I will write plain words only, no formulas — everything with mathematics in it goes
here.

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
