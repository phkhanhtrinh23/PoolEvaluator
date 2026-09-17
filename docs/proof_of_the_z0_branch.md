# Proof of $P(C_i^j=1\mid Z_i^j=0)=(1-\beta)\gamma_j$

You are right that this needs a proof. Here it is, and it has exactly one step that is an
**assumption** rather than an identity. I will mark that step clearly and then measure how
good it is.

## Setup: three indicator variables

Fix one item $i$ and one classifier $j$. Three binary events, all about the same item:

| symbol | definition | in words |
|---|---|---|
| $Z_i^j$ | $\mathbf 1(r_i^j=y_i)$ | classifier $j$ is **correct** |
| $C_i^j$ | $\mathbf 1(r_i^j=\hat y_i)$ | classifier $j$ **agrees with the pseudo-label** |
| $W_i$ | $\mathbf 1(\hat y_i=y_i)$ | the **pseudo-label is correct** |

Note $W$ carries no index $j$ — there is one pseudo-label per item, shared by all
classifiers. The model's parameters are

$$\alpha_j=P(Z_i^j=1),\qquad \beta=P(W_i=1),\qquad \gamma_j=P\big(C_i^j=1\mid Z_i^j=0,\ W_i=0\big).$$

Read that last one carefully: $\gamma_j$ is **defined** as a probability conditional on *both*
being wrong. That is not a convention chosen for convenience — it is what makes the
derivation below work, and §4 shows what breaks if it is defined any other way.

## Step 1 — the law of total probability (exact, no assumptions)

$W_i$ is either 0 or 1, so we may split on it. For any events this is an identity:

$$
P(C=1\mid Z=0)
= P(C=1,\,W=1\mid Z=0)\;+\;P(C=1,\,W=0\mid Z=0)
$$

Apply the chain rule to each term:

$$
P(C=1\mid Z=0)
= \underbrace{P(W=1\mid Z=0)\,P(C=1\mid Z=0,W=1)}_{\text{(a)}}
\;+\;\underbrace{P(W=0\mid Z=0)\,P(C=1\mid Z=0,W=0)}_{\text{(b)}}
$$

Nothing has been assumed. This is just $P(A)=P(A\cap B)+P(A\cap B^c)$ followed by
$P(A\cap B)=P(B)P(A\mid B)$, both valid for any events.

## Step 2 — term (a) is exactly zero (exact, no assumptions)

Look at $P(C=1\mid Z=0,\,W=1)$. We are conditioning on:

- $Z=0$, i.e. $r_i^j\neq y_i$ — the classifier's answer is **not** the truth;
- $W=1$, i.e. $\hat y_i=y_i$ — the pseudo-label **is** the truth.

Substituting the second into the first: $r_i^j \neq y_i = \hat y_i$, so $r_i^j\neq\hat y_i$,
which is $C=0$. So $C=1$ is **impossible** under this conditioning:

$$\boxed{P(C=1\mid Z=0,\,W=1)=0}$$

This is a logical impossibility, not a small probability. A wrong answer cannot equal a right
one. Term (a) vanishes identically and we are left with

$$
\boxed{\;P(C=1\mid Z=0)\;=\;P(W=0\mid Z=0)\;\cdot\;\gamma_j\;}
$$

where the second factor is $\gamma_j$ **by its definition**. This equation is **exact**. No
assumption has been used anywhere.

## Step 3 — the one assumption

The claimed formula is $(1-\beta)\gamma_j$. Comparing with the exact result, the claim is
equivalent to

$$
P(W=0\mid Z=0)\;\stackrel{?}{=}\;P(W=0)\;=\;1-\beta .
$$

That is the statement that $W$ and $Z$ are **independent**: whether the pseudo-label is
correct tells you nothing about whether classifier $j$ is correct. It is an assumption, and
it is the *only* one in the derivation.

$$
\boxed{
\text{Assumption (I): } W_i \perp Z_i^j
\quad\Longrightarrow\quad
P(C=1\mid Z=0)=(1-\beta)\gamma_j
}
$$

## Step 4 — why your Bayes route does not give a derivation

You proposed

$$P(C=1\mid Z=0)=\frac{P(Z=0\mid C=1)\,P(C=1)}{P(Z=0)} .$$

That is **correct** — it is Bayes' rule, and it holds. To answer the specific question: the
denominator is not $P(Z=0,\beta)$ but $P(Z=0)$, and it is known,

$$P(Z_i^j=0)=1-\alpha_j$$

directly from the definition of $\alpha_j$. The problem is the numerator. The marginal
agreement rate is

$$P(C=1)=\alpha_j\beta+(1-\alpha_j)\,P(C=1\mid Z=0),$$

which already contains the quantity we are trying to compute, and $P(Z=0\mid C=1)$ is
obtained from it by another application of Bayes. So the route is **circular**: it re-expresses
the unknown in terms of two quantities that are themselves defined through the unknown. It is
consistent, but it derives nothing.

The productive decomposition is the one in Step 1 — split on $W$, not on $C$ — because
splitting on $W$ produces a term that is *identically zero* for a logical reason. That is
where the content comes from.

## Step 5 — is assumption (I) true? Measured on the labeled splits

$P(W=0\mid Z=0)$ and $1-\beta$ are both directly countable on labeled data, so this is
checkable rather than arguable.

| case | $1-\beta$ | $P(W{=}0\mid Z{=}0)$ mean | min | max | gap | ratio |
|---|---:|---:|---:|---:|---:|---:|
| text2sql/spider | 0.1833 | 0.7460 | 0.6061 | 0.8261 | +0.5627 | **4.1×** |
| text2sql/bird | 0.5417 | 0.8404 | 0.7857 | 0.9104 | +0.2988 | 1.6× |
| vision/mnist→usps | 0.0027 | 0.3022 | 0.1304 | 0.5556 | +0.2995 | **113×** |
| vision/mnist→svhn | 0.0020 | 0.2699 | 0.0909 | 0.5556 | +0.2679 | **135×** |
| graph/AC | 0.1683 | 0.5721 | 0.1480 | 0.8551 | +0.4039 | 3.4× |
| graph/DA | 0.1832 | 0.6470 | 0.1851 | 0.8852 | +0.4638 | 3.5× |

**Assumption (I) is false in every case, and the deviation is always in the same direction.**

The direction is not an accident, and it has a one-line explanation: the pseudo-label is a
*vote over these same classifiers*. On an item where classifier $j$ is wrong, the item is
usually hard, so the other classifiers are more likely to be wrong too, so the vote is more
likely to be wrong. $W$ and $Z$ are positively associated by construction. **Conditioning on
$Z=0$ selects hard items, and hard items have bad pseudo-labels.**

The ratio is worst exactly where you would expect: on mnist→svhn the labeled split is MNIST,
where the vote is right 99.8% of the time overall ($1-\beta=0.0020$), but on the rare items
where a given classifier fails, the vote is wrong 27% of the time — 135 times more often.

## What to do about it

Three options, and this is genuinely your call:

**(1) Keep $(1-\beta)\gamma_j$ and state assumption (I) as a modelling assumption.** This is
what the classical presentation does and what `Trinh_proof.tex` now says. It is defensible:
the closed forms in the document depend only on $d_j$ being *constant in the parameter being
maximised*, not on how $d_j$ factorises. The cost is that $\beta$ then has two jobs — it is
both the pseudo-label accuracy in the $Z=1$ branch and a stand-in for $P(W=0\mid Z=0)$ in the
$Z=0$ branch — and the table above says those two numbers differ by 1.6× to 135×.

**(2) Keep the exact form.** Do not factorise at all. Define one measured constant

$$\delta_j \;=\; P(W=0\mid Z=0)\,\gamma_j \;=\; P(C=1\mid Z=0)$$

and count it directly on the labeled split as a single ratio. This is exact, needs no
independence assumption, and removes $\beta$ from the $Z=0$ branch entirely — which also
restores the closed-form $\beta$ M-step and removes the $\beta\to1$ degeneracy, because
$\beta$ no longer multiplies $\gamma$. The cost is that you lose the interpretable two-factor
split.

**(3) Keep the factorisation but replace $1-\beta$ with the measured $P(W=0\mid Z=0)$.**
A middle option: still two factors, still interpretable, but the first factor is counted
rather than borrowed from $\beta$.

Option (2) is what the code's `collision_frozen` mode computes in effect, and it is the arm
that scored best (6.67 vs 17.11 at budget 40). Option (1) is what is currently written in the
`.tex`. If you want the `.tex` to state the exact form and carry assumption (I) as an
explicitly-flagged approximation rather than silently, say so and I will make that edit.
