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
|
# 2026-09-17 — Yes: that is `model_wrong`. And it collapses my "three options" into two.

You remember correctly. The arm with **no $(1-\beta)$ factor at all** is `model_wrong`:

$$
\gamma_j^{\text{model}} \;=\; P\big(r_i^j\neq\hat y_i \;\big|\; r_i^j\neq y_i\big)
$$

counted over the **wider** denominator — every item where classifier $j$ is wrong, saying
nothing about the pseudo-label. It is used as $P(C=0\mid Z=0)$ **directly**. No factorisation,
no $\beta$, no $\hat\beta$, no independence assumption.

## This forces a correction to my earlier entry

I presented three options for handling the independence step. Checking them numerically,
**two of the three are the same estimator, and it is one that already exists.**

**Option 2 was $\delta_j=P(C=1\mid Z=0)$ counted as one ratio.** But

$$
\delta_j \;=\; 1-\gamma_j^{\text{model}} ,
$$

exact complements on the same denominator. Verified: on spider the two vectors sum to
$1.000000000000$ in every coordinate. **Option 2 is `model_wrong`, written in the agreement
direction instead of the disagreement direction.**

**Option 3 was $P(W{=}0\mid Z{=}0)\times\gamma^{\text{coll}}_j$, both factors measured.** But
that product is exactly what the proof said $P(C=1\mid Z=0)$ equals, so it is $\delta_j$ too.
Verified on all six cases: max deviation $1.1\times10^{-16}$.

$$
\boxed{\text{Option 2}\;=\;\text{Option 3}\;=\;\texttt{model\_wrong}}
$$

So there are not three choices. There are **two distinct estimators, reachable three ways**:

| estimator | $P(C=0\mid Z=0)$ | assumption | MAE b0 | MAE b40 |
|
# 2026-09-17 — Reviewer report on the collision-aware formulation

You asked for an honest ICLR/NeurIPS review. Here it is, written as a reviewer who has read
the `.tex` and the measurements in this repository. I have **not** done a literature search,
so treat §Novelty as the part most likely to be wrong.

**Recommendation as it currently stands: weak reject (4/10), as a *central* contribution.**
The idea is sound and the component is real. It is not, on this evidence, a paper-carrying
idea. It is, however, one rewrite away from something I would argue to accept — see §What
would flip my vote.


# 2026-09-17 — GLAD (Whitehill et al., NIPS 2009): why it got in, and how we compare

I read the actual PDF, not a summary — the automatic summariser got two facts wrong (it said
the exponent was $\alpha_i+\beta_j$ and the M-step was closed form; both are false).

**Paper:** *Whose Vote Should Count More: Optimal Integration of Labels from Labelers of
Unknown Expertise.* Whitehill, Wu, Bergsma, Movellan, Ruvolo. NIPS 22 (2009).

## What it actually says

$$
p(L_{ij}=Z_j \mid \alpha_i,\beta_j)=\frac{1}{1+e^{-\alpha_i\beta_j}},
\qquad
\log\frac{p(L_{ij}=Z_j)}{1-p(L_{ij}=Z_j)}=\alpha_i\beta_j
$$

A **bilinear** log-odds: $\alpha_i$ is labeler expertise (negative $=$ adversarial), $1/\beta_j$
is image difficulty. Verbatim from §3.1: *"Using gradient ascent, we find values of $\alpha$
and $\beta$ that locally maximize $Q$"* — conjugate gradient via libgsl. **No closed form.**
Our `.tex` section on this is correct.

Labelers are **conditionally independent given the true label**. There is no correlation
structure of any kind.

## Why it was accepted

1. **Timing.** 2009 was year zero for crowdsourcing in ML — Mechanical Turk had just become a
   research instrument. Dawid–Skene (1979) existed but sat in biostatistics. A principled
   model for MTurk labels was genuinely new *to that audience*.
2. **One memorable idea, in one equation.** Item difficulty as a second axis, and negative
   $\alpha$ for adversaries falling out for free. You can recite the model from memory.
3. **It demonstrated scale.** 1 million images, EM converging in ~10 minutes on one 2.8 GHz
   core.
4. **Clean validation.** Simulations recovering the true $\alpha,\beta$ as labelers increase,
   plus real MTurk data, against the one baseline everybody used (majority vote).
5. **The bar.** NIPS 2009 had roughly 1,100 submissions. A clean idea with a working
   demonstration was sufficient. The same paper submitted today would very likely be rejected
   for thin baselines and a single application.

## The parallel you should care about most

**GLAD has the same class of identifiability problem we found, and it got in anyway.**

$\alpha_i\beta_j$ is a product of two free parameters: rescaling $\alpha\to c\alpha$,
$\beta\to\beta/c$ leaves every prediction unchanged. That is a flat direction, exactly like
ours in $(1-\beta)\gamma_j$.

What they did about it — §3.1, verbatim: they impose *"Gaussian priors $(\mu=1,\sigma=1)$ for
$\alpha$"* and reparameterise *"$\beta=e^{\beta'}$ and imposed a Gaussian prior
$(\mu=1,\sigma=1)$ on $\beta'$"*. They **name the degeneracy and regularise it in the open**.

That is the template. Our fix — freezing the multiplier at a measured $\hat\beta$ — is the
same move and is arguably better justified, because we have labeled data to measure from
rather than a prior pulled from the air. **Reviewers do not reject a model for having a flat
direction. They reject it for not noticing.**

## Is it better than us?

Split the question.

**As a paper, for its venue and year: yes, clearly.** One claim, one equation, one baseline,
one convincing application. Our work has five entangled components and no single sentence you
would remember.

**As a model, in 2026: no.** Point by point:

| | GLAD (2009) | ours |
|
# 2026-09-17 — How the collision term removes GLAD's independence assumption

Two mechanisms, and they are different from each other. Then a measurement of how much is
actually removed, which is **not** "all of it".

## 1. What GLAD assumes

Given the true label $Z_j$, the labels of different annotators are independent:

$$
P\big(L_{1j},L_{2j},\dots,L_{mj}\mid Z_j\big)\;=\;\prod_{i=1}^{m}P\big(L_{ij}\mid Z_j\big)
$$

This is what makes the E-step a product of per-annotator likelihood ratios, and it is shared by
Dawid–Skene, GLAD and every model in that family. In words: **once you know the truth, one
annotator's mistake tells you nothing about another's.**

## 2. Why it fails for a pool of models

For human annotators it is a reasonable idealisation. For a pool of LLMs or classifiers it is
plainly false: two models fine-tuned from the same checkpoint do not merely each have an error
rate — when they fail they fail **on the same items** and produce **the same wrong answer**.
So

$$
P\big(r^1=a,\;r^2=a \mid y\neq a\big)\;\gg\;P\big(r^1=a\mid y\big)\,P\big(r^2=a\mid y\big).
$$

Under independence a chorus of ten near-clones looks like ten independent confirmations. It is
one confirmation repeated ten times, and the estimator becomes confidently wrong.

## 3. Mechanism one — the shared pseudo-label as a latent common cause

This is the part that does the work, and it is easy to miss because it looks like a change of
observable rather than a change of dependence structure.

GLAD conditions on the **true label** $y$, which is per item but enters each annotator's
likelihood separately. We instead condition on the pair $(Z_i^j,\ \hat y_i)$, where $\hat y_i$
is the **pseudo-label — one object shared by every model on that item**:

$$
P\big(C_i^1,\dots,C_i^J \mid Z_i^1,\dots,Z_i^J,\ \hat y_i\big)
=\prod_{j}P\big(C_i^j\mid Z_i^j,\ \hat y_i\big)
$$

Still a product — but **conditional on $\hat y_i$**. Now marginalise $\hat y_i$ out, which is
what the model actually claims about the observable answers:

$$
P\big(r^1,\dots,r^J\mid y\big)
=\mathbb{E}_{\hat y}\Big[\textstyle\prod_j P\big(r^j\mid Z^j,\hat y\big)\Big]
\;\neq\;\prod_j P\big(r^j\mid y\big).
$$

**An expectation of a product over a shared variable does not factorise.** That inequality is
the removal of conditional independence. Concretely, two wrong models are both pulled toward
the same $\hat y$, with strengths $\gamma_1$ and $\gamma_2$, so

$$
P\big(r^1=r^2=a,\ \text{both wrong}\big)\;\ge\;P(\hat y=a,\ \hat y\ \text{wrong})\,\gamma_1\gamma_2,
$$

which is far larger than the product of marginals whenever the $\gamma$'s are large.

$$
\boxed{\ \hat y \text{ is a latent common cause; } \gamma_j \text{ is model } j\text{'s loading on it.}\ }
$$

Structurally this is a **one-factor model**: a single shared factor per item, one loading per
model, inducing a rank-one correlation among the errors. That is the same device community-BCC
and factor-analytic annotator models use, which is worth knowing because it is also where a
reviewer will look for prior work.

## 4. Mechanism two — the $e$ matrix, for what one factor cannot express

A single shared factor only produces correlation *through the consensus*. Two models that
collide with each other but not with $\hat y$ are invisible to $\gamma$. That is what $e$ is
for:

$$
e_{jk}=P\big(r^j=r^k\ \wedge\ \text{both wrong}\big)
$$

the **full pairwise matrix**, measured directly, no factor assumption. Its excess over the
cross-group chance level discounts votes when $\hat y$ is formed. So:

$$
\boxed{
\begin{array}{ll}
\gamma \ (\text{vector},\ J) & \text{rank-one correlation, through the shared pseudo-label, inside the likelihood}\\
e \ (\text{matrix},\ J\times J) & \text{full pairwise correlation, inside the vote that builds }\hat y
\end{array}}
$$

They are complementary, which is also why the ablation showed $e$ is worth 7.16 MAE points at
budget 40 even with $\gamma$ already present.

## 5. What is *not* removed — be honest about this

We did not eliminate the assumption; **we moved it up one level.** The model still asserts

$$
C_i^j \ \perp\ C_i^k \ \Big|\ Z_i^j,\ Z_i^k,\ \hat y_i ,
$$

i.e. conditional on the truth **and the pseudo-label**, models still err independently. That is
strictly weaker than GLAD's assumption, and strictly stronger than no assumption.

## 6. How much is actually removed — measured

For every off-diagonal pair, mean absolute gap between observed and predicted:

- **(1) GLAD's assumption:** $P(j\text{ wrong},k\text{ wrong})$ vs $P(j\text{ wrong})P(k\text{ wrong})$
- **(2) our residual:** $P(\text{both agree with }\hat y\mid\text{both wrong})$ vs $\gamma_j\gamma_k$

| case | (1) GLAD gap | (2) our residual | removed | median pair sample |
|
# 2026-09-17 — Where $(1-\beta)\gamma$ comes from, step by step, and the honest verdict

**Short answer: the formula cannot be derived. It can only be *assumed*, and the assumption is
measurably false. So yes — as a derivation it is wrong. As a declared modelling assumption it
is legitimate, provided you declare it.**

Below is every step, then the same thing in raw counts so nothing is abstract.

## Step 0 — name the three events

Fix one item $i$ and one classifier $j$. Drop the subscripts. Three events:

$$
\begin{aligned}
A &= \{\,r^j \neq y\,\} && \text{classifier } j \text{ is WRONG} && (Z=0)\\
B &= \{\,\hat y = y\,\} && \text{pseudo-label is RIGHT} && (W=1)\\
G &= \{\,r^j = \hat y\,\} && \text{classifier } j \text{ AGREES with the pseudo-label} && (C=1)
\end{aligned}
$$

Definitions of the two parameters:

$$
\beta \;=\; P(B) \qquad\text{(a MARGINAL probability — remember this)}
$$
$$
\gamma_j \;=\; P\big(G \mid A \cap B^c\big) \qquad\text{(given BOTH are wrong, they coincide)}
$$

**What we want:** $P(G\mid A)$.

## Step 1 — split on $B$ (exact)

$B$ happens or it does not, so for any events whatsoever:

$$
P(G\mid A)\;=\;P(G\cap B\mid A)\;+\;P(G\cap B^c\mid A)
$$

Apply the chain rule to each piece:

$$
P(G\mid A)\;=\;\underbrace{P(B\mid A)\,P(G\mid A\cap B)}_{\text{term 1}}
\;+\;\underbrace{P(B^c\mid A)\,P(G\mid A\cap B^c)}_{\text{term 2}}
$$

No assumptions. This is just $P(X)=P(X\cap Y)+P(X\cap Y^c)$ and $P(X\cap Y)=P(Y)P(X\mid Y)$.

## Step 2 — term 1 is exactly zero (exact)

Look at $P(G\mid A\cap B)$. We are told:

- from $A$: $\ r^j \neq y$
- from $B$: $\ \hat y = y$

Substitute the second into the first: $r^j \neq y = \hat y$, therefore $r^j \neq \hat y$,
therefore $G$ is **impossible**.

$$
\boxed{P(G\mid A\cap B)=0}
$$

A wrong answer cannot equal a right answer. Term 1 disappears entirely.

## Step 3 — term 2 is $\gamma_j$ by definition (exact)

$P(G\mid A\cap B^c)$ is *literally* the definition of $\gamma_j$ from Step 0. So:

$$
\boxed{\;P(G\mid A)\;=\;P(B^c\mid A)\cdot\gamma_j\;}
$$

**This is exact. Nothing has been assumed. This is the correct formula.**

## Step 4 — the step where it breaks

Compare what we derived with what is claimed:

$$
\text{derived: } P(B^c\mid A)\cdot\gamma_j
\qquad\text{vs}\qquad
\text{claimed: } (1-\beta)\cdot\gamma_j
$$

They agree **if and only if**

$$
\underbrace{P(B^c\mid A)}_{\text{CONDITIONAL on } j \text{ being wrong}}
\;=\;
\underbrace{1-\beta \;=\; P(B^c)}_{\text{MARGINAL}}
$$

and $P(X\mid Y)=P(X)$ **is the definition of independence**. So:

$$
\boxed{(1-\beta)\gamma_j \text{ is valid} \iff \hat y\text{'s correctness is independent of whether classifier } j \text{ is correct.}}
$$

There is no derivation of this. It is an assumption, and it is the *only* assumption in the
whole argument.

## Step 5 — the same thing in raw counts, spider, classifier 0

$N=120$ labeled items.

| count | value |
|
# 2026-09-19 — $P(C\mid Z{=}0)=\gamma_j$ with $\beta$ kept for $Z{=}1$: never tried before, and it wins at budget 0

Your model, stated exactly:

$$
\boxed{
\begin{array}{ll}
P(C_i^j=1\mid Z_i^j=1)=\beta, & P(C_i^j=0\mid Z_i^j=1)=1-\beta\\[2pt]
P(C_i^j=1\mid Z_i^j=0)=\gamma_j, & P(C_i^j=0\mid Z_i^j=0)=1-\gamma_j
\end{array}}
$$

$\beta$ is still a free parameter and still fitted; it simply no longer appears in the $Z=0$
branch. Implemented as `gamma_mode="collision_nobeta"`. **This arm had not been run before.**

## Two consequences of the form itself

**1. The $\beta$ M-step returns to closed form.** With $\beta$ absent from the $Z=0$ branch,
$Q_\beta$ falls back into the span $\mathcal{S}$ and the maximiser is a ratio of counts. No
bounded numerical search, and the $\beta\to1$ degeneracy cannot occur, because nothing
multiplies $\gamma$.

**2. It is the $\beta=0$ special case of $(1-\beta)\gamma_j$** — it asserts the pseudo-label is
*always* wrong. Since $\gamma_j$ is conditional on both being wrong, using it for the wider
event "the classifier is wrong" **overstates** agreement by exactly
$1/P(\hat y\text{ wrong}\mid\text{model wrong})$. On spider that factor is 1.21–1.40.

So the three arms bracket the truth from both sides:

| | $P(C{=}1\mid Z{=}0)$, spider model 0 | vs truth 0.826 |
|
# 2026-09-19 — FULL RESULTS: every arm, three modalities, three budgets

MAE in accuracy points, lower is better. Each modality column is the mean of its two
cases: text2sql $=$ {spider, bird}, vision $=$ {mnist$\to$usps, mnist$\to$svhn},
graph $=$ {AC, DA}. `ALL` is the unweighted mean of the three modality columns. Every arm
runs the identical pipeline — same $e$ vote discount, same $A_\mu$ selector, same
`OracleExpert`, same warm-start, same $(\text{old}+\text{temp})/2$ refresh — and differs
only in the $Z=0$ branch, except the last row which additionally removes the $e$ discount.
Rows sorted best-first by `ALL`.

## Budget 0 — no expert

| arm | text2sql | vision | graph | **ALL** |
|---|---:|---:|---:|---:|
| **`collision_nobeta`** | **3.61** | 33.00 | 12.15 | **16.25** |
| `pairwise` | 3.67 | 35.32 | **12.13** | 17.04 |
| `pairwise_calibrated` | 3.67 | 35.32 | **12.13** | 17.04 |
| `model_wrong` | 3.87 | 36.72 | 14.19 | 18.26 |
| frozen, no $e$ discount | 7.85 | 34.47 | 14.11 | 18.81 |
| `collision_frozen` | 7.91 | 34.47 | 14.06 | 18.81 |
| `collision` (free $\beta$) | 14.01 | **33.49** | 16.70 | 21.40 |

## Budget 10

| arm | text2sql | vision | graph | **ALL** |
|---|---:|---:|---:|---:|
| **`collision_frozen`** | 6.58 | **21.56** | 13.83 | **13.99** |
| `model_wrong` | 5.42 | 27.47 | **12.60** | 15.16 |
| `collision_nobeta` | **4.95** | 23.80 | 18.98 | 15.91 |
| `pairwise_calibrated` | 5.03 | 30.52 | 16.16 | 17.23 |
| `pairwise` | 6.33 | 34.26 | 14.55 | 18.38 |
| `collision` (free $\beta$) | 13.27 | 27.82 | 15.57 | 18.88 |

## Budget 40

| arm | text2sql | vision | graph | **ALL** |
|---|---:|---:|---:|---:|
| **`collision_frozen`** | 2.95 | **7.36** | **9.70** | **6.67** |
| `model_wrong` | 3.13 | 9.62 | 11.80 | 8.18 |
| `pairwise_calibrated` | **2.26** | 26.72 | 10.12 | 13.03 |
| frozen, no $e$ discount | 3.20 | 28.18 | 10.12 | 13.83 |
| `pairwise` | 3.02 | 32.31 | 9.96 | 15.10 |
| `collision_nobeta` | 4.46 | 24.37 | 16.70 | 15.18 |
| `collision` (free $\beta$) | 10.27 | 25.84 | 15.24 | 17.11 |

## What the three tables say together

**1. The winner changes with the budget.** No arm is best everywhere.

$$
\boxed{
\begin{array}{lll}
\text{budget }0 & \texttt{collision\_nobeta} & 16.25\\
\text{budget }10 & \texttt{collision\_frozen} & 13.99\\
\text{budget }40 & \texttt{collision\_frozen} & \mathbf{6.67}
\end{array}}
$$

`collision_nobeta` leads at 0 and finishes **second-to-last** at 40. `collision_frozen` is
*sixth of seven* at budget 0 and first by a wide margin at 40. Reporting only one budget would
invert the conclusion.

**2. Vision decides the ranking; text2sql does not.** At budget 40 the text2sql column spans
2.26–10.27 — under 8 points — while vision spans 7.36–28.18, nearly 21. The `ALL` ordering is
essentially the vision ordering. `pairwise_calibrated` is the **best text2sql arm at budget 40
(2.26)** and still finishes fifth overall, sunk entirely by vision's 26.72.

**3. The pairwise arms are identical without a judge.** `pairwise` and `pairwise_calibrated`
agree to the last digit at budget 0 (3.67 / 35.32 / 12.13) and separate only once validation
starts — as they must, since the calibration is defined on validated items.

**4. Graph is where the expert helps least.** Best graph numbers by budget: 12.13 → 12.60 →
9.70. Forty oracle labels buy about 2.4 points there, against roughly 26 on vision
(33.00 → 7.36) and 1.4 on text2sql, which is already near its floor at budget 0.

**5. The free-$\beta$ collision arm is last at every budget.** 21.40, 18.88, 17.11. It is the
only arm never to win a single modality column at any budget, apart from vision at budget 0
(33.49) where it is a hair ahead of `collision_nobeta`'s 33.00 — and that one cell is inside
the noise of a case where every arm is around 33–37.

**6. The $e$ discount is worth nothing without a judge and 7 points with one.** Comparing the
two `frozen` rows: 18.81 vs 18.81 at budget 0 — identical to two decimals — and 6.67 vs 13.83
at budget 40, all of it from vision (7.36 vs 28.18).

## The defensible claim

All seven arms share one likelihood and differ only in what occupies the $Z=0$ branch. Written
as a multiplier $m_j$ on the collision rate, $P(C=1\mid Z=0)=m_j\gamma_j$:

| arm | $m_j$ | best at |
|---|---|---|
| `collision_nobeta` | $1$ | budget 0 |
| `model_wrong` | $P(\hat y\text{ wrong}\mid Z{=}0)$, exact | never (2nd at 10 and 40) |
| `collision_frozen` | $1-\hat\beta$, measured | budgets 10 and 40 |
| `collision` | $1-\beta$, fitted | never |

The exact multiplier is never the best choice, and the fitted one is always the worst. That
is the result worth reporting — not a claim that any single parameterisation is correct.

---|---:|---|
| `collision_nobeta` (multiplier $=1$) | 0.952 | **too high** |
| `model_wrong` (exact) | 0.826 | exact |
| `collision_frozen` (multiplier $=1-\hat\beta$) | 0.183 | **far too low** |

## Results — all six cases, MAE in accuracy points

| case | \multicolumn{3}{c}{nobeta} | | | model_wrong | | | frozen | | | collision | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | b0 | b10 | b40 | b0 | b10 | b40 | b0 | b10 | b40 | b0 | b10 | b40 |
| spider | **3.57** | **2.29** | 4.39 | 3.61 | 4.44 | 2.66 | 9.18 | 3.37 | **2.29** | 10.66 | 10.59 | 7.10 |
| bird | **3.64** | 7.62 | 4.53 | 4.14 | 6.39 | **3.61** | 6.63 | 9.78 | **3.61** | 17.36 | 15.95 | 13.43 |
| mnist→usps | 32.47 | 28.17 | 21.49 | 16.39 | 13.29 | 6.15 | 10.91 | 4.24 | **1.68** | **9.44** | **2.63** | 2.28 |
| **mnist→svhn** | **33.53** | **19.42** | 27.25 | 57.05 | 41.64 | 13.09 | 58.03 | 38.89 | **13.03** | 57.54 | 53.00 | 49.39 |
| graph/AC | 10.92 | 19.06 | 18.31 | **8.58** | 13.39 | **8.22** | 9.12 | 13.51 | 8.85 | 11.63 | 10.40 | 10.53 |
| graph/DA | **13.38** | 18.90 | 15.09 | 19.79 | **11.81** | 15.37 | 19.00 | 14.15 | **10.55** | 21.77 | 20.74 | 19.95 |
| **MEAN** | **16.25** | 15.91 | 15.18 | 18.26 | 15.16 | 8.18 | 18.81 | **13.99** | **6.67** | 21.40 | 18.88 | 17.11 |

## What it shows

**It is the best arm at budget 0 — the new best, 16.25.** It beats the exact form (18.26) and
the measured-multiplier form (18.81), and wins outright on 4 of 6 cases. On MNIST→SVHN it is
better by **24 accuracy points** (33.53 against ~57–58 for all three others), the largest
single-case margin anywhere in this comparison, and SVHN is the hardest shift in the study.

**It does not improve with the expert.** 16.25 → 15.91 → 15.18 across budgets 0, 10, 40 — a
total gain of **1.07 points** where `collision_frozen` gains 12.14. On SVHN it goes
33.53 → 19.42 → **27.25**, worse at 40 than at 10.

$$
\boxed{
\begin{array}{ll}
\text{best without a judge} & \texttt{collision\_nobeta}\ (16.25)\\
\text{best with a judge} & \texttt{collision\_frozen}\ (6.67)
\end{array}}
$$

## Why, most likely

The arm asserts the pseudo-label is always wrong. That is badly false on easy data — on
MNIST→USPS the vote is right about 99.7% of the time, and the arm is 32.47 there, its worst
relative showing. It is roughly *right* on MNIST→SVHN, where the vote genuinely is mostly
wrong, and that is exactly where it wins by 24 points.

So it is not a better model. It is a model whose single wrong assumption happens to point the
right way under severe shift, which is precisely the regime where the prior is catastrophic and
every other arm is anchored to a badly wrong number. It buys robustness at budget 0 by
refusing to trust the consensus at all.

The flat response to the expert follows from the same thing: $\gamma_j$ is the only quantity in
the $Z=0$ branch and the arm has no way to revise its belief that the consensus is worthless,
so revealing true labels cannot talk it out of that.

## Recommendation

Worth reporting, **not** worth making the default. Two honest uses:

- as the **budget-0 configuration** — if no expert is available, it is the best of the four;
- as evidence for the paper's real story: the three arms differ only in the multiplier on
  $\gamma_j$ (1, exact, or $1-\hat\beta$), and which one wins depends entirely on the budget
  and the severity of shift. That is a cleaner and more defensible finding than claiming any
  one of them is correct.

---|---:|
| $\lvert A\rvert$ — classifier 0 wrong | 23 |
| $\lvert B^c\rvert$ — pseudo-label wrong | 22 |
| $\lvert A\cap B^c\rvert$ — both wrong | 19 |
| $\lvert A\cap B^c\cap G\rvert$ — both wrong **and same wrong answer** | 19 |
| $\lvert A\cap G\rvert$ — classifier wrong and agrees with $\hat y$ | 19 |

Now the two factors:

$$
1-\beta=\frac{22}{120}=0.1833
\qquad\text{but}\qquad
P(B^c\mid A)=\frac{19}{23}=0.8261
$$

**4.51 times larger.** And $\gamma_0=\dfrac{19}{19}=1.0000$.

| | value | error |
|---|---:|---:|
| **truth**, $\lvert A\cap G\rvert/\lvert A\rvert=19/23$ | **0.826087** | — |
| **exact form** $P(B^c\mid A)\gamma_0 = 0.8261\times1.0000$ | 0.826087 | $0$ |
| **claimed form** $(1-\beta)\gamma_0 = 0.1833\times1.0000$ | 0.183333 | **0.642754** |

The claimed formula is off **by a factor of 4.51** on this one classifier.

## Step 6 — why it fails, and why it is not fixable by being careful

Look at the counts again. Of the 22 items where the pseudo-label is wrong, **19 are items where
classifier 0 is also wrong.** That is not coincidence — **the pseudo-label is a vote over these
same classifiers.** When classifier 0 fails, the item is hard, so the other classifiers tend to
fail too, so the vote fails.

$$
\boxed{A \text{ and } B^c \text{ are positively associated BY CONSTRUCTION, so independence can never hold.}}
$$

This is why it is not a small-sample issue or a dataset quirk. It is structural, and it is in
the same direction on all six datasets (ratios 1.6× to 135×).

## Step 7 — so were we wrong?

Two different questions, two different answers.

**As a derivation: yes, wrong.** You cannot get $(1-\beta)\gamma_j$ from the definitions. The
step $P(B^c\mid A)\to P(B^c)$ is an unjustified substitution of a marginal for a conditional.

**As a model: not wrong, but incomplete.** Every model makes assumptions. Writing
$P(C=1\mid Z=0)=(1-\beta)\gamma_j$ is a legitimate *modelling choice* provided you say
"we assume $\hat y \perp Z^j$". What is not legitimate is presenting it as though it followed
from the definitions, which is what an unflagged $(1-\beta)\gamma_j$ does.

And it is worth knowing the cost is not only theoretical. In the version where $\beta$ is a
free parameter, this same factorisation is what lets $\beta$ drift to 1 and annihilate
$\gamma$ — the arm that scored 17.11, worst of everything tested.

## Step 8 — what to do

$$
\boxed{
\begin{array}{lll}
\text{(a) keep it, declare it} & (1-\beta)\gamma_j \text{ + one sentence stating the assumption} & \text{honest, still approximate}\\
\text{(b) use the exact form} & P(B^c\mid A)\cdot\gamma_j,\ \text{first factor COUNTED} & \text{exact, no assumption}\\
\end{array}}
$$

Option (b) costs you one extra count on the labeled split — the same items you already scan to
get $\gamma_j$, just a different denominator. And recall from an earlier entry that (b) is
numerically identical to `model_wrong`, which is already implemented and measured.

My recommendation is (b) for the model and (a)-with-a-sentence only if you want to keep the
two-factor reading for exposition.

---|---:|---:|---:|---:|
| text2sql/spider | 0.1198 | 0.0454 | **62%** | 19 |
| text2sql/bird | 0.1535 | 0.0703 | **54%** | 57 |
| graph/AC | 0.0668 | 0.0296 | **56%** | 205 |
| graph/DA | 0.0734 | 0.0247 | **66%** | 129 |
| vision/mnist→usps | 0.0028 | 0.0725 | — | **4** |
| vision/mnist→svhn | 0.0027 | 0.0125 | — | **4** |

**On the four cases where there is dependence to remove, we remove 54–66% of it.** Not all,
consistent with §5 — the residual is the correlation a single shared factor cannot express.

**The two vision rows are not evidence of anything and I am not going to read them as such.**
The labeled split there is MNIST, where the models are 99.2% accurate, so joint errors are rare
(GLAD's gap is already 0.003 — there is nothing to remove) and the conditioning set has a
**median of 4 items per pair**. Those residual numbers are sampling noise.

## 7. What this is worth saying in the paper

The defensible claim is narrow and true:

> *GLAD and Dawid–Skene assume annotators err independently given the true label. For pools of
> models with shared provenance this fails: we measure joint-error dependence of 0.07–0.15
> above the independent prediction. Routing every model's agreement through a shared latent
> pseudo-label with a per-model loading removes 54–66% of it, and an explicit pairwise
> correlation matrix in the consensus handles part of the remainder.*

That is a claim with a number attached, a stated mechanism, and an honest residual. It is much
stronger than "we relax conditional independence", which every reviewer has read a hundred
times.

---|---|---|
| annotator correlation | **none** — conditional independence assumed | the whole point ($\gamma$, and $e$ pairwise) |
| item difficulty | **yes**, $\beta_j$ | no — we have no per-item parameter |
| M-step | gradient ascent, no closed form | closed form for $\alpha$; for $\beta$ too when the multiplier is frozen |
| nuisance parameters | all fitted | measured on a labeled split |
| known labels | "clamped" via the prior | clamped, **and actively chosen** |
| choosing what to label | **future work** | implemented and measured ($A_\mu$) |

Two of those deserve emphasis:

**We relax precisely what GLAD assumes away.** GLAD's conditional independence is the
assumption the collision $\gamma$ exists to remove. That is a legitimate lineage to claim.

**GLAD names our outer loop as future work.** From §3.2, verbatim: *"This would allow using
the algorithm in an active manner to choose in real-time which images should be labeled next
so as to minimize the uncertainty about the image labels."* You implemented that, with an
acquisition function and measurements. A 2009 NIPS paper's closing wish is your section 9.

## The concrete gap this exposes

**GLAD is discussed in your `.tex` but has never been run.** `baselines/` contains
`dawid_skene`, `majority`, `independent`, `agreement_line`, `llm_judge` — **no GLAD**. A
reviewer who sees a whole section arguing against GLAD's M-step, with no GLAD row in any
table, will ask why. It is a couple of hundred lines: E-step is the same Bayes rule, M-step is
`scipy.optimize.minimize` on $(\alpha,\beta)$ with the two Gaussian priors.

I would implement it and put it in the main table. If we beat it, that is the baseline
comparison the review report said was missing. If we do not beat it on some modality, better
to know now.

## What to take from this

GLAD succeeded on **clarity and scope discipline**, not technical depth — it is a simpler
model than ours and has a known degeneracy it handles with priors. The lesson is not "make the
model more sophisticated". It is:

- one claim a reviewer can repeat;
- name your degeneracy and fix it in the open, as they did;
- compare against the obvious baselines, including this one;
- show it works at a scale someone would actually use.

---

## What is genuinely good

**1. The problem is real and the fix is correct.** The binary reduction $\gamma=1$ assumes two
wrong classifiers never coincide. That is obviously false for models sharing a checkpoint, and
the measurements show it costs a lot: the binary EM is worse than the collision model on
**every one of 15 dataset/shift pairs**, often by 5–10 accuracy points. Identifying a wrong
independence assumption and fixing it with one measured vector is a clean contribution.

**2. It is cheap.** One length-$J$ vector from a labeled split. No architecture, no extra
inference, no tuning. Reviewers like contributions with this cost profile.

**3. The derivation is honest about what survives.** Showing that $\alpha$'s closed form is
untouched while $\beta$'s is destroyed, and characterising exactly why (the term leaves the
span $\mathcal S$), is the right level of rigour.

---

## The three objections I would raise, in order of severity

### Objection 1 — the model as stated is weakly identified. This is the serious one.

The paper's own parameterisation, $P(C=1\mid Z=0)=(1-\beta)\gamma_j$ with $\beta$ free, has a
near-flat direction. The observable per classifier is its agreement rate
$a_j=\alpha_j\beta+(1-\alpha_j)(1-\beta)\gamma_j$, so

$$
\frac{\partial a_j}{\partial\beta}=\alpha_j-(1-\alpha_j)\gamma_j ,
$$

a **difference of positive terms**, which can vanish or change sign. Measured on BIRD it
averages $+0.039$ and is sign-indefinite across models. The consequence is not hypothetical:
the profile likelihood is **monotone increasing to $\beta=1$ on 3 of 4 cases**, so the MLE sits
on the boundary, where $(1-\beta)\gamma_j\to0$ and **the collision statistic the paper is about
is multiplied by zero.**

A reviewer will state this bluntly: *the paper introduces a parameter, and at the maximum
likelihood estimate that parameter deletes the paper's own contribution from the model.* On
the reported numbers the free-$\beta$ collision arm is the **worst** of everything tested
(17.11 MAE at budget 40, against 6.67 for the same formulation with the multiplier frozen).

This alone is enough for a reject if the paper does not address it.

### Objection 2 — a factorisation the authors can show is false, kept anyway

$(1-\beta)\gamma_j$ requires $P(\hat y\neq y\mid Z=0)=P(\hat y\neq y)$. Measured on the six
labeled splits, the two differ by **1.6× to 135×**, always in the same direction. And the
direction is structural, not incidental: the pseudo-label is a vote over the same classifiers,
so conditioning on one being wrong selects hard items, whose votes are also wrong. The
assumption is guaranteed to fail by construction.

The reviewer's question is sharp: *the exact quantity is one count away — why approximate it?*
The paper needs an answer, and "it preserves an interpretable two-factor reading" is a weak one
when the interpretation is numerically wrong by two orders of magnitude.

### Objection 3 — the titular contribution is the smallest effect in the system

This is the framing problem, and it is the one I would press hardest in discussion. From the
repository's own ablations, at budget 40:

| component | MAE improvement |
|---|---:|
| the expert-validation outer loop | **12.14** |
| the correlated-error matrix $e$ (vote discount) | **7.16** |
| free $\beta$ $\to$ measured $\hat\beta$ | **10.44** |
| **the choice of $\gamma$ parameterisation** | **1.51** |

The collision $\gamma$ — the thing the formulation is named for — accounts for roughly
**1.5 accuracy points out of a ~12-point improvement**, and is dominated by the outer loop, by
a *different* correlated-error object ($e$), and by a single implementation decision about
$\beta$. A paper titled around the collision model is claiming credit for the smallest term.

---

## Novelty — my biggest uncertainty

Correlated and dependent annotators are a well-worked area: Dawid–Skene with dependence,
Bayesian Classifier Combination and its dependent/community variants, and the
colluding-annotator literature all model "annotators that err together". A reviewer will ask
what $\gamma_j$ adds beyond a community or low-rank correlation structure, and *why the
pairwise matrix $e$ is not simply the known object*.

I cannot answer that from this repository. **If a reviewer finds a close prior formulation,
objection 3 becomes fatal**, because the remaining contribution is engineering. This should be
checked before submission, not after.

---

## Is it "vague"?

**No — the opposite.** The derivation is more explicit than most submissions: every step
stated, the closed forms characterised, the boundary behaviour derivable. Vagueness is not the
problem.

The problem is that the formulation is **precisely specified and empirically dominated by its
own components.** That is a worse position to be in than vague, because a reviewer can verify
it from your own tables.

---

## What would flip my vote

**Lead with the identifiability result.** It is the most interesting thing here and I have not
seen it stated for this model class:

> *The standard collision-aware parameterisation is weakly identified in $\beta$. Its
> profile likelihood is monotone to the boundary on real data; at that boundary the collision
> statistic is annihilated and expert feedback cannot reach the model, because the sensitivity
> of the likelihood to the collision rate is exactly $1-\beta$. Freezing the multiplier at a
> measured value recovers 10.4 MAE points.*

That is a crisp negative-plus-fix result with theory, a mechanism, and a measured remedy —
the shape of paper that gets accepted. The collision model becomes the *setting*, not the
claim, which also dissolves objection 3.

**Then:**
- run real crowdsourcing baselines (DS variants, BCC, community BCC, GLAD), not only internal
  ablations and PoolEval;
- report a real LLM judge, not only the oracle expert — every headline number here uses a
  perfect expert;
- more than two datasets per modality;
- state plainly that no arm dominates: `pairwise` wins text-to-SQL (1.68/2.84), the frozen
  collision arm wins vision and graph. A paper that reports this honestly is stronger than one
  that hides it behind a mean.

---

## Bottom line

The collision formulation is a **correct and useful component with a real identifiability
defect**, not a headline contribution. As the centre of a paper: weak reject. As the setting
for an identifiability result, with the system honestly ablated: I would argue to accept.

|---|---|---|---|---|
| `collision` | $1-(1-\beta)\gamma^{\text{coll}}$, $\beta$ **fitted** | independence | 21.40 | 17.11 |
| `collision_frozen` | $1-(1-\hat\beta)\gamma^{\text{coll}}$, $\hat\beta$ **measured** | independence | 18.81 | **6.67** |
| `model_wrong` | $\gamma^{\text{model}}$ counted directly | **none** | **18.26** | 8.18 |

## The uncomfortable result

**The exact, assumption-free form loses at budget 40.** `model_wrong` needs no independence
assumption and is therefore the theoretically correct one — and it scores 8.18 against
`collision_frozen`'s 6.67.

But look at budget 0, where the ordering **reverses**: `model_wrong` is best at 18.26,
`collision_frozen` worst of the two at 18.81.

$$
\boxed{
\begin{array}{lll}
\text{no judge} & \text{the exact form wins} & 18.26 \text{ vs } 18.81\\
\text{with judge} & \text{the factorised form wins} & 6.67 \text{ vs } 8.18
\end{array}}
$$

## Why, most likely

It is not transfer. I measured how far each statistic moves from the labeled split to the
target, and $\gamma^{\text{model}}$ transfers as well or better on 4 of 6 cases:

| case | $\gamma^{\text{coll}}$ shift | $\gamma^{\text{model}}$ shift |
|---|---:|---:|
| spider | 0.0745 | 0.0767 |
| bird | 0.0729 | **0.0625** |
| mnist→usps | 0.2859 | **0.2746** |
| mnist→svhn | 0.4371 | **0.2364** |
| graph/AC | **0.0707** | 0.0942 |
| graph/DA | **0.0842** | 0.1106 |

The likelier explanation is **how many things the expert can repair**. In `model_wrong` the
$Z=0$ branch is a single number $\gamma_j^{\text{model}}$, so expert validation has **one**
channel to correct it through. In `collision_frozen` the branch is a product of **two**
separately-refreshed quantities, $\hat\beta$ and $\gamma^{\text{coll}}_j$, so the expert has
two channels. With no expert that extra structure is pure added approximation error, which is
why the exact form wins at budget 0. With 40 expert labels the extra repair surface more than
pays for the approximation, which is why it wins at budget 40.

That is a hypothesis consistent with the budget-0/budget-40 reversal, not something I have
isolated. It would be testable by refreshing only one of the two factors.

## What this means for the `.tex`

The earlier "pick 1, 2 or 3" question was badly posed and I withdraw it. The real choice is:

- **the document as it stands** describes `collision`, the free-$\beta$ version — the arm that
  scores 17.11, worst of the three;
- **one sentence** turns it into `collision_frozen` — the best arm at 6.67 — by saying the
  $Z=0$ branch uses the measured $\hat\beta$ rather than the fitted $\beta$;
- **a different sentence** turns it into `model_wrong` — exact, no assumption, best without a
  judge at 18.26 but worse with one at 8.18.

All three keep your $\gamma$ as a per-model vector. Tell me which and it is a small edit.

---|---|---|
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
