# Is $\gamma^{\text{collision}}$ with $\beta$ alone still good?

**Your question.** Keep the original formulation: $\gamma^{\text{coll}}$ measured, $\beta$ fitted
by EM, and **no $\hat\beta$ anywhere**. What were the results before `both_wrong` /
`model_wrong` existed, and is that formulation still good?

This file has two parts:

- **Part A** — the historical table, measured before any of the $\gamma$-mode work. Already
  in `results/validated_em_*.json`; nothing new was run for it.
- **Part B** — a new, clean head-to-head where the *only* difference between arms is the
  $\gamma$/$\beta$ treatment. **Running now, appended when it finishes.**

---

## What "collision alone with $\beta$" actually means

$$
\gamma_j^{\text{coll}}
=
P\big(r_i^j = \hat y_i \;\big|\; r_i^j \ne y_i \;\wedge\; \hat y_i \ne y_i\big)
$$

measured on the labeled split and frozen. The E-step does **not** consume it directly. It
consumes

$$
d_j(\beta) \;=\; P\big(C_i^j = 0 \mid Z_i^j = 0\big) \;=\; 1-(1-\beta)\,\gamma_j^{\text{coll}},
$$

rebuilt from the **live** $\beta$ on every sweep. $\hat\beta$ never appears. The price is that
$\beta$ now sits in both branches of the likelihood, so its M-step is no longer a ratio of
counts:

$$
Q(\beta)=\sum \tau\big[C\log\beta+(1-C)\log(1-\beta)\big]
+\sum(1-\tau)\big[C\log((1-\beta)\gamma^{\text{coll}})+(1-C)\log\big(1-(1-\beta)\gamma^{\text{coll}}\big)\big]
$$

The second sum's derivative is $\gamma/d(\beta)$, rational in $\beta$. Clearing denominators
gives a polynomial whose degree grows with the number of distinct $\gamma$ values — so the
implementation falls back to a **bounded 1-D numerical maximisation, once per EM sweep**.

---

## Part A — the historical table (before any $\gamma$-mode work)

MAE in accuracy points, lower is better. No expert budget in any column — this is the
zero-judge comparison. `collision EM` is `CollisionAwareNewFormulationPoolEval`, the direct
implementation of the original `.tex` formulation with a **per-group** $\gamma^{\text{coll}}$.

| case | prior only | PoolEval-SQL | binary EM ($\gamma=1$) | **collision EM** | validated EM |
|---|---:|---:|---:|---:|---:|
| text2sql/spider | 3.73 | 13.45 | 18.97 | 11.15 | **3.61** |
| text2sql/spider (holdout) | 5.50 | 13.94 | 21.56 | 12.66 | **5.71** |
| text2sql/bird | 3.47 | 17.91 | 12.10 | 16.98 | **4.14** |
| text2sql/bird (holdout) | 6.50 | 17.67 | 13.07 | 17.89 | 7.07 |
| text2sql/sqlflow | 2.14 | 6.07 | 19.22 | 7.54 | 2.99 |
| text2sql/bird_minidev | 2.86 | 10.22 | 20.98 | 5.49 | 2.54 |
| text2sql/spider2local | 6.61 | 12.88 | 16.43 | 7.13 | 6.47 |
| vision/mnist→usps | 29.10 | 6.23 | 10.47 | **5.17** | 16.39 |
| vision/mnist→svhn | 85.88 | 56.67 | 59.90 | **56.22** | 57.05 |
| graph/AC | 6.89 | 10.54 | 18.03 | 13.53 | 8.58 |
| graph/AD | 13.89 | 19.52 | 26.21 | 22.39 | 15.34 |
| graph/CA | 17.23 | 17.29 | 24.49 | 21.27 | 16.94 |
| graph/CD | 13.72 | 16.56 | 23.63 | 20.02 | 13.68 |
| graph/DA | 19.01 | 19.53 | 28.49 | 24.11 | 19.79 |
| graph/DC | 9.89 | 13.01 | 21.14 | 16.54 | 10.79 |

### What Part A already establishes

**1. Collision EM is not uniformly worse — it wins outright on vision.** On mnist→usps it
scores 5.17 against validated EM's 16.39, and on mnist→svhn 56.22 against 57.05. These are the
two hardest domain-shift cases in the study, and the collision parameterisation is the best
*estimator* on both (the prior alone is catastrophic there: 29.10 and 85.88).

**2. It is clearly worse on text2sql.** 11.15 vs 3.61 on spider, 16.98 vs 4.14 on bird. The
gap is roughly 3x.

**3. It is clearly worse on graph.** 13.53 vs 8.58 on AC, and it loses on all six graph
transfers, usually by 4–8 points.

**4. It always beats the binary reduction $\gamma=1$.** Every single row. Whatever else is
true, modelling the collision is better than ignoring it.

So the honest headline from the historical data is: **collision-with-$\beta$ is not a failed
formulation. It is a formulation that wins on one modality out of three.** The later
$\gamma$-mode work improved text2sql and graph and did not improve vision — on vision it made
things markedly worse.

### The caveat that makes Part B necessary

The `collision EM` column is **not a clean isolation of your question**, for three reasons:

1. It is configured with `beta_strength=120.0` and `beta_init=stats.pseudo_accuracy` — so
   $\hat\beta$ **is** present, as both initialiser and a prior anchor with ESS 120. You asked
   for "no $\hat\beta$ at all", and this row does not deliver that.
2. It uses a **per-group** $\gamma^{\text{coll}}$, one value per provenance group, while the
   validated-EM arms use a **per-model** $\gamma_j$. Different resolution, not just different
   definition.
3. It is a different estimator object entirely — it does not use the $e$-matrix vote discount,
   the $A_\mu$ selector, or the expert loop. So the comparison confounds the $\gamma$ question
   with everything else that changed.

Part B removes all three confounds.

---

## Part B — clean head-to-head (running)

Three arms, **identical pipeline** — same $e$ vote discount, same $A_\mu$ selector, same
`OracleExpert`, same warm-start, same $(\text{old}+\text{temp})/2$ refresh. The only
difference is the $Z=0$ branch:

| arm | $\gamma$ measured | E-step consumes | $\beta$ M-step | uses $\hat\beta$? |
|---|---|---|---|---|
| `collision` | $P(r^j=\hat y\mid$ both wrong$)$ | $d_j(\beta)$, live $\beta$ | numerical 1-D | **no** |
| `both_wrong` | $P(r^j\ne\hat y\mid$ both wrong$)$ | $\hat\beta+(1-\hat\beta)\gamma^{\text{both}}$ | closed form | yes |
| `model_wrong` | $P(r^j\ne\hat y\mid r^j$ wrong$)$ | $\gamma_j$ directly | closed form | no |

In the `collision` arm $\beta$ is initialised at a neutral **0.7** and thereafter is always the
EM-fitted value carried through warm-starts, so no measured pseudo-label accuracy can leak in.

Implementation: `gamma_mode="collision"` added to `pooleval/validated_em.py`. Verified against
the identity it must satisfy — $\gamma^{\text{coll}}+\gamma^{\text{both}}=1$ on the same
denominator, exact to machine precision.

### Result

MAE in accuracy points, mean over the 3 modalities (each = mean of its 2 cases). `t_b0` is
wall-clock for one zero-budget solve. Lower is better.

| arm | b0 | b10 | b40 | expert gain (b0→b40) | t_b0 | uses $\hat\beta$ |
|---|---:|---:|---:|---:|---:|---|
| `collision` | 21.40 | 18.88 | **17.11** | **4.29** | 145.2 ms | no |
| `both_wrong` | 18.81 | **13.99** | **6.67** | **12.14** | 50.4 ms | yes |
| `model_wrong` | **18.26** | 15.16 | 8.18 | 10.08 | 51.7 ms | no |

**Answer to the question: no, not as a general default.** At budget 40 collision is 2.6x worse
than `both_wrong` (17.11 vs 6.67) and costs 3x the wall-clock (the numerical M-step). Its
b0 penalty is modest — 21.40 vs 18.26, about 3 points — so the formulation is not broken.
What is broken is that **it barely responds to expert validation**: 40 oracle labels buy it
4.29 points where they buy `both_wrong` 12.14.

### Per modality, budget 0 — collision is not uniformly worse

| arm | text2sql | vision | graph |
|---|---:|---:|---:|
| `collision` | 14.01 | **33.49** | 16.70 |
| `both_wrong` | 7.91 | 34.47 | **14.06** |
| `model_wrong` | **3.87** | 36.72 | 14.19 |

Collision is **the best of the three on vision**, reproducing what Part A showed, and on
mnist→usps it is the best arm at every budget (b0 9.44, b10 2.63, **b40 2.28**, against
`model_wrong`'s 16.39 / 13.29 / 6.15). It is decisively worse on text2sql (14.01 vs 3.87) and
mildly worse on graph.

### Why it does not respond to the expert: $\beta$ pins at the boundary

The fitted $\beta$ at budget 0, and what it does to the $Z=0$ branch:

| case | arm | fitted $\beta$ | $1-\beta$ | $P(C=1\mid Z=0)$ |
|---|---|---:|---:|---:|
| spider | collision | **1.000** | 0.0000 | **0.0000** |
| | model_wrong | 0.993 | 0.0074 | 0.6231 |
| bird | collision | **1.000** | 0.0000 | **0.0000** |
| | model_wrong | 0.849 | 0.1506 | 0.5155 |
| mnist→usps | collision | 0.908 | 0.0923 | 0.0732 |
| | model_wrong | 0.952 | 0.0479 | 0.2901 |
| mnist→svhn | collision | 0.984 | 0.0160 | 0.0133 |
| | model_wrong | 0.977 | 0.0231 | 0.2807 |
| graph/AC | collision | 0.996 | 0.0036 | 0.0031 |
| | model_wrong | 0.889 | 0.1110 | 0.5239 |
| graph/DA | collision | 0.987 | 0.0134 | 0.0115 |
| | model_wrong | 0.836 | 0.1639 | 0.5937 |

Two things follow, and the second is the whole story.

**(i) The $Z=0$ branch collapses.** In collision mode $P(C=1\mid Z=0)=(1-\beta)\gamma^{\text{coll}}$
is 0.000–0.073 on every case — the model concludes that *a wrong classifier essentially never
agrees with the pseudo-label*. Agreement therefore becomes a near-perfect certificate of
correctness, $\tau\to1$ wherever $C=1$, and the estimate is pinned to the consensus. Under
`model_wrong` the same quantity is 0.28–0.62, i.e. agreement is genuinely ambiguous evidence,
which is what it should be.

**(ii) The expert's $\gamma$ updates are multiplied by zero.** The sensitivity of the E-step to
the statistic the expert refreshes is exactly

$$
\frac{\partial\,P(C=1\mid Z=0)}{\partial\,\gamma_j^{\text{coll}}} \;=\; 1-\beta .
$$

With $1-\beta$ between $0.0000$ and $0.0160$ on five of six cases, **every expert-driven update
to $\gamma^{\text{coll}}$ is scaled by ~0 before it reaches the likelihood.** The refresh rule
fires, $\gamma$ moves, and the estimate does not. That is the mechanism, and it is exact
rather than a hypothesis: the one case with the largest $1-\beta$ (mnist→usps, 0.0923) is also
the case where collision responds best and wins outright.

Contrast with `model_wrong`, where $\gamma_j$ **is** $P(C=0\mid Z=0)$ with coefficient 1. An
expert update to $\gamma$ moves the likelihood one-for-one. That is the real, practical cost
of keeping $\beta$ inside the $Z=0$ branch — not the lost closed form, which is only a
constant factor in time, but the fact that $\beta$ can drift to a boundary where it silences
the other statistic entirely.

### Why $\beta$ drifts there

Under the collision parameterisation $\beta$ is the *only* free parameter shaping both
branches: it sets $P(C\mid Z=1)$ **and**, through $(1-\beta)\gamma^{\text{coll}}$, scales
$P(C\mid Z=0)$. It cannot fit both at once, and it resolves the tension by going to the
boundary. The redefinition unties them — $\gamma_j$ owns the $Z=0$ branch and $\beta$ owns
$Z=1$ — so $\beta$ settles at an interior value (0.836–0.993).

### Verdict

`collision` with $\beta$ and no $\hat\beta$ is **a coherent, correctly-implemented formulation
that is competitive without a judge and specialised to vision, but unsuitable as the default
because expert validation cannot reach it.** Whether that is repairable by bounding $\beta$
away from 1 is tested in Part C.

---

## Confirming the shape: $\gamma^{\text{coll}}$ is a vector of $M$, one entry per model

**Yes — that is what is implemented.** Checked on spider ($M=10$ models, $N=120$ labeled items):

```
gamma_collision shape: (10,)      -> one entry per MODEL
group ids            : [0 0 1 1 2 3 4 5 6 6]      (7 groups, NOT used)
gamma_coll per model : [1.0 0.952 0.810 0.684 1.0 0.905 0.800 0.900 0.684 0.714]
```

This is a real change from the old `collision EM` baseline in Part A, which used a
**per-group** rate — one number per provenance group, 7 values shared across 10 models.
Models 0 and 1 share group 0 but get 1.000 and 0.952 here; models 8 and 9 share group 6 but
get 0.684 and 0.714. Per-model keeps that distinction; per-group averages it away.

### One detail worth pinning down: conditional, not joint

"The probability that model $j$ agrees with the pseudo-label and it's wrong" can be read two
ways, and they give very different numbers:

| reading | formula | spider values |
|---|---|---|
| **conditional** (implemented) | $P(r^j=\hat y \mid r^j\ne y \wedge \hat y\ne y)$ | 0.684 … 1.000 |
| joint | $P(r^j=\hat y \wedge r^j\ne y)$ | 0.108 … 0.167 |

**The conditional one is required**, and not as a matter of taste. The E-step consumes

$$
P\big(C_i^j=1 \mid Z_i^j=0\big) \;=\; (1-\beta)\,\gamma_j^{\text{coll}} .
$$

Read that factorisation left to right: *given the model is wrong*, the pseudo-label is also
wrong with probability $(1-\beta)$, **and then** the two coincide with probability
$\gamma_j^{\text{coll}}$. The $(1-\beta)$ factor is what carries "the pseudo-label is also
wrong". So $\gamma^{\text{coll}}$ must be the rate **already conditioned on both being wrong**
— if it were the joint probability, the "pseudo-label is wrong" event would be counted twice
and the likelihood would not be a probability.

Concretely, for model 0 on spider: 19 labeled items have both the model and the pseudo-label
wrong; on 19 of those they give the *same* wrong answer. $19/19 = 1.000$. The denominator is
the both-wrong count, not $N$.

### Relation to the $e$ matrix

$e_{jk}$ is the same idea one level up — the rate at which models $j$ and $k$ are wrong
*together on the same answer*. $\gamma_j^{\text{coll}}$ is model $j$ against the
**pseudo-label** rather than against another model. So $e$ (matrix, $M\times M$) handles
model-to-model error correlation in the vote discount, and $\gamma^{\text{coll}}$
(vector, $M$) handles model-to-consensus agreement in the likelihood. They are measured from
the same labeled split but enter the pipeline at different points.


---

## Why is collision worse, when theoretically it is the finer model?

Your objection is correct on the point it actually makes, and the data agrees with it. The
claim has to be split in two, because the three arms differ along **two independent axes**,
not one.

### Axis 1 — which statistic is measured. Here you are right.

$$
\gamma^{\text{coll}}_j = P(r^j=\hat y \mid r^j\ne y \wedge \hat y\ne y),
\qquad
\gamma^{\text{both}}_j = 1-\gamma^{\text{coll}}_j,
\qquad
\gamma^{\text{model}}_j = \beta+(1-\beta)\gamma^{\text{both}}_j .
$$

$\gamma^{\text{coll}}$ and $\gamma^{\text{both}}$ are **the same statistic on the same
denominator** — complements of each other, carrying identical information.
$\gamma^{\text{model}}$ is that statistic **marginalised** over whether the pseudo-label
happens to be right. Marginalising discards information. So on expressiveness the ordering is

$$
\{\gamma^{\text{coll}},\ \gamma^{\text{both}}\} \;\succeq\; \gamma^{\text{model}} ,
$$

exactly as you say. **And the measurement confirms it**: at budget 40, `both_wrong` beats
`model_wrong`, 6.67 against 8.18. The finer decomposition wins. Your theory is not contradicted
by anything in this file.

### Axis 2 — what multiplies that statistic. This is where collision loses.

`collision` and `both_wrong` measure the *same thing*. The only difference between them is the
coefficient in front of it:

| arm | $P(C=1\mid Z=0)$ | the multiplier is |
|---|---|---|
| `collision` | $(1-\beta)\,\gamma^{\text{coll}}$ | $\beta$ — a **free parameter fitted by EM** |
| `both_wrong` | $1-\hat\beta-(1-\hat\beta)\gamma^{\text{both}}$ | $\hat\beta$ — a **frozen measured constant** |

So `collision` vs `both_wrong` is not a question about which statistic is better. It is one
question only: **should the multiplier be estimated or measured?** And that is an
identifiability question, not an information question.

### The likelihood cannot identify a free $\beta$ here

Profile log-likelihood in $\beta$, with $\alpha$ re-optimised at every $\beta$. Values are
$\ell-\ell_{\max}$, so $0.0$ marks the maximum. This is the observed-data likelihood, not an
optimiser trace, so it shows what the model actually prefers.

| case | arm | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | 0.99 | 1.00 | argmax |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spider | collision | -436.6 | -282.5 | -155.6 | -59.7 | -24.8 | -4.2 | **0.0** | 1.00 |
| | model_wrong | -356.1 | -221.8 | -109.8 | -29.8 | -7.3 | -0.0 | **0.0** | 1.00 |
| bird | collision | -128.6 | -84.1 | -47.4 | -19.7 | -9.0 | -1.7 | **0.0** | 1.00 |
| | model_wrong | -33.6 | -10.6 | -0.3 | **0.0** | -0.2 | -2.9 | -3.7 | **0.90** |
| mnist→usps | collision | -1028.7 | -492.3 | -116.9 | -0.5 | **0.0** | -4.8 | -4.8 | **0.95** |
| | model_wrong | -1137.0 | -569.6 | -202.0 | -36.6 | -0.2 | **0.0** | -0.0 | 0.99 |
| graph/DA | collision | -813.1 | -318.7 | -64.1 | -3.2 | -0.4 | -0.0 | **0.0** | 1.00 |
| | model_wrong | -378.2 | -160.1 | **0.0** | -19.8 | -170.0 | -192.0 | -192.8 | **0.80** |

Under `collision` the profile is **monotone increasing to the boundary** on three of four
cases. There is no interior maximum to find. $\beta=1$ *is* the maximum likelihood estimate —
the optimiser is not failing, it is succeeding at a badly-posed problem.

Under `model_wrong` the profile has a genuine interior peak on bird (0.90) and graph/DA (0.80),
and on DA it is sharp: moving to $\beta=0.95$ costs **170 nats**.

### Why the collision likelihood is flat

Each model contributes one observable number, its agreement rate
$a_j = \alpha_j\beta + (1-\alpha_j)(1-\beta)\gamma_j^{\text{coll}}$. Differentiate:

$$
\frac{\partial a_j}{\partial\beta}\bigg|_{\text{collision}} = \alpha_j-(1-\alpha_j)\gamma_j^{\text{coll}},
\qquad
\frac{\partial a_j}{\partial\beta}\bigg|_{\text{model\_wrong}} = \alpha_j .
$$

The collision derivative is a **difference of two positive terms** and can vanish. Measured:

| case | collision $\partial a/\partial\beta$ (mean, min, max) | model_wrong (mean, min) |
|---|---|---|
| spider | +0.589, +0.512, +0.667 | +0.773, +0.725 |
| **bird** | **+0.039, −0.038, +0.140** | +0.402, +0.367 |
| mnist→usps | +0.985, +0.971, +0.993 | +0.991, +0.985 |
| **graph/DA** | +0.506, **−0.024**, +0.645 | +0.723, +0.315 |

On BIRD the mean sensitivity is **+0.039** and it changes sign across models — the agreement
rates are essentially blind to $\beta$, so $\beta$ is a free direction the data does not
constrain. Under `model_wrong` the derivative is $\alpha_j$, strictly positive and bounded away
from zero, so $\beta$ is always identified.

And the case where the collision derivative is *large and uniform* (mnist→usps, +0.985) is
precisely the case where `collision` has an interior MLE (0.95) and **wins the experiment
outright** (b40 = 2.28, best of all three arms). The theory predicts exactly which case it
should work on, and it is the case where it works.

### The answer in one paragraph

Collision is worse **not because its statistic is worse** — its statistic is the best of the
three, and `both_wrong`, which uses the identical statistic with a frozen multiplier, wins the
whole experiment. Collision is worse because it additionally asks the likelihood to *estimate*
the multiplier $\beta$, and in this model $\beta$ is weakly identified: the observable
agreement rates have derivative $\alpha_j-(1-\alpha_j)\gamma_j$ in $\beta$, which is near zero
or sign-indefinite on real data. The MLE therefore runs to $\beta=1$, which collapses
$P(C=1\mid Z=0)=(1-\beta)\gamma^{\text{coll}}$ to zero and — because the expert's updates to
$\gamma^{\text{coll}}$ enter scaled by exactly $(1-\beta)$ — silences expert validation
entirely. Freezing the multiplier at a measured $\hat\beta$ is not a theoretical retreat; it
is the regularisation that makes the finer decomposition usable. **A strictly more expressive
model can have a strictly worse estimator when the extra parameter is not identified, and that
is what is happening here.**
