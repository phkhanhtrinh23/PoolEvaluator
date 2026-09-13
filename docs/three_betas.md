# Three quantities called "pseudo-label accuracy", and which is which

> **Update.** `gamma_mode` now defaults to `model_wrong`, which needs none of this. The
> conversion described below is no longer on the live path; it survives as the relationship
> between the three conditionings, and as a worked example of why the default changed.
> §"The assumption behind it fails" is the reason.

Three different numbers in this codebase all mean "how often is the pseudo-label right".
They are computed in completely different ways, live at different levels of the loop, and
only one of them is a parameter of the model. Conflating them is easy and was the source of
a real confusion, so they are separated here and the variable names now keep them apart.

| | name in code | how obtained | when | overwritten by EM? |
|---|---|---|---|---|
| (1) | `LabeledStatistics.pseudo_accuracy` | **counted** against gold on the labeled split | once, before any EM | no |
| (2) | same, after `add_validated` | **counted** against the expert's answers | after EM + Expert, every judge call | no |
| (3) | the `beta` inside `validated_em` | **fitted** by the M-step | every EM sweep | yes — it *is* the fitted value |

---

## (1) Measured on the labeled split — a plain count, never inferred

```python
yhat         = hard_labels(LatentPlan(tc, e_excess).posterior(prior))  # argmax consensus
pseudo_hits  = np.sum(yhat == 0)       # class 0 == "matches gold"
pseudo_total = tc.shape[1]             # number of labeled items
self._pseudo_acc = (pseudo_hits + λ) / (pseudo_total + 2λ)
```

$$
\hat\beta_{\text{meas}}
=
\frac{\#\{\,i\in\mathcal{L}\;:\;\hat y_i=y_i\,\}+\lambda}{|\mathcal{L}|+2\lambda}
$$

Run the same voting rule on the labeled split, count how often its argmax equals gold.
Laplace-smoothed by $\lambda$ so the quantity is defined even on a tiny split. **Directly
computed, no EM involved.**

## (2) Refreshed after each expert answer — also a count

```python
temp_pseudo      = np.mean(consensus == truth)          # over validated target items
self._pseudo_acc = 0.5 * (self._pseudo_acc + temp_pseudo)   # the (old+temp)/2 rule
```

`consensus` is the pseudo-label the model held **before** pinning; `truth` is the expert's
answer. Still pure counting.

This is the one exposed to selection bias. The validated items were chosen *because the
consensus looked doubtful there*, and on an overruled item `consensus == truth` is `0` by
construction — so the sample is systematically displaced for a quantity defined as an
average over *all* items. See §"Is (old+temp)/2 good?" below.

## (3) Fitted inside EM — this is the M-step $\beta$

```python
beta = (float((tau * C).sum()) + beta_strength * beta_prior) / (float(tau.sum()) + beta_strength)
```

$$
\beta=\frac{\sum_{i,j}\tau_i^j C_i^j+s_\beta\pi_\beta}{\sum_{i,j}\tau_i^j+s_\beta}
$$

**Not counted — inferred**, from the soft posteriors $\tau$ and the agreements $C$, with no
access to any gold label. Refitted every sweep. With `beta_strength = 0` (the default) it
reduces to $\sum\tau C/\sum\tau$. This is the $\beta$ that appears in the model and in the
derivation; it is the one the closed-form result of `Trinh_proof.tex` is about.

---

## Why the rename

The measured quantity used to be called `self._beta`, which made it look like the M-step
parameter. It is now `LabeledStatistics.pseudo_accuracy`, with `pseudo_hits` /
`pseudo_total` behind it.

It is **not** named `gamma`, even though its only live job is to feed $\gamma$: `stats.gamma`
already exists and means $\gamma^{\text{both}}_j$, the counted both-wrong rate. Using the
same word for the conversion constant would produce expressions like
`gamma_to_conditional(gamma, gamma)`, which is worse than the problem being fixed. In the
`.tex` it is written $\hat\beta_{\text{meas}}$.

---

## What each one is actually used for

(3) is the model parameter. (1) and (2) do two jobs, and **only one of them is overwritten**:

```python
gamma = _clip(stats.conditional_gamma())[:, None]   # line 561 — BEFORE the loop, frozen
...
beta  = float(_clip(init.get("beta", stats.pseudo_accuracy)))   # line 569 — initial value
for iteration in ...:                               # the loop
    beta = (Σ τC + s_β π_β) / (Σ τ + s_β)           # line 603 — reassigned every sweep
```

| role of the measured value | lives | overwritten by EM? |
|---|---|---|
| conversion constant inside $\gamma$ | computed once, before the loop | **no** |
| starting value of the fitted $\beta$ | reassigned each sweep | yes |

The conversion is

$$
\gamma_j \;=\; \hat\beta_{\text{meas}}+\big(1-\hat\beta_{\text{meas}}\big)\,\gamma^{\text{both}}_j ,
$$

frozen for the whole EM run. Freezing it is what keeps the M-step closed form: if $\gamma$
moved with $\beta$ inside the loop, $Q_\beta$ would leave the span $\mathcal{S}$ and the
bounded numerical search of the collision model would be needed again.

### The loop, in order

```
build_stats(labeled split)          (1) ONCE:  e, gamma_both, pseudo_accuracy from gold
        |
validated_em(...)                   first EM; gamma frozen from those values
        |
   +--- while budget remains -----------------------------------+
   |   select object                                            |
   |   answer = expert.query(...)                               |
   |   stats.add_validated(...)     (2) EVERY call: all three   |
   |                                    refreshed (old+temp)/2  |
   |   if overruled:                                            |
   |       validated_em(..., warm-start)   gamma REBUILT from   |
   |                                       refreshed values,    |
   |                                       frozen again         |
   +------------------------------------------------------------+
```

`add_validated` runs **before** the warm-started re-solve, so the next EM run sees the
updated statistics rather than stale ones. On a confirmation the statistics still refresh
but no re-solve fires.

### How far the bias propagates

Two separate error sources, and I originally reported only the harmless one.

**Source 1 --- estimation error in $\hat\beta_{\text{meas}}$. This one is damped.** The map
$\beta\mapsto\beta+(1-\beta)\gamma^{\text{both}}$ has slope $1-\gamma^{\text{both}}<1$
whenever wrong answers ever collide. On BIRD after 20 judge calls:

| | value |
|---|---|
| measured $\hat\beta_{\text{meas}}$ | 0.2832 |
| true pseudo-label accuracy | 0.4270 |
| $P(C{=}0\mid Z{=}0)$ at the measured value | 0.7121 |
| $P(C{=}0\mid Z{=}0)$ at the true value | 0.7699 |
| **difference** | **0.058** |

A 0.14 error becomes 0.058. Second-order, as claimed.

### The assumption behind it fails

**Source 2 --- $\beta$ is the wrong quantity to substitute. This one is not damped, and it
dominates.**

The derivation's final step reads: *"assuming the pseudo-label's correctness is independent
of which classifier we are looking at, $P(B\mid A)=\beta$."* The correct conditional is
$P(\hat y_i = y_i \mid r_i^j \neq y_i)$ --- the chance the consensus is right **given this
classifier got it wrong**. Measured on the labeled splits:

| labeled split | $\beta=P(\hat y=y)$ | $P(\hat y=y \mid \text{model }j\text{ wrong})$ | gap |
|---|---|---|---|
| spider | 0.8167 | **0.2540** | **−0.563** |
| bird | 0.4583 | **0.1596** | **−0.299** |

The events are strongly dependent, and in hindsight obviously so: a classifier is usually
wrong on items that are **hard**, and the consensus is wrong on hard items too. Substituting
the marginal $\beta$ for the conditional overstates the first branch badly.

The consequence, measured directly against the same quantity counted without any conversion:

| labeled split | identity predicts $\gamma^{\text{model}}$ | directly counted | error |
|---|---|---|---|
| spider | 0.8508 | **0.3769** | **0.474** |
| bird | 0.6711 | **0.4845** | **0.187** |

An error of 0.47 on a quantity bounded in $[0,1]$. **This is not second-order.** My earlier
statement that the conversion costs only a damped 0.058 was wrong: it bounded the wrong
error term. The identity is algebraically correct and its independence premise is false, so
the conversion is *systematically biased*, not merely noisy.

### What flipping the default buys

`gamma_mode="model_wrong"` counts $P(C{=}0\mid Z{=}0)$ directly and the whole issue
disappears. Measured with **no judge calls at all**, so this is the conversion's cost in
isolation:

| split | $\gamma$ fed to the E-step | MAE |
|---|---|---|
| spider, `both_wrong` | 0.8466 | 9.18 |
| spider, **`model_wrong`** | **0.3769** | **3.61** |
| bird, `both_wrong` | 0.6715 | 6.63 |
| bird, **`model_wrong`** | **0.4845** | **4.14** |

Spider's MAE more than halves. That also explains the earlier ablation cleanly: `both_wrong`
was worst on all four blocks (12.54 vs 11.30 mean) because it feeds the E-step a $\gamma$
that is systematically too high.

$$
\boxed{\text{Count the conditional the model consumes. Do not reconstruct it from a
different conditional via an independence assumption that does not hold.}}
$$

### The assumption fails in every domain (`experiments/run_gamma_mode_diagnostic.py`)

The gap was first measured on the two Text2SQL splits. Extending it to all ten labeled
splits shows it is not a Text2SQL quirk:

| labeled split | $\beta$ | $P(\hat y=y\mid j\text{ wrong})$ | gap | $\gamma$ counted | identity predicts | error |
|---|---|---|---|---|---|---|
| text2sql/spider | 0.817 | 0.254 | −0.563 | 0.377 | 0.851 | 0.474 |
| text2sql/bird | 0.458 | 0.160 | −0.299 | 0.485 | 0.671 | 0.187 |
| vision/mnist→usps | 0.997 | 0.698 | −0.300 | 0.710 | 0.998 | 0.288 |
| vision/mnist→svhn | 0.998 | 0.730 | −0.268 | 0.719 | 0.998 | 0.279 |
| graph/AC | 0.832 | 0.428 | −0.404 | 0.476 | 0.855 | 0.382 |
| graph/AD | 0.827 | 0.424 | −0.403 | 0.471 | 0.850 | 0.381 |
| graph/CA | 0.857 | 0.398 | −0.459 | 0.441 | 0.876 | 0.436 |
| graph/CD | 0.855 | 0.397 | −0.457 | 0.441 | 0.874 | 0.434 |
| graph/DA | 0.817 | 0.353 | −0.464 | 0.406 | 0.842 | 0.436 |
| graph/DC | 0.820 | 0.361 | −0.459 | 0.411 | 0.845 | 0.434 |

| domain | mean \|gap\| | mean conversion error |
|---|---|---|
| text2sql | 0.431 | 0.330 |
| vision | 0.284 | 0.284 |
| graph | 0.441 | 0.417 |

### An open anomaly, recorded rather than explained away

`model_wrong` is the default on this evidence, and it improves `validated EM (no judge)` on
three of four blocks --- text2sql/A **7.91 → 3.87**, text2sql/B 6.44 → 4.96, graph
14.87 → 14.19. **On vision it is worse: 34.47 → 36.72, and `ACQ A_mu` 7.36 → 9.62.**

The natural hypothesis --- that vision's conversion is nearly unbiased, so `both_wrong`
keeps its extra conditioning for free --- was tested above and is **false**: vision's gap is
0.284 with a conversion error of 0.284, the identity predicting $\gamma\approx0.998$ where
the direct count gives 0.710.

So the vision exception is **unexplained**. It is left as a global default rather than a
per-domain switch for two reasons: the effect is ~6% relative on a block where every method
sits at 30--57 MAE and the estimator is failing for an unrelated reason (the gauge trap ---
94% of consensus labels wrong on SVHN), and hard-coding a domain exception whose proposed
mechanism has just been disproved would be fitting the measurement rather than the problem.

## Is `(old+temp)/2` good? The measured answer

Two questions, both from `experiments/run_refresh_rule.py` (6 cases across all three
tasks, budget 20, 3 seeds).

### Question 1 — does it recover the truth?

Recovered $\hat\beta_{\text{meas}}$ against the **true** target pseudo-label accuracy:

| case | **true** | entropy | info_gain | $A_\mu$ | $A_\mu$ sampled | random |
|---|---|---|---|---|---|---|
| text2sql/spider | **0.767** | 0.657 | 0.657 | 0.524 | 0.559 | **0.791** |
| text2sql/bird | **0.427** | 0.103 | 0.077 | 0.283 | 0.184 | **0.529** |
| vision/mnist→usps | **0.887** | 0.428 | 0.447 | 0.718 | 0.572 | **0.886** |
| vision/mnist→svhn | **0.060** | 0.105 | 0.106 | **0.053** | 0.093 | 0.088 |
| graph/AC | **0.748** | 0.529 | 0.409 | 0.392 | 0.400 | **0.836** |
| graph/CD | **0.677** | 0.242 | 0.144 | 0.317 | 0.244 | **0.739** |

**No, under active selection — yes, under random.** Every active selector undershoots
badly; random recovers the truth to within 0.02–0.09 on all five non-adversarial cases.
Same arithmetic, different sample: the rule is not at fault, the *selection* is.

$A_\mu$ is less biased than label entropy on 4 of 6 (bird 0.283 vs 0.103, usps 0.718 vs
0.428), so it softens the problem without escaping it.

### Question 2 — does it help the final answer anyway?

MAE under $A_\mu$ selection, `(old+temp)/2` against the pooled-counts alternative:

| case | true $\beta$ | **(old+temp)/2** | pooled counts | Δ |
|---|---|---|---|---|
| text2sql/spider | 0.767 | **5.42** | 8.62 | −3.20 |
| text2sql/bird | 0.427 | 7.61 | **5.27** | +2.34 |
| vision/mnist→usps | 0.887 | **2.86** | 3.23 | −0.37 |
| vision/mnist→svhn | 0.060 | **18.31** | 55.69 | **−37.38** |
| graph/AC | 0.748 | **7.33** | 8.64 | −1.30 |
| graph/CD | 0.677 | **12.92** | 14.99 | −2.07 |
| **MEAN** | | **9.07** | 16.07 | −7.00 |
| MEAN excluding svhn | | **7.23** | 8.15 | −0.92 |

**Yes — it wins 5 of 6.** And the biggest win is the case where the "collapse" looks worst:
on MNIST→SVHN the labeled split is MNIST with $\hat\beta_{\text{meas}}=0.998$ while the
target's true value is **0.060**. The geometric filter reaches 0.053 in 20 calls and scores
18.31; pooled counts — which weighs 20 target items against 3000 source items — stays at
0.992 and scores 55.69.

$$
\boxed{\text{Under severe shift the ``collapse'' is not over-reaction. It is the correct answer,
and the only rule fast enough to reach it.}}
$$

The single loss is BIRD, which is the selection-bias case: no real shift, so there is
nothing to adapt to, and the filter's speed only imports the bias.

### And which selector, under this rule?

| | entropy | info_gain | $A_\mu$ | $A_\mu$ sampled | random |
|---|---|---|---|---|---|
| MEAN | 14.50 | 15.09 | **9.07** | 12.97 | 12.99 |
| MEAN excluding svhn | 7.94 | 8.15 | 7.23 | **6.26** | 7.87 |

**`(old+temp)/2` + $A_\mu$ is the best pairing measured** — 9.07, against 14.50 for the
original pairing (entropy + the same refresh) and 16.07 for ($A_\mu$ + pooled counts). The
two changes are complementary: $A_\mu$ finds the items that move the accuracy estimate, and
the fast filter lets the statistics actually move once those items are revealed.

Note it wins *despite* a biased $\hat\beta_{\text{meas}}$, not because of an accurate one.

### Verdict

| | |
|---|---|
| **keep it?** | **yes** — 5 of 6 cases, and the mean gap is large |
| **what it is** | an exponential filter; weights are $2^{-1},2^{-2},\dots$, source retains $2^{-k}$ |
| **its strength** | adapts fast enough to cross a real domain gap (0.998 → 0.053 in 20 calls) |
| **its weakness** | inherits the selection bias of whatever picked the items |
| **when it loses** | no real shift *and* an active selector — BIRD is the example |
| **the refinement** | estimate $\hat\beta_{\text{meas}}$ from the probability-sampled audit; leave $e$ and $\gamma$ on every validated item, since they are conditional rates and far less exposed |

The refinement is a second-order improvement, not a correction of something broken: the
0.14 error in $\hat\beta_{\text{meas}}$ propagates to only 0.058 in $\gamma$, and vanishes
under `gamma_mode="model_wrong"`.

---

## The formula $\hat\beta_{\text{meas}}$ is actually used for

$\hat\beta_{\text{meas}}$ is not reported and is not a model parameter. It has exactly one
live job: converting the quantity we can **count** into the quantity the E-step **needs**.

### What we count versus what the model needs

The E-step needs the conditional

$$
\gamma_j \;=\; P\big(C_i^j=0 \;\big|\; Z_i^j=0\big),
$$

the chance that a *wrong* classifier disagrees with the pseudo-label. But the natural thing
to count on a labeled split is the **both-wrong** rate

$$
\gamma_j^{\text{both}} \;=\; P\big(C_i^j=0 \;\big|\; Z_i^j=0 \ \wedge\ \hat y_i\neq y_i\big),
$$

because that is the statistic the binary reduction destroys and the one a collision is
about. The two condition on different events, so one must be converted into the other.

### The conversion

```python
def gamma_to_conditional(gamma_both, beta):
    return beta + (1.0 - beta) * gamma_both
```

$$
\boxed{\;
\gamma_j
\;=\;
\hat\beta_{\text{meas}}
\;+\;
\big(1-\hat\beta_{\text{meas}}\big)\,\gamma_j^{\text{both}}
\;}
$$

### Derivation

Condition on whether the pseudo-label happens to be correct, given that the classifier is
wrong. Write $A=\{r_i^j\neq y_i\}$ (classifier wrong) and $B=\{\hat y_i=y_i\}$
(pseudo-label right). By the law of total probability,

$$
P(C_i^j=0\mid A)
=
P(C_i^j=0\mid A,B)\,P(B\mid A)
\;+\;
P(C_i^j=0\mid A,B^c)\,P(B^c\mid A).
$$

**Branch $B$ — the pseudo-label is right.** Then $\hat y_i=y_i$. The classifier is wrong, so
$r_i^j\neq y_i=\hat y_i$, meaning $C_i^j=0$ **with certainty**:

$$P(C_i^j=0\mid A,B)=1.$$

This is the step that makes the formula non-obvious. When the pseudo-label is correct, a
wrong classifier *always* disagrees with it — there is no way for the two to coincide.

**Branch $B^c$ — the pseudo-label is also wrong.** Both are wrong, which is precisely the
conditioning event of $\gamma^{\text{both}}$:

$$P(C_i^j=0\mid A,B^c)=\gamma_j^{\text{both}}.$$

Taking the pseudo-label's correctness as independent of which classifier is examined,
$P(B\mid A)=\hat\beta_{\text{meas}}$ and $P(B^c\mid A)=1-\hat\beta_{\text{meas}}$.
Substituting gives the boxed result. $\blacksquare$

### It is the `.tex`'s $d_j(\beta)$

Two wrong answers either collide or they do not, so
$\gamma_j^{\text{both}}=1-\gamma_j^{\text{collision}}$. Then

$$
\gamma_j
=
\beta+(1-\beta)\big(1-\gamma_j^{\text{coll}}\big)
=
\beta+(1-\beta)-(1-\beta)\gamma_j^{\text{coll}}
=
1-(1-\beta)\gamma_j^{\text{coll}}
=
d_j(\beta).
$$

So the measured $\gamma_j$ **is** the old $d_j(\beta)$ of the collision section. Nothing
about the model changed; the number is now read off data instead of carried as a function
of an unknown — which is exactly what restores the closed-form $\beta$ M-step.

### Sanity checks

| case | $\gamma_j$ | reading |
|---|---|---|
| $\hat\beta_{\text{meas}}=1$ | $1$ | pseudo-label always right ⟹ a wrong classifier always disagrees |
| $\hat\beta_{\text{meas}}=0$ | $\gamma_j^{\text{both}}$ | pseudo-label always wrong ⟹ the conditioning events coincide |
| $\gamma_j^{\text{coll}}=1$ | $\beta$ | collisions certain ⟹ the original binary model of the `.tex` |
| $\gamma_j^{\text{both}}=1$ | $1$ | wrong answers never collide ⟹ disagreement is certain |

The map is a convex combination of $1$ and $\gamma^{\text{both}}_j$, so it is monotone in
both arguments and stays in $[0,1]$ without clipping.

### Where the result is consumed

Computed once per EM run, **before** the loop, and frozen (`validated_em`, line 561):

```python
gamma = _clip(stats.conditional_gamma())[:, None]     # frozen for the whole run
```

then used in the E-step every sweep (line 603) as the $Z=0$ branch of the likelihood:

```python
like_z1 = np.where(C == 1.0, beta,        1.0 - beta)      # P(C | Z = 1)
like_z0 = np.where(C == 1.0, 1.0 - gamma, gamma)           # P(C | Z = 0)
tau     = alpha * like_z1 / (alpha * like_z1 + (1 - alpha) * like_z0)
```

$$
\tau_i^j
=
\frac{\alpha_j\,\beta^{C}(1-\beta)^{1-C}}
{\alpha_j\,\beta^{C}(1-\beta)^{1-C}+(1-\alpha_j)\,(1-\gamma_j)^{C}\gamma_j^{\,1-C}}
$$

Freezing $\gamma$ is what keeps the M-step closed form. If $\gamma$ moved with $\beta$
inside the loop, $Q_\beta$ would contain
$\log(1-\gamma^{\text{coll}}+\beta\gamma^{\text{coll}})$, leave the span $\mathcal{S}$, and
the bounded numerical search of the collision model would be needed again.

### Error propagation, and how to avoid the conversion entirely

The map has slope

$$
\frac{\partial\gamma_j}{\partial\hat\beta_{\text{meas}}}
=
1-\gamma_j^{\text{both}} ,
$$

which is **below 1 whenever wrong answers ever collide** — so an error in
$\hat\beta_{\text{meas}}$ is *damped*, not amplified. On BIRD after 20 judge calls,
$\gamma^{\text{both}}\approx0.60$, so the slope is $\approx0.40$ and a $0.14$ error becomes
$0.058$ (table above).

And the conversion is avoidable --- and is now avoided by default. `gamma_mode="model_wrong"`
(the default since this was measured) counts

$$
\hat\gamma_j=\frac{\#\{i:\;r_i^j\neq\hat y_i\ \wedge\ r_i^j\neq y_i\}+\lambda}
{\#\{i:\;r_i^j\neq y_i\}+2\lambda}
$$

directly — conditioning on the classifier being wrong, which is $Z_i^j=0$ as the `.tex`
defines it — so `conditional_gamma()` returns it unchanged and $\hat\beta_{\text{meas}}$
never enters the model at all. On Text2SQL protocol A that mode measured **6.06** mean MAE
against **8.91** for the default, so avoiding the conversion is not merely tidier.

### All three modes are the same count with a different denominator

`gamma_counts` builds one numerator and three possible denominators from the same two
masks:

```python
disagree     = tc != yhat[None, :]                      # C = 0
model_wrong  = tc != 0                                  # Z = 0   (classifier wrong)
pseudo_wrong = np.broadcast_to(yhat != 0, tc.shape)     #         (pseudo-label wrong)

cond = {"both_wrong":   model_wrong & pseudo_wrong,
        "model_wrong":  model_wrong,
        "pseudo_wrong": pseudo_wrong}[mode]

numerator, denominator = (disagree & cond).sum(axis=1), cond.sum(axis=1)
```

so every mode is $\hat\gamma_j=(\text{num}+\lambda)/(\text{den}+2\lambda)$ and only the
conditioning set changes:

$$
\hat\gamma_j^{\,\text{model}}
=
\frac{\#\{i:\;r_i^j\neq\hat y_i\ \wedge\ r_i^j\neq y_i\}+\lambda}
{\#\{i:\;r_i^j\neq y_i\}+2\lambda}
$$

$$
\boxed{\;
\hat\gamma_j^{\,\text{both}}
=
\frac{\#\{i:\;r_i^j\neq\hat y_i\ \wedge\ r_i^j\neq y_i\ \wedge\ \hat y_i\neq y_i\}+\lambda}
{\#\{i:\;r_i^j\neq y_i\ \wedge\ \hat y_i\neq y_i\}+2\lambda}
\;}
$$

$$
\hat\gamma_j^{\,\text{pseudo}}
=
\frac{\#\{i:\;r_i^j\neq\hat y_i\ \wedge\ \hat y_i\neq y_i\}+\lambda}
{\#\{i:\;\hat y_i\neq y_i\}+2\lambda}
$$

Note the numerator of `both_wrong` and `model_wrong` differ too: adding $\hat y_i\neq y_i$
to the conditioning set removes those items from **both** numerator and denominator.

### So what happens to $P(C=0\mid Z=0\wedge\hat y\neq y)$ under `model_wrong`?

**It is never computed, because the model never asks for it.** To be explicit, since the
warning further down is easy to misread: `model_wrong` performs **no conversion of any
kind**. It counts the exact quantity the E-step consumes. The numerical warning below
concerns an *optional* inverse that nothing in the pipeline requires. The E-step's $Z=0$ branch
needs $P(C_i^j=0\mid Z_i^j=0)$ and nothing else; `model_wrong` supplies exactly that, so
`conditional_gamma()` returns the counted value unchanged:

```python
def conditional_gamma(self):
    if self.gamma_mode == "both_wrong":
        return gamma_to_conditional(self._gamma, self._pseudo_acc)   # needs beta_hat
    return self._gamma                                               # returns it as is
```

The both-wrong rate is a quantity you may still *want* — it is the complement of the
collision rate, $\gamma^{\text{coll}}_j=1-\gamma^{\text{both}}_j$, and so is the natural
thing to report when discussing how often two wrong answers coincide. Two ways to get it:

**Count it directly.** Run `gamma_counts(..., mode="both_wrong")` alongside. It is a second
pass over the same labeled matrix, costs nothing, and is exact. This is the right way if
you want the number.

**Or invert the conversion --- but you never have to, and should not.** Solving
$\gamma^{\text{model}}_j=\beta+(1-\beta)\gamma^{\text{both}}_j$ for the both-wrong rate,

$$
\boxed{\;
\gamma_j^{\text{both}}
=
\frac{\gamma_j^{\text{model}}-\beta}{1-\beta}
\;}
$$

which is exact but **numerically bad, and worse than useless as $\beta\to1$**: the
denominator vanishes, so a small error in $\hat\beta_{\text{meas}}$ or in
$\gamma^{\text{model}}$ is divided by nearly zero. The forward direction damps error by
$1-\gamma^{\text{both}}<1$; the inverse *amplifies* it by $1/(1-\beta)$. The asymmetry has
a plain reading: when the pseudo-label is almost always right, almost no item satisfies
"both wrong", so the both-wrong rate is being estimated from almost no data and the
inversion is reconstructing a quantity the data barely constrains.

$$
\boxed{
\text{Count the rate you need. Convert only in the damping direction, and never invert.}
}
$$

### Summary of the three routes

| route | conversion | risk |
|---|---|---|
| `both_wrong`: count $\gamma^{\text{both}}$, convert **forward** to $\gamma$ | damping, $\times(1-\gamma^{\text{both}})$ | safe, but drags in $\hat\beta_{\text{meas}}$ |
| **`model_wrong`: count $\gamma$ directly** | **none** | **none** |
| on `model_wrong`, **invert** to recover $\gamma^{\text{both}}$ for reporting | amplifying, $\times 1/(1-\beta)$ | **bad --- count it directly instead** |

Only the third route is dangerous, and it is never required: if the both-wrong rate is
wanted for reporting, a second `gamma_counts(..., mode="both_wrong")` pass gives it exactly.

This is the concrete argument for `model_wrong` as the default: it is the conditioning the
E-step actually uses, so the system contains **no conversion at all** --- not a safe one,
not a risky one.

---

## What flipping the default actually did, end to end

All four blocks, `both_wrong` → `model_wrong`, corrected anchor in both columns:

| method | Text2SQL/A | Text2SQL/B | image | node |
|---|---|---|---|---|
| prior only | 3.60 → 3.60 | 4.72 → 4.72 | 57.49 → 57.49 | 13.44 → 13.44 |
| PoolEval-SQL (paper) | 15.68 → 15.68 | 12.15 → 12.15 | 31.45 → 31.45 | 16.07 → 16.07 |
| collision EM (`.tex`) | 14.06 → 14.06 | 10.14 → 10.14 | 30.69 → 30.69 | 19.64 → 19.64 |
| **validated EM (no judge)** | 7.91 → **3.87** (-4.03) | 6.44 → **4.96** (-1.48) | 34.47 → **36.72** (+2.26) | 14.87 → **14.19** (-0.69) |
| + judge b=40 (reject-all) | 4.68 → **4.16** (-0.52) | 3.27 → **3.69** (+0.42) | 24.88 → **24.70** (-0.18) | 9.11 → **7.96** (-1.15) |

The first three rows are identical by construction --- none of them consumes `gamma_mode` ---
which is the sanity check that the harness is isolating the change.

**The headline estimator improves on three of four blocks**, most on text2sql/A
(**7.91 → 3.87**). That block now reads 3.87 against the labeled prior's 3.60: for the first
time the unsupervised estimator is within a third of a point of a prior built from gold
labels, where the gap used to be 4.3 points.

**Vision is the exception and it is unexplained** --- 34.47 → 36.72, and `ACQ A_mu`
7.36 → 9.62. The hypothesis that vision's conversion is nearly unbiased was tested above and
is false (gap 0.284, conversion error 0.284). It is left as a global default rather than a
per-domain switch because the effect is ~6% relative on the one block where the estimator is
already failing for an unrelated reason, and because hard-coding a domain exception whose
proposed mechanism has been disproved would be fitting the measurement rather than the
problem.

Full acquisition tables under the new default: `docs/acquisition_results.md`.

**The one systematic exception** to `model_wrong` being better is deterministic $A_\mu$
acquisition, which prefers `both_wrong` on all four blocks. Because that is the paper's
headline configuration, it was measured on its own across all 15 cases --- see
`docs/gamma_mode_headline.md`. Short version: `both_wrong` wins that comparison by 0.93 MAE,
the margin is concentrated in the vision block, it vanishes for the *sampled* variant of the
same score, and the mechanism proposed to explain it failed its own falsification test. The
default stays `model_wrong` as a judgement call, not as a measurement.
