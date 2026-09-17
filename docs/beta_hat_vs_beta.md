# What $\hat\beta$ is, and why $\gamma$ cannot use the EM $\beta$

Answering four questions, in order:

1. What is $\hat\beta$?
2. Which $\gamma$ are we talking about? (`Trinh_proof.tex` defines two, and renames the symbol)
3. Does the formula for $\gamma$ contain $\hat\beta$?
4. Why not just use $\beta$?
5. Why do we need $\hat\beta$ at all?

Cross-references: `docs/three_betas.md` (the full three-way split), `Trinh_proof.tex`
§"Redefining $\gamma$" (line 3319).

---

## 0. The cast, so no symbol is ambiguous

| symbol | meaning | shape | how obtained |
|---|---|---|---|
| $y_i$ | the true label of item $i$ | — | unknown on the target |
| $\hat y_i$ | the **pseudo-label**: the winner of the weighted vote | — | computed each EM sweep |
| $r_i^j$ | model $j$'s answer on item $i$ | — | observed |
| $Z_i^j$ | $\mathbf 1(r_i^j = y_i)$ — "is model $j$ **correct**?" | latent | inferred ($\tau_i^j$) |
| $C_i^j$ | $\mathbf 1(r_i^j = \hat y_i)$ — "does model $j$ **agree with the vote**?" | observed | computed |
| $\alpha_j$ | $P(Z_i^j=1)$ — model $j$'s accuracy | vector, $M$ | fitted by EM |
| $\beta$ | $P(\hat y_i = y_i)$ — pseudo-label accuracy | scalar | **fitted by EM** |
| $\hat\beta$ | $P(\hat y_i = y_i)$ — the *same quantity* | scalar | **counted on gold labels** |
| $\gamma_j$ | $P(C_i^j=0 \mid Z_i^j=0)$ | vector, $M$ | counted, then expert-refreshed |
| $e_{jk}$ | wrong-and-agreeing rate for the pair $(j,k)$ | matrix, $M\times M$ | counted, then expert-refreshed |

The single most important line in this table: **$\beta$ and $\hat\beta$ denote the same
real-world quantity, obtained two completely different ways.** That is the whole answer to
questions 3 and 4, and everything below is the unpacking of it.

---

## 1. What is $\hat\beta$?

$\hat\beta$ is the pseudo-label accuracy **measured by counting**, on data where the gold
labels are known.

Take the labeled split $\mathcal L$. Run exactly the same voting rule the pipeline uses.
Look at how often its winner equals the gold label. That fraction is $\hat\beta$:

$$
\hat\beta_{\text{meas}}
\;=\;
\frac{\#\{\,i\in\mathcal L \;:\; \hat y_i = y_i\,\}+\lambda}
     {|\mathcal L| + 2\lambda}
$$

$\lambda$ is Laplace smoothing, so the quantity is defined even on a 10-item split.

In code (`build_stats` → `LabeledStatistics`):

```python
yhat         = hard_labels(LatentPlan(tc, e_excess).posterior(prior))  # the vote winner
pseudo_hits  = np.sum(yhat == 0)     # class 0 encodes "equals gold"
self._pseudo_acc = (pseudo_hits + lam) / (tc.shape[1] + 2 * lam)
```

Three properties to hold onto:

- **It is a scalar**, one number for the whole dataset. It is not per-model.
- **It requires gold labels.** You cannot compute it on the unlabeled target.
- **No EM is involved.** It is a count, available before the first EM sweep and constant
  during every sweep.

It is later refreshed by the expert, still by counting, with the standard blend:
`temp = mean(consensus == truth)` over validated target items, then
$\hat\beta \leftarrow \tfrac12(\hat\beta + \text{temp})$.

---

## 2. Which $\gamma$? `Trinh_proof.tex` defines two of them

This is a genuine trap in the document, and it is worth stating plainly before going
further: **the symbol $\gamma$ means two different things in two different parts of
`Trinh_proof.tex`, and one is close to the complement of the other.**

### 2a. The original $\gamma$ — an AGREEMENT probability (§ The Collision Probability, line 2090)

> "$\gamma_{g(j)}$ … the probability that a wrong model and a wrong pseudo-label produce the
> **same** wrong answer."

So the original $\gamma$ is the chance that the model's answer **collides with** — i.e.
*agrees with* — the pseudo-label, conditional on both being wrong. Write it
$\gamma_j^{\text{collision}}$:

$$
\gamma_j^{\text{collision}}
\;=\;
P\big(r_i^j = \hat y_i \;\big|\; r_i^j \ne y_i \;\wedge\; \hat y_i \ne y_i\big)
$$

Under this definition the quantity the E-step actually consumes is **not** $\gamma$ itself but

$$
d_j(\beta) \;=\; P\big(C_i^j = 0 \mid Z_i^j = 0\big) \;=\; 1-(1-\beta)\,\gamma_j^{\text{collision}} ,
$$

and note the problem the document then goes on to solve: $d_j$ is a **function of the free
parameter $\beta$**. That is the dependence §4 below shows must be removed.

### 2b. The redefined $\gamma$ — a DISAGREEMENT probability (§ Redefining $\gamma$, line 3319)

The document then explicitly reuses the symbol. Its own words:

> "The symbol $\gamma_j$ is retained, but it now names a *different quantity*."

$$
\boxed{\;
\gamma_j
\;=\;
P\big(C_i^j = 0 \mid Z_i^j = 0\big)
\;=\;
P\big(r_i^j \ne \hat y_i \;\big|\; r_i^j \ne y_i\big)
\;}
$$

That is **disagreement**, and the conditioning is on the model being wrong *alone* — nothing
is assumed about the pseudo-label. The whole point of the redefinition is that $\gamma$ now
**is** the probability the E-step consumes, instead of being an ingredient from which that
probability has to be built. The two $Z=0$ branches become simply

$$
P\big(C_i^j = 1 \mid Z_i^j = 0\big) = 1-\gamma_j,
\qquad
P\big(C_i^j = 0 \mid Z_i^j = 0\big) = \gamma_j ,
$$

with no $\beta$ in sight. **This is the $\gamma$ the code implements**, and the one meant
everywhere else in this file and in `docs/three_betas.md`. `gamma_counts` returns
`(disagree & cond).sum() / cond.sum()` — a disagreement ratio.

### 2c. How the two are related

$$
\gamma_j^{\text{both}} = 1-\gamma_j^{\text{collision}},
\qquad
\gamma_j^{\text{model}} = \hat\beta + (1-\hat\beta)\,\gamma_j^{\text{both}}
= 1-(1-\hat\beta)\,\gamma_j^{\text{collision}} = d_j(\hat\beta) .
$$

So all three names describe the same underlying evidence at different levels of
decomposition. `both_wrong` is the middle level; `collision` is its complement; `model_wrong`
is the fully marginalised level that the E-step wants.

### 2d. And no — none of them is $\hat\beta$

It is worth being explicit, because the two quantities do touch each other through the
conversion in §2c and that makes them easy to conflate:

| | quantity | it is a statement about … | shape |
|---|---|---|---|
| $\hat\beta$ | $P(\hat y_i = y_i)$ | **is the pseudo-label correct?** | scalar |
| $\gamma_j^{\text{collision}}$ | $P(r_i^j = \hat y_i \mid \text{both wrong})$ | does model $j$ **agree**? | vector, $M$ |
| $\gamma_j$ (redefined) | $P(r_i^j \ne \hat y_i \mid r_i^j \ne y_i)$ | does model $j$ **disagree**? | vector, $M$ |

$\hat\beta$ says nothing about any individual model — it is one number describing the vote's
correctness. $\gamma$ says nothing about whether the vote is correct — it is $M$ numbers
describing each model's relationship to the vote. $\beta$ is an *ingredient* of $\gamma$ under
`both_wrong`, and not even that under the `model_wrong` default.

### 2e. A cleanup worth doing

Reusing one symbol for two quantities is the source of this whole confusion. The fix is a
rename pass over `Trinh_proof.tex`: call the § Collision Probability one
$\gamma^{\text{coll}}$ throughout and reserve plain $\gamma$ for the redefined quantity. Not
done yet — flagged here so it is not forgotten.

---

## 3. Does the formula for $\gamma$ contain $\hat\beta$?

**It depends on the mode, and under the default the answer is no.**

$\gamma_j$ is always defined as the same thing — the quantity the E-step consumes:

$$
\gamma_j \;=\; P\big(C_i^j = 0 \mid Z_i^j = 0\big)
\;=\; P\big(r_i^j \ne \hat y_i \;\big|\; r_i^j \ne y_i\big)
$$

in words: *given model $j$ is wrong on this item, how often does it also **disagree** with
the pseudo-label?* What differs between modes is **what you counted**, and therefore whether
a conversion is needed to turn the count into that probability.

### `model_wrong` — the default. No $\hat\beta$.

Count directly over the event $\{$model $j$ is wrong$\}$:

$$
\gamma_j^{\text{model}}
=
\frac{\#\{\,i \;:\; r_i^j \ne \hat y_i \;\wedge\; r_i^j \ne y_i\,\}}
     {\#\{\,i \;:\; r_i^j \ne y_i\,\}}
$$

The conditioning event is exactly $Z_i^j = 0$ as the `.tex` defines it. What you counted **is**
what the E-step wants. No conversion, **no $\hat\beta$ anywhere**. In code:

```python
def conditional_gamma(self):
    if self.gamma_mode == "both_wrong":
        return gamma_to_conditional(self._gamma, self._pseudo_acc)
    return self._gamma          # model_wrong and pairwise: already P(C=0|Z=0)
```

### `both_wrong` — the only place $\hat\beta$ enters. Yes.

Count over the narrower event $\{$model wrong **and** pseudo-label wrong$\}$:

$$
\gamma_j^{\text{both}}
=
\frac{\#\{\,i \;:\; r_i^j \ne \hat y_i \;\wedge\; r_i^j \ne y_i \;\wedge\; \hat y_i \ne y_i\,\}}
     {\#\{\,i \;:\; r_i^j \ne y_i \;\wedge\; \hat y_i \ne y_i\,\}}
$$

This is a *different* probability from the one the E-step needs, so it must be converted:

$$
\boxed{\;\gamma_j^{\text{model}} \;=\; \hat\beta \;+\; (1-\hat\beta)\,\gamma_j^{\text{both}}\;}
$$

**Why that identity holds, in one paragraph.** Condition on $A = \{$model $j$ is wrong$\}$.
Split $A$ by whether the pseudo-label happens to be right:

- With probability $\hat\beta$, the pseudo-label is **right**. The model is wrong. A wrong
  answer cannot equal a right one, so they *must* differ — $C=0$ with probability $1$.
  Contribution: $\hat\beta \cdot 1$.
- With probability $1-\hat\beta$, the pseudo-label is **wrong**. Now both are wrong, which is
  precisely the conditioning event of $\gamma^{\text{both}}$, so they differ with probability
  $\gamma_j^{\text{both}}$. Contribution: $(1-\hat\beta)\cdot\gamma_j^{\text{both}}$.

Add the two branches: $\gamma_j^{\text{model}} = \hat\beta + (1-\hat\beta)\gamma_j^{\text{both}}$. $\square$

This is the same object the `.tex` calls $d_j(\beta) = 1-(1-\beta)\gamma_j^{\text{collision}}$,
since $\gamma^{\text{both}} = 1 - \gamma^{\text{collision}}$.

**So: $\hat\beta$ appears in the $\gamma$ formula only under `both_wrong`, purely as the
price of having counted the wrong denominator.** Switching the default to `model_wrong` was
precisely a way of not paying it.

---

## 4. Why not just use $\beta$?

Because **$\beta$ is a variable being optimised, and $\gamma$ has to be a constant while that
optimisation happens.** Three separate reasons, each sufficient on its own.

### Reason 1 — it destroys the closed-form M-step

This is the decisive one, and it is the entire motivation for the "Redefining $\gamma$"
section of the `.tex`.

The EM likelihood has two branches per observation:

$$
P(C_i^j \mid Z_i^j=1) : \;\; \beta^{C}(1-\beta)^{1-C}
\qquad\qquad
P(C_i^j \mid Z_i^j=0) : \;\; (1-\gamma_j)^{C}\,\gamma_j^{\,1-C}
$$

With $\gamma_j$ a **constant**, the $Z=0$ branch contributes only constants to
$\log$-likelihood as far as $\beta$ is concerned. So $\beta$ appears solely in the $Z=1$
branch, the score equation $\partial Q/\partial\beta = 0$ is linear, and you get

$$
\beta = \frac{\sum_{i,j}\tau_i^j C_i^j + s_\beta\pi_\beta}{\sum_{i,j}\tau_i^j + s_\beta}
$$

— a ratio of weighted counts, one line of code, exact.

Now suppose $\gamma_j$ used the live $\beta$, i.e. $\gamma_j = d_j(\beta)$. The $Z=0$ branch
becomes $\beta$-dependent and the objective gains a term

$$
\sum_j (1-\tau_i^j)\,\log d_j(\beta) \;=\; \sum_j (1-\tau_i^j)\,\log\big(1-\gamma_j+\beta\gamma_j\big)
$$

whose derivative is $\gamma_g / d_g(\beta)$ — a rational function of $\beta$. The score
equation stops being linear. Clearing denominators turns it into a **polynomial root-finding
problem**: with one shared $\gamma$ the `.tex` derives a cubic with coefficients
$c_2=-\gamma(A+B+D)$, $c_1=A(2\gamma-1)-B(1-\gamma)+D\gamma$, $c_0=A(1-\gamma)$; with a
different $\gamma_g$ per group, the degree grows with the number of groups and no closed form
exists at all. You would be running a numerical root solve inside every M-step of every sweep
of every warm-start of every judge iteration.

The `.tex` states the consequence directly (line 2516): *"Once $\gamma_j$ has been computed,
it is held fixed during the M-step."*

### Reason 2 — circularity

$\beta$ is computed **from** $\tau$. $\tau$ is computed **from** $\gamma$. If $\gamma$ were
computed from $\beta$, the three would form a closed loop with no entry point:

$$
\beta \;\longrightarrow\; \gamma \;\longrightarrow\; \tau \;\longrightarrow\; \beta
$$

You would need a fixed point of that loop, and nothing guarantees it exists, is unique, or is
reachable. With $\hat\beta$ frozen, the loop is cut: $\hat\beta$ is an input from outside,
$\gamma$ is fixed once, and the remaining $\tau \to \beta \to \tau$ cycle is ordinary EM with
its usual monotone-ascent guarantee.

The code makes the cut visible — $\gamma$ is computed **once, before the loop**:

```python
gamma = _clip(stats.conditional_gamma())[:, None]   # line 709, OUTSIDE the loop
...
for iteration in range(1, max_iters + 1):
    ...
    # P(C | Z = 1) uses beta; P(C | Z = 0) uses the measured, frozen gamma.
    like_z1 = np.where(C == 1.0, beta,      1.0 - beta)     # live beta
    like_z0 = np.where(C == 1.0, 1.0 - gamma, gamma)        # frozen gamma
```

`beta` is re-read every iteration; `gamma` is not, because it is not recomputed.

### Reason 3 — the two are estimated from different information

$\gamma$ is a statistic about *what actually happened on data where we know the answers.*
Computing it needs $y_i$ — look again at its definition, the conditioning event is
$r_i^j \ne y_i$, which mentions the true label. So $\gamma$ is inherently a **gold-label**
quantity, measured on the labeled split and thereafter revised only by the expert.

The M-step $\beta$ is the opposite: it is **inferred without any gold label**, from soft
posteriors $\tau$ and observed agreements $C$. Feeding it into $\gamma$ would mix an inferred
quantity into a measured one and silently propagate every EM fluctuation into a statistic that
is supposed to be evidence.

---

## 5. So why do we need $\hat\beta$ at all?

$\hat\beta$ is **not a model parameter and is never reported as a result.** It has exactly
three jobs, all of them plumbing:

**(a) The conversion, under `both_wrong` only.** As derived in §3. This is the job that gave
it its name, and it disappears under the default mode.

**(b) The starting value of the EM $\beta$.** EM needs somewhere to begin:

```python
beta = float(_clip(init.get("beta", stats.pseudo_accuracy)))
```

A measured count is a far better starting point than a guess — and the random-init ablation
measured how much better: randomising $\beta$'s start costs real MAE that budget only partly
repairs.

**(c) The prior mean of the $\beta$ M-step**, $\pi_\beta$ in the formula above:

```python
beta_prior = float(_clip(stats.pseudo_accuracy))
```

With the default `beta_strength = 0.0` this term has weight zero and does nothing; it exists
so the anchor *can* be switched on.

### The honest caveat

$\hat\beta$ is measured on the **labeled split**, while $\beta$ describes the **target**.
Under domain shift these differ, sometimes enormously — on MNIST→SVHN the labeled split is
MNIST, where $\hat\beta = 0.998$, while the target pseudo-labels are far worse than that.

Two things keep this from being fatal:

- The conversion **damps** the error rather than amplifying it:
  $\partial\gamma_j/\partial\hat\beta = 1-\gamma_j^{\text{both}} \le 1$. Measured on BIRD, a
  $0.14$ error in $\hat\beta$ propagates to only $0.058$ in $\gamma$.
- Under `model_wrong` — the default — **the conversion is not performed at all**, so this
  error path does not exist. $\hat\beta$ then only initialises EM, and EM re-estimates
  $\beta$ from target data anyway.

That is the real reason the default was flipped: not that `model_wrong` has better theory —
`both_wrong` keeps the exact chain-rule factorisation and is equally sound — but that it needs
one fewer measured constant to be transported across a domain shift.

---

## One-paragraph summary

`Trinh_proof.tex` uses $\gamma$ for two different quantities: the original one (§ The
Collision Probability) is an **agreement** rate, $P(r^j = \hat y \mid \text{both wrong})$, and
the redefined one (§ Redefining $\gamma$) is a **disagreement** rate,
$P(r^j \ne \hat y \mid r^j \ne y)$. The code implements the second. Separately, $\hat\beta$
is the pseudo-label accuracy **counted** on gold-labeled data, while $\beta$ is that same
quantity **inferred** by EM without gold labels — and neither is $\gamma$, because $\hat\beta$
is one scalar about whether the vote is right while $\gamma$ is $M$ numbers about how each
model relates to the vote. Under the default `model_wrong` mode $\gamma$'s formula contains no
$\beta$ of any kind. Under `both_wrong` it contains $\hat\beta$, and it must be $\hat\beta$
rather than $\beta$ because a $\gamma$ that moved with the live $\beta$ would make the
$\beta$ M-step a polynomial root-find instead of a ratio of counts, and would close a circular
dependency $\beta\to\gamma\to\tau\to\beta$ that ordinary EM has no way to resolve.
