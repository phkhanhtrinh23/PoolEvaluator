# Validated EM: measured `e` and `gamma`, plus the i-EM judge loop

This note records what changed relative to `Trinh_proof.tex`, why the change makes
the M-step *easier* rather than harder, and what the experiments measured.

Code: `pooleval/validated_em.py` · tests: `tests/test_validated_em.py` ·
experiments: `experiments/run_validated_em.py`

---

## 1. What changed, in one table

| | `Trinh_proof.tex` (collision version) | this note |
|---|---|---|
| correlated error | provenance **groups** declared by hand; loading $u_g$ fitted | matrix $e\in[0,1]^{M\times M}$ **measured** on a labeled split |
| $\gamma$ | $\gamma_{g(j)} = P(\text{two wrong answers collide})$, a free parameter | $\gamma_j = P(C=0\mid Z=0)$, **measured** and frozen |
| $\alpha$ M-step | closed form | closed form (**unchanged**) |
| $\beta$ M-step | **not** closed form — bounded 1-D search | **closed form again** |
| labels | none | expert validates one item at a time, chosen by information gain |

---

## 2. The model

Per item $i$ and model $j$:

$$
Z_i^j=\mathbf 1\{\text{model }j\text{ is correct on item }i\},\qquad
C_i^j=\mathbf 1\{r_i^j=\hat y_i\},
$$

$$
\alpha_j=P(Z_i^j=1),\qquad \beta=P(\hat y_i=y_i).
$$

The correct branch is untouched:

$$
P(C_i^j=1\mid Z_i^j=1)=\beta,\qquad P(C_i^j=0\mid Z_i^j=1)=1-\beta.
$$

The wrong branch is where the change lives:

$$
\boxed{\;P(C_i^j=0\mid Z_i^j=0)=\gamma_j\;}
\qquad\text{measured, not fitted.}
$$

### 2.1 Why this is the same object the `.tex` called $d_j(\beta)$

The `.tex` derives

$$
d_j(\beta)=P(C_i^j=0\mid Z_i^j=0)=1-(1-\beta)\gamma_j^{\text{coll}} .
$$

So the new $\gamma_j$ **is** $d_j$. The difference is epistemic, not structural: $d_j$
used to be a *function of the free parameter* $\beta$, and is now a *number read off
labeled data*. Setting $\gamma_j^{\text{coll}}=1$ gives $\gamma_j=\beta$ and recovers
the original binary model; that identity is checked in
`test_gamma_one_recovers_the_original_no_collision_model`.

### 2.2 What gets measured, exactly

The spec counts, on the labeled split, the items where *the prediction and the
pseudo-label differ and both are wrong*. That is a rate **conditional on both being
wrong**, call it $\gamma^{\text{both}}_j$, and it is the exact complement of the old
collision rate:

$$
\gamma^{\text{both}}_j = 1-\gamma_j^{\text{coll}} .
$$

Converting it to the conditional the E-step needs is one line. Given the model is
wrong, the pseudo-label is right with probability $\beta$ — and then the two
*necessarily* differ; otherwise they differ with probability $\gamma^{\text{both}}_j$:

$$
\boxed{\;\gamma_j \;=\; P(C=0\mid Z=0)\;=\;\beta+(1-\beta)\,\gamma^{\text{both}}_j\;}
$$

which equals $1-(1-\beta)\gamma_j^{\text{coll}}=d_j(\beta)$, closing the loop.
(`gamma_to_conditional`, verified in `test_conditional_gamma_reproduces_d_of_beta_from_the_tex`.)

Two other conditionings are implemented and measured, because the spec admits them:

| mode | conditioning event $Z=0$ | note |
|---|---|---|
| `both_wrong` *(default)* | model wrong **and** pseudo-label wrong | the counted event; needs the $\beta$ conversion above |
| `model_wrong` | model wrong | matches $Z_i^j$ as defined in the `.tex`; plugs straight in, no conversion |
| `pseudo_wrong` | pseudo-label wrong | the literal reading of "the pseudolabel is different from the true label" |

---

## 3. The E-step and both M-steps

**E-step.** Bayes, unchanged in form:

$$
\tau_i^j=\frac{\alpha_j\,\beta^{C}(1-\beta)^{1-C}}
{\alpha_j\,\beta^{C}(1-\beta)^{1-C}+(1-\alpha_j)\,(1-\gamma_j)^{C}\gamma_j^{\,1-C}} .
$$

On a **validated** item the true answer is known, so $Z$ is *observed*:
$\tau_i^j=\mathbf 1\{r_i^j=e(i)\}$. That is what makes a judge call inform every
model's $\alpha$, not just one item's label.

**$\alpha$ M-step — unchanged.** $\gamma$ enters only the $C\mid Z$ terms, never the
$Z\mid\alpha$ terms, so the `.tex` lemma applies verbatim with
$A=s_j\pi_j+\sum_i\tau_i^j$ and $B=s_j(1-\pi_j)+N-\sum_i\tau_i^j$:

$$
\boxed{\;\alpha_j=\frac{\sum_i \tau_i^j+s_j\pi_j}{N+s_j}\;}
$$

**$\beta$ M-step — closed form restored.** This is the concrete pay-off. In the
collision model $\beta$ appeared in *both* branches, so $Q_\beta$ mixed
$\log\beta$ with $\log\big(1-\gamma^{\text{coll}}+\beta\gamma^{\text{coll}}\big)$ —
outside the span $\mathcal S$ of the lemma, hence the bounded 1-D search. Freezing
the $Z=0$ branch deletes $\beta$ from it, leaving

$$
Q_\beta=\sum_{i,j}\tau_i^j\big[C_i^j\log\beta+(1-C_i^j)\log(1-\beta)\big]
       +s_\beta\big[\pi_\beta\log\beta+(1-\pi_\beta)\log(1-\beta)\big],
$$

which is in $\mathcal S$, so

$$
\boxed{\;\beta=\frac{\sum_{i,j}\tau_i^j C_i^j+s_\beta\pi_\beta}
{\sum_{i,j}\tau_i^j+s_\beta}\;}
$$

`test_beta_m_step_is_closed_form_and_maximises_q_beta` checks this against
`scipy.optimize.minimize_scalar` to $10^{-7}$.

---

## 4. Where `e` enters

$e_{jk}$ = fraction of labeled items on which models $j$ and $k$ return the **same**
answer and that answer is **wrong**. Execution errors carry distinct negative ids, so
two crashes never count as a collision.

Only correlation *above chance* should be penalised, so the discount uses the excess
over the cross-group off-diagonal mean, $\tilde e = \max(e-\bar e_{\text{cross}},0)$
with zero diagonal. A model's vote at item $i$ is then worth

$$
\text{disc}_j=\frac{1}{1+\sum_{k\neq j,\; r_i^k=r_i^j}\tilde e_{jk}},
\qquad
U(i,\ell)\propto\exp\Big(\textstyle\sum_{j:\,r_i^j=\ell}\alpha_j\,\text{disc}_j\Big).
$$

**This is a strict generalisation of what the repo already does.** Put
$\tilde e_{jk}=u_g$ for same-group pairs and $0$ otherwise and the sum collapses to
$u_g\,(n_{g,\ell}-1)$, reproducing `pooleval/latent.py`'s
$1/(1+u_g(n_{g,\ell}-1))$ exactly — checked over 50 random pools in
`test_vote_discount_reproduces_the_group_loading_formula_exactly`. Provenance grouping
is the special case in which correlation is *declared* instead of *measured*.

---

## 5. The validation loop

```
solve EM to convergence
repeat until budget spent:
    IG(o) = H(P) - sum_l U(o,l) H(P_l)          # Hung et al. Eq. (9)
    o*    = argmax IG                            # Eq. (10)
    a     = expert chooses among o*'s candidate answers
    refresh e and gamma with the revealed label   # <-- before the warm start
    if a == pseudo_label[o*]:  keep going, no re-solve
    else:                      pin U(o*,.) one-hot, warm-start EM to convergence
```

$H(P)=\sum_o H(o)$ over the latent-answer posteriors (Eq. 7), and $H(P_l)$ is the
whole-set entropy after *pretending* the expert said $\ell$ and re-solving — which is
what separates this from uncertainty sampling: pinning $o$ moves every $\alpha_j$,
which moves every *other* item's posterior.

### 5.1 The refresh rule

After the judge answers and **before** the warm start, both statistics are recomputed
on the validated target items (`temp`) and blended:

$$
\boxed{\;\text{new}=\tfrac{1}{2}\big(\text{old}+\text{temp}\big)\;}
$$

a one-line exponential filter: each update halves the weight of everything before it,
so source-domain numbers hand over to target-domain ones without ever being discarded.
Entries whose `temp` is undefined (a model with no eligible validated item yet) keep
their old value, so the blend is a no-op there rather than a pull toward the $1/2$
fallback.

Validated items enter $\gamma$ through the pseudo-label the consensus **would have**
produced *without* the pin. Using the pinned label instead would make every wrong
model disagree by construction and drive $\gamma\to1$, which says nothing about the
unvalidated items $\gamma$ is used on. (`test_pinning_does_not_inflate_gamma...`)

`update_rule="counts"` — pool the validated items into the source counts, weighting
every labeled item equally — is implemented as the comparison.

---

## 6. What was run

Three domains, in this order. Every method sees the same observation matrix and the
same anchor prior; only the estimator changes.

| domain | pool | target items | labeled split |
|---|---|---|---|
| Text-to-SQL | 10 OpenAI models over 7 provenance groups, real SQL executed on live SQLite | 150 per dataset (5 datasets) | Spider/BIRD **train** (120 items, re-executed offline), or a 40% slice of the target |
| Image classification | 15 CNNs (5 architecture families x 3 seeds) | 600 (MNIST -> USPS, MNIST -> SVHN) | 3000 source-domain validation images |
| Node classification | 15 GNNs (5 families x 3 seeds) | 600 (six citation-network shifts) | 3000 source-graph validation nodes |

Two protocols for where the labeled statistics come from:

* **A -- source split.** The genuine pre-built labeled set: Spider/BIRD train, or the
  source-domain validation split. No target label is touched, so this is the
  deployable setting. For Text2SQL it is available for `spider` and `bird`, whose
  cached source generations re-execute exactly (the rebuilt per-model EX reproduces
  the stored seen prior to within 2/120 items on one model, from SQL timeouts).
* **B -- target holdout.** A 40% slice of the target items is treated as labeled and
  the estimator is scored on the disjoint 60%. This is the in-distribution control
  and the only protocol available for the datasets whose source generations cannot be
  reproduced offline (`sqlflow`, `bird_minidev`, `spider2local` were generated with a
  different source sample, so the cached prediction ids no longer line up).

The expert is simulated in two regimes:

* **restricted choice** -- it must pick one of the answers the pool produced. When no
  model is right the correct answer is not on the menu and the call is wasted.
* **may reject all** -- it can also say "none of these", which pins the item to a
  class no model produced. This is exactly what `zoo/judge.py`'s `RealJudge` does
  with a live LLM; `pooleval.validated_em.JudgeExpert` adapts it to this loop.

Reproduce:

```bash
python experiments/run_validated_em.py --domain text2sql \
    --budgets 0 5 10 20 40 --ig-candidates 50 --ig-iters 5 \
    --out results/validated_em_text2sql.json
python experiments/run_validated_em.py --domain vision \
    --n-target 600 --n-labeled 3000 --ig-candidates 25 \
    --out results/validated_em_vision.json
python experiments/run_validated_em.py --domain graph \
    --n-target 600 --n-labeled 3000 --ig-candidates 25 \
    --out results/validated_em_graph.json
python experiments/report_validated_em.py          # regenerates the tables below
```

---

## 7. Diagnostic: information gain is almost exactly per-item entropy here

Eq. (9) is expensive — one warm-started re-solve per (candidate item, candidate
answer) pair — and its whole justification is that validating item $o$ also moves
*other* items, through the model reliabilities. Measured on the real pools, that
spillover is almost absent:

| pool | anchor $s$ | mean $H(o)$ | mean $IG(o)$ | mean $IG-H(o)$ | max $\lvert IG-H(o)\rvert$ | share of $IG$ that is $H(o)$ |
|---|---|---|---|---|---|---|
| spider | 120 | 0.103 | 0.105 | +0.0014 | 0.013 | **98.7%** |
| spider | 1 | 0.096 | 0.094 | −0.0019 | 0.029 | 102.0% |
| bird | 120 | 0.842 | 0.905 | +0.0631 | 0.150 | **93.0%** |
| bird | 1 | 0.784 | 0.767 | −0.0173 | 0.352 | 102.3% |

and the two rankings are effectively the same object:

| pool | Spearman$(IG, H)$ | Kendall $\tau$ | same top-1 | top-5 overlap |
|---|---|---|---|---|
| spider | **1.000** | 0.999 | yes | 5/5 |
| bird | **0.995** | 0.975 | yes | 4/5 |

**Why.** In Hung et al. each worker carries a full $K\times K$ confusion matrix
estimated from few items, so one revealed label moves those matrices appreciably and
the spillover is real. Here each model carries a *single scalar* $\alpha_j$, and its
M-step is $(\sum_i\tau_i^j+s_j\pi_j)/(N+s_j)$ — pinning one item of $N=150$ against an
anchor of strength $s\approx120$ moves $\alpha_j$ by at most $1/(N+s)\approx 0.004$.
The knock-on change in every other item's posterior is second-order, so
$IG(o)\approx H(o)$, the pinned item's own entropy going to zero. Dropping the anchor
to $s=1$ does not rescue it: the spillover becomes slightly *negative* on average.

This is a property of the model, not of the approximation — it holds with the
candidate set at all $N$ items and the re-solve warm-started for 5 sweeps. The
practical consequence is in the tables: `[info_gain]` and `[entropy]` differ by less
than 0.1 MAE points nearly everywhere, at roughly 100x the selection cost.

---

## 7b. Acquisition: what the query is *for*

Hung et al.'s Eq. (9) minimises uncertainty about **every label**. That is the right
objective for recovering labels and the wrong one for estimating a **mean**. §7 measured
the consequence: on a pool whose consensus is usually wrong, entropy is *anti*-correlated
with consensus failure, because when the whole pool fails the same way it agrees
confidently, so the gauge trap sits at LOW entropy and uncertainty sampling is
structurally blind to it.

The fix is not to abandon information gain. It is to take the information gain of the
quantity being estimated.

### The criterion

Write $\mu$ for the accuracy being estimated and $V=\operatorname{Var}(\mu\mid D)$.
By the law of total variance,

$$
A_\mu(q)=V-\Big[p_q V_q^{(1)}+(1-p_q)V_q^{(0)}\Big]
        =\operatorname{Var}\big(E[\mu\mid D,Z_q]\;\big|\;D\big),
$$

so $A_\mu(q)$ is *how much learning the answer at $q$ could move the accuracy estimate*.
For binary $Z_q$ it collapses to

$$
\boxed{\;A_\mu(q)=p_q(1-p_q)\,(m_1-m_0)^2\;}
\qquad\text{equivalently}\qquad
A_\mu(q)=\frac{\operatorname{Cov}(\mu,Z_q\mid D)^2}{\operatorname{Var}(Z_q\mid D)} .
$$

Two factors, and the second is the whole point. $p_q(1-p_q)$ is item uncertainty — the
part entropy sampling approximates. $(m_1-m_0)^2$ is **global leverage**: how far the
accuracy estimate would move depending on the answer. Under independent errors
$m_1-m_0=1/N$ for every item, the leverage factor is constant, and maximising $A_\mu$
reduces exactly to uncertainty sampling — so entropy is the *special case*, not a rival.
Under correlated errors an item can have $p_q=0.95$ and leverage two orders of magnitude
larger, and $A_\mu$ selects it.

In this repo the target is a vector of $M$ per-model accuracies, and the expert names one
of several candidate answers rather than a binary outcome, so the implemented form is

$$
A_\mu(q)=\sum_j \operatorname{Var}_{\ell\sim U(q,\cdot)}\big(E[\alpha_j\mid D,\,e(q)=\ell]\big),
$$

computed from the same hypothetical re-solves as the entropy criterion
(`acquisition_scores`), so the two cost identically. The binary reduction is verified
numerically to $10^{-9}$ in `test_mean_gain_reduces_to_p_times_one_minus_p_times_leverage_squared`.

### Keeping the final number valid

Horvitz–Thompson is an **estimator, not a sampling strategy**. It undoes a selection bias
whose size it knows; an item with $\pi_i=0$ is unrecoverable at any weight, because
$1/0$ is not a number. A deterministic top-$k$ therefore **cannot** be HT-corrected after
the fact — those items have $\pi=1$ and every other item $\pi=0$. The design has to carry
known, strictly positive inclusion probabilities from the start.

`sampling_probabilities` does that:

$$
p_i=(1-\epsilon)\,\frac{\exp(A_\mu(i)/T)}{\sum_j\exp(A_\mu(j)/T)}+\frac{\epsilon}{n},
$$

so high-value items stay strongly preferred while every item keeps $p_i\ge\epsilon/n>0$.
`horvitz_thompson` raises on $\pi_i=0$ rather than silently returning an infinity.

The two roles are then kept apart, which is what makes the final number defensible:

| role | who does it |
|---|---|
| **which labels to buy** | $A_\mu$ |
| **how the mean stays valid** | probability sampling + HT / model-assisted / IPW |

`run_validation(pilot=k)` reserves the first $k$ calls for a simple random sample of the
whole target set — exactly known $\pi=k/N$, a genuine audit. The remaining budget is spent
actively; those items improve the model but never do audit duty, because adaptive
sampling without replacement leaves no clean marginal inclusion probability.

### The estimators (`pooleval/estimators.py`)

$$
\hat\mu_{\mathrm{HT}}=\frac1N\sum_{i\in S}\frac{Z_i}{\pi_i},
\qquad
\hat\mu_{\mathrm{MA}}=\frac1N\sum_i \hat p_i+\frac1N\sum_{i\in S}\frac{Z_i-\hat p_i}{\pi_i}.
$$

Both are unbiased for the same reason: $E[I_i]=\pi_i$ cancels the weight. The
model-assisted form costs **variance** when the model is bad, never bias — verified by
Monte Carlo over the sampling design with a model that says 5% when the truth is 70%
(`test_model_assisted_is_unbiased_even_when_the_model_is_badly_wrong`).
`inverse_variance_fuse` combines the model-based and design-based estimates by precision.

**Cross-fitting is not optional.** Audit items are *pinned* in the fitted model, so its
prediction there IS the label: residuals would be identically zero, the correction term
would vanish, and `model_assisted` would silently collapse back into the plain model
average — the exact bias it exists to remove. The implementation re-solves with the audit
labels withheld, so a prediction on an audited item never saw that item's label. This was
a live bug in the first version, caught by the numbers being implausibly bad.


---

## 8. Results

All three domains, one anchor convention: the Beta anchor strength is capped at `N` so
the prior can never outweigh the target evidence, and the vision/graph ports declare the
prior's sd the way the rest of the repo does rather than as a binomial standard error.
That correction matters more than anything else measured here -- see §9.

<!-- generated by experiments/report_validated_em.py -->

MAE in accuracy points (x100), lower is better. Best per column in bold. Budgets [0, 5, 10, 20, 40], expert accuracy 1.0, IG candidates 25, IG sweeps 5.


### Text2SQL -- protocol A (labeled source split)

**Estimators and the judge budget**

| method | spider | bird | **mean** |
|---|---|---|---|
| prior only (no target data) | 3.73 | **3.47** | **3.60** |
| PoolEval-SQL (current paper) | 13.45 | 17.91 | 15.68 |
| binary agreement EM (gamma=1) | 18.97 | 12.10 | 15.54 |
| collision EM (group gamma, .tex) | 11.15 | 16.98 | 14.06 |
| validated EM (no judge) | 9.18 | 6.63 | 7.91 |
| validated EM + judge b=5 [info_gain] | 6.45 | 6.63 | 6.54 |
| validated EM + judge b=5 [entropy] | 6.45 | 6.63 | 6.54 |
| validated EM + judge b=5 [random] | 9.41 | 6.63 | 8.02 |
| validated EM + judge b=5 [info_gain, may reject all] | 6.00 | 9.13 | 7.56 |
| validated EM + judge b=10 [info_gain] | 7.03 | 13.40 | 10.21 |
| validated EM + judge b=10 [entropy] | 7.02 | 13.40 | 10.21 |
| validated EM + judge b=10 [random] | 9.41 | 6.63 | 8.02 |
| validated EM + judge b=10 [info_gain, may reject all] | 7.57 | 10.19 | 8.88 |
| validated EM + judge b=20 [info_gain] | 7.03 | 11.88 | 9.45 |
| validated EM + judge b=20 [entropy] | 7.02 | 11.87 | 9.44 |
| validated EM + judge b=20 [random] | 9.36 | 15.35 | 12.36 |
| validated EM + judge b=20 [info_gain, may reject all] | 8.20 | 9.38 | 8.79 |
| validated EM + judge b=40 [info_gain] | 5.94 | 11.88 | 8.91 |
| validated EM + judge b=40 [entropy] | 5.94 | 11.87 | 8.90 |
| validated EM + judge b=40 [random] | 7.92 | 10.83 | 9.37 |
| validated EM + judge b=40 [info_gain, may reject all] | **3.20** | 6.17 | 4.68 |

**Ablations at the largest budget**

| method | spider | bird | **mean** |
|---|---|---|---|
| ABL  - e discount (votes undiscounted) | 5.66 | 10.04 | 7.85 |
| ABL  gamma mode = model_wrong | 3.27 | 8.86 | 6.06 |
| ABL  gamma mode = pseudo_wrong | 3.07 | 7.63 | 5.35 |
| ABL  update rule = pooled counts | 6.49 | 6.69 | 6.59 |
| ABL  temp from latest item only | 2.84 | 6.93 | 4.88 |
| ABL  refresh on overrule only | **2.30** | **5.94** | **4.12** |
| ABL  clamp confirmed items too | 6.29 | 12.79 | 9.54 |
| ABL  expert accuracy 0.80 | 3.06 | 7.88 | 5.47 |
| ABL  expert accuracy 0.80, may reject all | 3.06 | 7.88 | 5.47 |

### Text2SQL -- protocol B (target holdout)

**Estimators and the judge budget**

| method | spider | bird | sqlflow | bird_minidev | spider2local | **mean** |
|---|---|---|---|---|---|---|
| prior only (no target data) | 5.50 | 6.50 | 2.14 | 2.86 | 6.61 | 4.72 |
| PoolEval-SQL (current paper) | 13.94 | 17.67 | 6.07 | 10.22 | 12.88 | 12.15 |
| binary agreement EM (gamma=1) | 21.56 | 13.07 | 19.22 | 20.98 | 16.43 | 18.25 |
| collision EM (group gamma, .tex) | 12.66 | 17.89 | 7.54 | 5.49 | 7.13 | 10.14 |
| validated EM (no judge) | 10.52 | 9.79 | 2.70 | **2.26** | 6.94 | 6.44 |
| validated EM + judge b=5 [info_gain] | 9.63 | 7.98 | 4.35 | 2.97 | 5.24 | 6.04 |
| validated EM + judge b=5 [entropy] | 9.74 | 7.98 | 4.35 | 2.97 | 5.24 | 6.06 |
| validated EM + judge b=5 [random] | 6.60 | 9.79 | 2.70 | **2.26** | 6.94 | 5.66 |
| validated EM + judge b=5 [info_gain, may reject all] | 8.45 | 10.86 | **1.90** | 4.61 | 6.45 | 6.45 |
| validated EM + judge b=10 [info_gain] | 9.63 | 8.73 | 4.00 | 4.05 | 5.24 | 6.33 |
| validated EM + judge b=10 [entropy] | 9.74 | 8.73 | 4.00 | 3.04 | 5.24 | 6.15 |
| validated EM + judge b=10 [random] | 6.60 | 13.69 | 4.81 | 5.13 | 7.02 | 7.45 |
| validated EM + judge b=10 [info_gain, may reject all] | 8.88 | 11.17 | 3.67 | 5.44 | 3.19 | 6.47 |
| validated EM + judge b=20 [info_gain] | 9.57 | 8.73 | 3.93 | 4.96 | 5.24 | 6.49 |
| validated EM + judge b=20 [entropy] | 9.62 | 8.73 | 3.93 | 5.04 | 5.24 | 6.51 |
| validated EM + judge b=20 [random] | 6.60 | 13.69 | 2.67 | 6.00 | 7.02 | 7.20 |
| validated EM + judge b=20 [info_gain, may reject all] | 6.89 | 8.95 | 2.92 | 3.99 | **2.13** | 4.97 |
| validated EM + judge b=40 [info_gain] | 7.21 | 11.15 | 3.25 | 5.81 | 5.24 | 6.53 |
| validated EM + judge b=40 [entropy] | 7.22 | 11.15 | 3.27 | 5.81 | 5.24 | 6.54 |
| validated EM + judge b=40 [random] | 10.78 | 14.18 | 2.91 | 3.44 | 7.02 | 7.67 |
| validated EM + judge b=40 [info_gain, may reject all] | **1.64** | **4.51** | 4.29 | 3.73 | **2.13** | **3.26** |

**Ablations at the largest budget**

| method | spider | bird | sqlflow | bird_minidev | spider2local | **mean** |
|---|---|---|---|---|---|---|
| ABL  - e discount (votes undiscounted) | 7.51 | 11.73 | 3.50 | 5.95 | 5.24 | 6.79 |
| ABL  gamma mode = model_wrong | 4.32 | 10.10 | 3.29 | 5.76 | **5.01** | 5.70 |
| ABL  gamma mode = pseudo_wrong | 3.18 | 7.93 | 4.33 | 5.50 | 5.71 | 5.33 |
| ABL  update rule = pooled counts | 7.76 | 9.42 | 2.63 | **2.26** | 6.03 | 5.62 |
| ABL  temp from latest item only | 2.86 | **5.06** | 4.44 | 4.22 | 5.24 | **4.36** |
| ABL  refresh on overrule only | **2.46** | 6.87 | 5.27 | 4.83 | 5.24 | 4.93 |
| ABL  clamp confirmed items too | 8.42 | 11.65 | 3.84 | 6.70 | 7.13 | 7.55 |
| ABL  expert accuracy 0.80 | 3.66 | 6.50 | **1.91** | 3.90 | 7.74 | 4.74 |
| ABL  expert accuracy 0.80, may reject all | 3.66 | 6.50 | **1.91** | 3.90 | 7.74 | 4.74 |

### Image classification

**Estimators and the judge budget**

| method | mnist->usps | mnist->svhn | **mean** |
|---|---|---|---|
| prior only (no target data) | 29.10 | 85.88 | 57.49 |
| PoolEval-SQL (current paper) | 6.23 | 56.67 | 31.45 |
| binary agreement EM (gamma=1) | 10.47 | 59.90 | 35.19 |
| collision EM (group gamma, .tex) | 5.17 | 56.22 | 30.69 |
| validated EM (no judge) | 10.91 | 58.03 | 34.47 |
| validated EM + judge b=5 [info_gain] | 17.92 | 53.42 | 35.67 |
| validated EM + judge b=5 [entropy] | 11.68 | 48.32 | 30.00 |
| validated EM + judge b=5 [random] | 10.91 | 43.19 | 27.05 |
| validated EM + judge b=5 [info_gain, may reject all] | 17.92 | 48.69 | 33.30 |
| validated EM + judge b=10 [info_gain] | 7.18 | 51.31 | 29.25 |
| validated EM + judge b=10 [entropy] | 9.70 | 48.86 | 29.28 |
| validated EM + judge b=10 [random] | **1.43** | 38.42 | 19.93 |
| validated EM + judge b=10 [info_gain, may reject all] | 7.18 | 47.83 | 27.51 |
| validated EM + judge b=20 [info_gain] | 5.28 | 49.27 | 27.28 |
| validated EM + judge b=20 [entropy] | 4.84 | 47.04 | 25.94 |
| validated EM + judge b=20 [random] | **1.43** | 33.07 | 17.25 |
| validated EM + judge b=20 [info_gain, may reject all] | 5.28 | 49.78 | 27.53 |
| validated EM + judge b=40 [info_gain] | 3.03 | 47.61 | 25.32 |
| validated EM + judge b=40 [entropy] | 3.44 | 37.06 | 20.25 |
| validated EM + judge b=40 [random] | **1.43** | **19.91** | **10.67** |
| validated EM + judge b=40 [info_gain, may reject all] | 3.03 | 46.73 | 24.88 |

**Ablations at the largest budget**

| method | mnist->usps | mnist->svhn | **mean** |
|---|---|---|---|
| ABL  - e discount (votes undiscounted) | 10.60 | 48.55 | 29.57 |
| ABL  gamma mode = model_wrong | 4.00 | 45.70 | 24.85 |
| ABL  gamma mode = pseudo_wrong | 9.09 | **39.88** | **24.49** |
| ABL  update rule = pooled counts | 2.77 | 55.13 | 28.95 |
| ABL  temp from latest item only | 19.52 | 44.24 | 31.88 |
| ABL  refresh on overrule only | 16.81 | 45.52 | 31.16 |
| ABL  clamp confirmed items too | **2.69** | 47.61 | 25.15 |
| ABL  expert accuracy 0.80 | 5.60 | 46.98 | 26.29 |
| ABL  expert accuracy 0.80, may reject all | 5.60 | 46.98 | 26.29 |

### Node classification

**Estimators and the judge budget**

| method | AC | AD | CA | CD | DA | DC | **mean** |
|---|---|---|---|---|---|---|---|
| prior only (no target data) | 6.89 | 13.89 | 17.23 | 13.72 | 19.01 | 9.89 | 13.44 |
| PoolEval-SQL (current paper) | 10.54 | 19.52 | 17.29 | 16.56 | 19.53 | 13.01 | 16.07 |
| binary agreement EM (gamma=1) | 18.03 | 26.21 | 24.49 | 23.63 | 28.49 | 21.14 | 23.66 |
| collision EM (group gamma, .tex) | 13.53 | 22.39 | 21.27 | 20.02 | 24.11 | 16.54 | 19.64 |
| validated EM (no judge) | 9.12 | 16.52 | 16.96 | 15.60 | 19.00 | 12.04 | 14.87 |
| validated EM + judge b=5 [info_gain] | 15.62 | 13.40 | 13.24 | 12.64 | 17.99 | 9.55 | 13.74 |
| validated EM + judge b=5 [entropy] | 10.84 | 13.93 | 13.24 | 12.02 | 17.38 | 10.20 | 12.94 |
| validated EM + judge b=5 [random] | 9.12 | 15.92 | 15.03 | 13.41 | 13.80 | 12.10 | 13.23 |
| validated EM + judge b=5 [info_gain, may reject all] | 15.62 | 13.40 | 13.24 | 12.64 | 17.99 | 9.55 | 13.74 |
| validated EM + judge b=10 [info_gain] | 7.60 | 14.50 | 11.58 | 11.05 | 13.62 | 10.76 | 11.52 |
| validated EM + judge b=10 [entropy] | 7.15 | 13.82 | 11.58 | 9.15 | 17.26 | 10.99 | 11.66 |
| validated EM + judge b=10 [random] | 8.38 | 15.76 | 11.72 | 13.63 | 14.19 | 11.23 | 12.48 |
| validated EM + judge b=10 [info_gain, may reject all] | 7.60 | 14.50 | 12.14 | 11.05 | 12.17 | 10.76 | 11.37 |
| validated EM + judge b=20 [info_gain] | 6.84 | 14.74 | 9.71 | 9.50 | 13.40 | 9.45 | 10.61 |
| validated EM + judge b=20 [entropy] | 6.81 | 13.79 | 9.92 | 9.71 | 11.31 | 9.17 | 10.12 |
| validated EM + judge b=20 [random] | 8.47 | 14.77 | 11.71 | 11.96 | 14.19 | 11.41 | 12.09 |
| validated EM + judge b=20 [info_gain, may reject all] | 9.00 | 14.63 | 12.18 | 8.88 | 12.96 | 9.46 | 11.18 |
| validated EM + judge b=40 [info_gain] | **6.67** | 14.30 | 10.08 | 9.03 | 10.18 | 6.43 | 9.45 |
| validated EM + judge b=40 [entropy] | 6.73 | 14.64 | **9.33** | 9.21 | 11.15 | **5.58** | 9.44 |
| validated EM + judge b=40 [random] | 8.52 | **11.22** | 11.99 | 11.74 | 15.89 | 11.33 | 11.78 |
| validated EM + judge b=40 [info_gain, may reject all] | 7.12 | 12.73 | 9.52 | **8.87** | **10.06** | 6.35 | **9.11** |

**Ablations at the largest budget**

| method | AC | AD | CA | CD | DA | DC | **mean** |
|---|---|---|---|---|---|---|---|
| ABL  - e discount (votes undiscounted) | 7.55 | 15.81 | 10.77 | 11.04 | 11.55 | 8.29 | 10.83 |
| ABL  gamma mode = model_wrong | **6.09** | 12.52 | 9.48 | **8.13** | **8.58** | 6.77 | **8.59** |
| ABL  gamma mode = pseudo_wrong | 8.63 | 11.84 | 11.63 | 8.41 | 16.81 | 8.52 | 10.97 |
| ABL  update rule = pooled counts | 8.29 | 15.68 | 14.43 | 14.38 | 16.50 | 11.03 | 13.38 |
| ABL  temp from latest item only | 7.72 | 12.43 | 13.50 | 9.20 | 10.83 | 9.89 | 10.60 |
| ABL  refresh on overrule only | 16.18 | **8.26** | 11.96 | 9.92 | 15.44 | 9.56 | 11.88 |
| ABL  clamp confirmed items too | 6.75 | 13.97 | **8.98** | 9.07 | 11.18 | **6.30** | 9.38 |
| ABL  expert accuracy 0.80 | 6.76 | 11.56 | 9.20 | 10.93 | 11.10 | 6.54 | 9.35 |
| ABL  expert accuracy 0.80, may reject all | 6.76 | 11.56 | 9.20 | 10.93 | 11.10 | 6.54 | 9.35 |

