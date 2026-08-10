# Comparison with the New Reliability Formulation

This document compares the estimator implemented in this repository with the
formulation in `new_formulation/Pool_Eval_Text2SQL_Crowd_Sourcing.pdf`.

## Conclusion

The repository does **not** implement or run the probabilistic model described in
the new formulation.

Both approaches alternate between inferring latent quantities and updating model
reliabilities, but their observations, latent variables, likelihoods, E-steps, and
M-steps are different. The new formulation has closed-form EM updates, while the
current `pooleval.latent.run_em` routine uses a multiclass, correlation-aware
weighted-consensus procedure with several heuristic update rules.

## New formulation

For classifier $j$ and instance $i$, the new formulation observes agreement with a
fixed pseudo-label: $C_i^j=\mathbf{1}(r_i^j=\hat y_i)$.

Its latent correctness variable is

$Z_i^j=\mathbf{1}(r_i^j=y_i)$,

and its parameters are

$\alpha_j=P(Z_i^j=1)$ and $\beta=P(\hat y_i=y_i)$.

Here, $\alpha_j$ is classifier $j$'s accuracy and $\beta$ is the global
pseudo-label quality.

### Closed-form EM updates

The E-step computes the posterior correctness probability
$\tau_i^j=\sigma\left(\log\frac{\alpha_j}{1-\alpha_j}+(2C_i^j-1)\log\frac{\beta}{1-\beta}\right)$.

The M-step has the closed-form maximizers
$\alpha_j^{\mathrm{new}}=\frac{1}{N}\sum_i\tau_i^j$ and
$\beta^{\mathrm{new}}=\frac{1}{NJ}\sum_{i,j}\left[\tau_i^jC_i^j+(1-\tau_i^j)(1-C_i^j)\right]$.

Therefore, no gradient-based optimizer is required. EM must still iterate because
the posterior $\tau$ depends on the current $\alpha,\beta$, while the updated
$\alpha,\beta$ depend on $\tau$.

The document also presents variational Bayes updates using Beta priors. Those
updates are likewise coordinate-wise closed form, apart from evaluating standard
digamma functions.

## Current repository implementation

The repository does not construct $C_i^j$, maintain $Z_i^j$, estimate $\beta$, or
use the posterior above.

Instead, `pooleval/latent.py` operates on the complete result-equivalence class
produced by every model. For each item, it constructs the candidate set

```python
item_classes = [np.unique(obs[:, i]) for i in range(N)]
```

and infers one shared multiclass latent answer $z_i$.

### Current E-step

Each model contributes its clipped estimated accuracy to the result class it
produced. That contribution is discounted when models from the same provenance
group produce the same result:

$s_{ik}=\sum_{j:r_i^j=k}\frac{a_j}{1+u_{g(j)}(n_{g(j),k}-1)}+\lambda_v\mathbf{1}(v_i=k)$.

The code applies a softmax to these class scores. This is a
reliability-weighted, provenance-discounted vote with an execution-verifier bonus;
it is not the Bernoulli posterior from the new formulation.

### Current M-step

The implementation converts the soft class posterior to a hard answer and measures
each model's agreement with it:

```python
latent_hat = np.array([max(post, key=post.get) for post in latent_post])
correct = (obs == latent_hat[None, :]).astype(float)
a_agree = correct.mean(axis=1)
```

It then either uses that agreement rate, uses an external prior, or combines the
two through inverse-variance fusion. Item difficulty is derived from hard
item-level agreement, and group loadings are derived from excess within-group
same-wrong-answer agreement.

These are explicit formulas, so the current implementation also does not use
gradient descent. However, they are not the closed-form maximizers derived in the
new formulation.

## Side-by-side comparison

| Aspect | New formulation | Current repository |
| --- | --- | --- |
| Observation | Binary agreement $C_i^j$ with one pseudo-label | Multiclass result-equivalence class `obs[j, i]` |
| Latent variable | Per-model correctness $Z_i^j$ | Shared answer class $z_i$ |
| Model reliability | $\alpha_j$ | `a[j]` |
| Pseudo-label quality | Global $\beta$, estimated | No $\beta$ |
| Item difficulty | None | `b[i]` |
| Correlated models | Conditional independence | Provenance-group loading `u[g]` |
| External reliability prior | Optional Beta prior in VB | Seen prior with inverse-variance fusion |
| Execution verifier | None | `verifier_guess[i]` |
| E-step | Closed-form Bernoulli posterior | Weighted multiclass vote and softmax |
| Accuracy update | Mean expected correctness | Agreement with hard inferred class |
| Optimization basis | Derived from a stated likelihood | Mixed consensus and moment-style updates |

## Why the implementations should not automatically be identical

The new formulation is simpler, but it assumes
$P(C=1\mid Z=0)=1-\beta$.

Thus, when a classifier is wrong, its probability of agreeing with the pseudo-label
is determined entirely by the probability that the pseudo-label is wrong. This is
natural in a binary setting where the two wrong/correct events imply the opposite
label. It is generally too strong for Text-to-SQL.

In Text-to-SQL, a model prediction and the pseudo-label may both be wrong while
producing different SQL queries and different result sets. The current repository
keeps result-class identities, allowing it to distinguish:

- two models producing the same wrong result;
- two models both being wrong but producing different results;
- a provenance group repeatedly producing the same error.

Collapsing these cases to binary pseudo-label agreement loses information used by
the repository's correlation and collusion handling.

## Objective and implementation caveat

The comments in `pooleval/latent.py` describe `run_em` as maximizing an anchored
posterior, but the implementation does not appear to be exact EM for one explicitly
implemented likelihood:

- the E-step uses accuracy directly as an additive vote rather than a model-derived
  log-likelihood ratio;
- the M-step discards the soft posterior and uses its `argmax`;
- the estimated difficulty `b` does not feed back into the E-step;
- group loading `u` is estimated using a separate excess-agreement statistic;
- prior fusion is performed after hard agreement rather than as the maximizer of
  the stated posterior.

Consequently, the usual EM guarantee that every iteration does not decrease a
single observed-data likelihood has not been established for this routine.
Disabling the prior, verifier, and correlation options in the repository's
`DawidSkene` baseline still calls this same routine; it does not turn it into the EM
algorithm from the new formulation.

## Recommended implementation direction

If the new formulation is intended to replace the current estimator, it should be
implemented as a separate estimator first. That implementation should:

1. define how the fixed pseudo-label $\hat y_i$ is selected;
2. construct the binary agreement matrix $C_i^j$;
3. initialize and iteratively update $\tau_i^j$, $\alpha_j$, and $\beta$
   using the document's equations;
4. optionally implement the Beta-prior variational Bayes version;
5. explicitly address the multiclass wrong-answer case before applying the binary
   conditional-agreement assumption to Text-to-SQL;
6. compare both estimators on identical simulated and real observations.

This preserves a clean distinction between the exact closed-form model in the new
document and the richer, but currently heuristic, multiclass consensus estimator.

## Is the new formulation better or worse?

The new formulation is **worse overall** than the old method on the saved real
Text-to-SQL zoo artifacts. It has one limited advantage: on the synthetic benchmark
it slightly improved average ranking metrics. That improvement did not generalize
to most real benchmarks.

### Real-data comparison

| Dataset | Old MAE ↓ | New MAE ↓ | Old Kendall ↑ | New Kendall ↑ |
| --- | ---: | ---: | ---: | ---: |
| Spider | **13.38** | 19.59 | **0.71** | 0.69 |
| SQLFlow | **11.28** | 16.38 | **0.72** | -0.67 |
| BIRD | 17.20 | **16.32** | **0.60** | -0.47 |
| BIRD-MiniDev | **13.80** | 20.90 | **0.58** | -0.49 |

The new method improves BIRD MAE by only 0.88 points, but its model ranking is
largely reversed. On the other three clean benchmarks, its absolute accuracy error
is also worse. The source/train prior by itself has lower MAE than both latent
estimators on all four datasets.

### Why the new formulation is worse

The first problem is the conditional-agreement assumption
$P(C=1\mid Z=0)=1-\beta$. This treats a wrong model and a wrong pseudo-label as
though they must agree. That can be reasonable for binary classification, where
there is only one alternative label, but Text-to-SQL has many possible wrong SQL
queries and result tables.

A model can use the wrong aggregation while the pseudo-label returns the wrong
rows. Both answers are wrong, but their executed results are different. Therefore,
$P(r=\hat y\mid r\ne y,\hat y\ne y)\ne1$ in general. The binary formulation
collapses all different wrong results into disagreement and cannot model their
collision probability correctly.

The second problem is an identification symmetry. The observed agreement
probability is
$P(C=1)=\alpha_j\beta+(1-\alpha_j)(1-\beta)$. Replacing the parameters with
$\alpha_j'=1-\alpha_j$ and $\beta'=1-\beta$ leaves the same observed agreement
probability:
$\alpha_j'\beta'+(1-\alpha_j')(1-\beta')=\alpha_j\beta+(1-\alpha_j)(1-\beta)$.

Binary agreement alone therefore cannot distinguish accurate models agreeing with
a good pseudo-label from inaccurate models agreeing with a bad pseudo-label. The
train-derived accuracy currently initializes $\alpha_j$, but it is not retained as
an explicit prior term in the new EM objective. The iterations can consequently
move away from the train-derived accuracy or settle in the complementary
orientation.

This behavior is visible on BIRD. Actual pseudo-label accuracy is $0.453$, fitted
$\beta=0.210$, old Kendall is $0.60$, and new Kendall is $-0.47$. The inferred
ordering is largely reversed even though the new method's aggregate MAE happens to
be slightly smaller.

### Why the old method is relatively better

The old method retains the full executed-result class for every model. It can
distinguish models returning the same result, models producing different wrong
results, repeated errors from one provenance group, and verifier-supported result
classes. The new method reduces all of this information to
$C_i^j\in\{0,1\}$, so important multiclass error structure is lost.

The old method is still poorly calibrated in absolute terms on these real pools.
The source/train prior is the best absolute estimator. However, the old method's
real-data rankings are substantially more reliable than those produced by the new
binary formulation.

### Practical conclusion

- Use the source/train prior when the goal is absolute execution-accuracy
  estimation.
- Prefer the old method over the current new formulation when the goal is ranking
  the real model pool.
- Do not replace the old estimator with the current binary formulation.
- Extend the new likelihood with a wrong-result collision parameter such as
  $\gamma=P(r=\hat y\mid Z=0,\hat y\ne y)$.
- Keep train-derived accuracy as an explicit prior in the objective instead of
  using it only to initialize $\alpha_j$.
