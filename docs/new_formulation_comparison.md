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

For classifier \(j\) and instance \(i\), the new formulation observes agreement
with a fixed pseudo-label:

\[
C_i^j = \mathbf{1}(r_i^j = \hat y_i).
\]

Its latent correctness variable is

\[
Z_i^j = \mathbf{1}(r_i^j = y_i),
\]

and its parameters are

\[
\alpha_j = P(Z_i^j=1), \qquad \beta = P(\hat y_i=y_i).
\]

Here, \(\alpha_j\) is classifier \(j\)'s accuracy and \(\beta\) is the global
pseudo-label quality.

### Closed-form EM updates

The E-step computes the posterior correctness probability

\[
\tau_i^j
=
\sigma\left(
\log\frac{\alpha_j}{1-\alpha_j}
+(2C_i^j-1)\log\frac{\beta}{1-\beta}
\right).
\]

The M-step has the closed-form maximizers

\[
\alpha_j^{\mathrm{new}}
=
\frac{1}{N}\sum_i \tau_i^j
\]

and

\[
\beta^{\mathrm{new}}
=
\frac{1}{NJ}\sum_{i,j}
\left[
\tau_i^j C_i^j
+(1-\tau_i^j)(1-C_i^j)
\right].
\]

Therefore, no gradient-based optimizer is required. EM must still iterate because
the posterior \(\tau\) depends on the current \(\alpha,\beta\), while the updated
\(\alpha,\beta\) depend on \(\tau\).

The document also presents variational Bayes updates using Beta priors. Those
updates are likewise coordinate-wise closed form, apart from evaluating standard
digamma functions.

## Current repository implementation

The repository does not construct \(C_i^j\), maintain \(Z_i^j\), estimate
\(\beta\), or use the posterior above.

Instead, `pooleval/latent.py` operates on the complete result-equivalence class
produced by every model. For each item, it constructs the candidate set

```python
item_classes = [np.unique(obs[:, i]) for i in range(N)]
```

and infers one shared multiclass latent answer \(z_i\).

### Current E-step

Each model contributes its clipped estimated accuracy to the result class it
produced. That contribution is discounted when models from the same provenance
group produce the same result:

\[
s_{ik}
=
\sum_{j:r_i^j=k}
\frac{a_j}{1+u_{g(j)}(n_{g(j),k}-1)}
+\lambda_v\mathbf{1}(v_i=k).
\]

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
| Observation | Binary agreement \(C_i^j\) with one pseudo-label | Multiclass result-equivalence class `obs[j, i]` |
| Latent variable | Per-model correctness \(Z_i^j\) | Shared answer class \(z_i\) |
| Model reliability | \(\alpha_j\) | `a[j]` |
| Pseudo-label quality | Global \(\beta\), estimated | No \(\beta\) |
| Item difficulty | None | `b[i]` |
| Correlated models | Conditional independence | Provenance-group loading `u[g]` |
| External reliability prior | Optional Beta prior in VB | Seen prior with inverse-variance fusion |
| Execution verifier | None | `verifier_guess[i]` |
| E-step | Closed-form Bernoulli posterior | Weighted multiclass vote and softmax |
| Accuracy update | Mean expected correctness | Agreement with hard inferred class |
| Optimization basis | Derived from a stated likelihood | Mixed consensus and moment-style updates |

## Why the implementations should not automatically be identical

The new formulation is simpler, but it assumes

\[
P(C=1\mid Z=0)=1-\beta.
\]

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

1. define how the fixed pseudo-label \(\hat y_i\) is selected;
2. construct the binary agreement matrix \(C_i^j\);
3. initialize and iteratively update \(\tau_i^j\), \(\alpha_j\), and \(\beta\)
   using the document's equations;
4. optionally implement the Beta-prior variational Bayes version;
5. explicitly address the multiclass wrong-answer case before applying the binary
   conditional-agreement assumption to Text-to-SQL;
6. compare both estimators on identical simulated and real observations.

This preserves a clean distinction between the exact closed-form model in the new
document and the richer, but currently heuristic, multiclass consensus estimator.
