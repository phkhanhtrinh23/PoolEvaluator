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

### Collision-aware binary formulation for case 3

The binary observation $C_i^j$ can still be used. The correction is to distinguish
“the model is wrong” from “the model is wrong and happens to produce the same wrong
table as the pseudo-label.” Define a wrong-result collision parameter
$\gamma_j=P(r_i^j=\hat y_i\mid Z_i^j=0,\hat y_i\ne y_i)$.

#### Step 1: variables and observations

For each target item $i$ and model $j$:

- $r_i^j$ is the executed result table produced by model $j$;
- $y_i$ is the unknown correct result table;
- $\hat y_i$ is the fixed pseudo-label selected by the old
  kernel/prior/provenance/verifier scorer;
- $C_i^j=\mathbf{1}(r_i^j=\hat y_i)$ is observed binary agreement;
- $Z_i^j=\mathbf{1}(r_i^j=y_i)$ is latent model correctness;
- $H_i=\mathbf{1}(\hat y_i=y_i)$ is pseudo-label correctness;
- $\alpha_j=P(Z_i^j=1)$ is target accuracy for model $j$;
- $\beta=P(H_i=1)$ is target pseudo-label quality;
- $\gamma_g$ is the probability that a wrong model in provenance group $g$
  produces exactly the same wrong table as a wrong pseudo-label.

The target estimator observes only $C$, model groups, fixed source statistics, and
the pseudo-labels. Target $Z$, $H$, $y$, and gold execution accuracy are withheld.

#### Step 2: corrected generative assumptions

The model-correctness prior is $Z_i^j\mid\alpha_j\sim
\mathrm{Bernoulli}(\alpha_j)$. Pseudo-label correctness is
$H_i\mid\beta\sim\mathrm{Bernoulli}(\beta)$.

Conditional agreement is deterministic in three cases and probabilistic in the
fourth:

- $P(C_i^j=1\mid Z_i^j=1,H_i=1)=1$;
- $P(C_i^j=1\mid Z_i^j=0,H_i=1)=0$;
- $P(C_i^j=1\mid Z_i^j=1,H_i=0)=0$;
- $P(C_i^j=1\mid Z_i^j=0,H_i=0,g(j)=g)=\gamma_g$.

Marginalizing $H_i$ gives
$P(C_i^j=1\mid Z_i^j=1)=\beta$ and
$P(C_i^j=1\mid Z_i^j=0)=(1-\beta)\gamma_{g(j)}$.

Define $q_{ij}=(1-\beta)\gamma_{g(j)}$. Then the observation likelihood can be
written compactly as
$P(C_i^j\mid Z_i^j=1)=\beta^{C_i^j}(1-\beta)^{1-C_i^j}$ and
$P(C_i^j\mid Z_i^j=0)=q_{ij}^{C_i^j}(1-q_{ij})^{1-C_i^j}$.

#### Step 3: source-derived parameters and priors

Collision rates are estimated on labeled source/meta data using the same
pseudo-label selector as target inference:
$\hat\gamma_g=\frac{h_g+1}{n_g+2}$, where $n_g$ counts source cells for which the
model and pseudo-label are both wrong, and $h_g$ counts those cells where their
wrong executed tables are identical. The added 1 and 2 are Laplace smoothing.

Source model accuracy $\pi_j$ becomes an explicit prior
$\alpha_j\sim\mathrm{Beta}(1+s_j\pi_j,1+s_j(1-\pi_j))$, where $s_j$ is the
effective source sample size. Source pseudo-label accuracy $\beta_0$ becomes
$\beta\sim\mathrm{Beta}(1+s_\beta\beta_0,1+s_\beta(1-\beta_0))$.

The experiments freeze $\hat\gamma_g$, use $s_j\approx120$ from the stored source
prior variance, and set $s_\beta=120$. No target gold is used to construct these
quantities.

#### Step 4: log posterior optimized by EM

Ignoring constants, the expected complete-data log posterior is
$Q(\alpha,\beta)=\sum_{i,j}\tau_i^j[\log\alpha_j+C_i^j\log\beta+(1-C_i^j)\log(1-\beta)]+(1-\tau_i^j)[\log(1-\alpha_j)+C_i^j\log q_{ij}+(1-C_i^j)\log(1-q_{ij})]+\sum_j s_j[\pi_j\log\alpha_j+(1-\pi_j)\log(1-\alpha_j)]+s_\beta[\beta_0\log\beta+(1-\beta_0)\log(1-\beta)]$.

Here, $\tau_i^j=P(Z_i^j=1\mid C_i^j,\alpha_j,\beta,\gamma_{g(j)})$ is recomputed in
the E-step. EM alternates between updating $\tau$ and maximizing this objective over
$\alpha$ and $\beta$ with $\gamma$ fixed.

#### Step 5: E-step

For an agreement $C_i^j=1$, Bayes' rule gives
$\tau_i^j=\frac{\alpha_j\beta}{\alpha_j\beta+(1-\alpha_j)(1-\beta)\gamma_{g(j)}}$.

For a disagreement $C_i^j=0$, Bayes' rule gives
$\tau_i^j=\frac{\alpha_j(1-\beta)}{\alpha_j(1-\beta)+(1-\alpha_j)[1-(1-\beta)\gamma_{g(j)}]}$.

These two expressions are evaluated elementwise for the complete $M\times N$
agreement matrix.

#### Step 6: closed-form alpha M-step

Holding $\tau$ and $\beta$ fixed, differentiate $Q$ with respect to $\alpha_j$ and
set the derivative to zero. The anchored MAP update is
$\alpha_j^{new}=\frac{\sum_i\tau_i^j+s_j\pi_j}{N+s_j}$.

When $s_j=0$, this reduces to the original maximum-likelihood update
$\alpha_j^{new}=\frac{1}{N}\sum_i\tau_i^j$.

#### Step 7: numerical beta M-step

Holding $\tau$ and $\alpha$ fixed, define
$Q_\beta(\beta)=\sum_{i,j}\tau_i^j[C_i^j\log\beta+(1-C_i^j)\log(1-\beta)]+(1-\tau_i^j)[C_i^j\log((1-\beta)\gamma_{g(j)})+(1-C_i^j)\log(1-(1-\beta)\gamma_{g(j)})]+s_\beta[\beta_0\log\beta+(1-\beta_0)\log(1-\beta)]$.

Its derivative is
$\frac{\partial Q_\beta}{\partial\beta}=\sum_{i,j}\tau_i^j[\frac{C_i^j}{\beta}-\frac{1-C_i^j}{1-\beta}]+(1-\tau_i^j)[-\frac{C_i^j}{1-\beta}+\frac{(1-C_i^j)\gamma_{g(j)}}{1-(1-\beta)\gamma_{g(j)}}]+s_\beta[\frac{\beta_0}{\beta}-\frac{1-\beta_0}{1-\beta}]$.

There is no algebraic maximizer because $\beta$ appears inside both
$\log(1-\beta)$ and $\log(1-(1-\beta)\gamma_g)$. The implementation minimizes
$-Q_\beta(\beta)$ on $[10^{-6},1-10^{-6}]$ using bounded Brent optimization. This
is a deterministic one-dimensional search and does not require learning rates or
high-dimensional gradient descent.

#### Step 8: convergence

After each E/M cycle, compute
$\Delta=\max(\max_j|\alpha_j^{new}-\alpha_j^{old}|,|\beta^{new}-\beta^{old}|)$.
Stop when $\Delta<10^{-8}$ or after 200 iterations. Probability clipping prevents
undefined logarithms at 0 and 1. Tests verify that the regularized log posterior is
non-decreasing across iterations.

#### Step 9: complete target algorithm

1. Apply the graded execution-equivalence kernel to all target model outputs.
2. Run the old prior/provenance/verifier scorer and select its highest-scoring
   result class as $\hat y_i$.
3. Construct $C_i^j=\mathbf{1}(r_i^j=\hat y_i)$.
4. Load source-derived $\pi_j$, $s_j$, $\beta_0$, $s_\beta$, and $\gamma_g$.
5. Initialize $\alpha_j=\pi_j$ and $\beta=\beta_0$.
6. Compute $\tau_i^j$ with the corrected E-step.
7. Update every $\alpha_j$ with its closed-form anchored formula.
8. Update $\beta$ by bounded maximization of $Q_\beta$.
9. Repeat steps 6–8 until convergence.
10. Return $\alpha_j$ as estimated target model accuracies and rank models by
    decreasing $\alpha_j$.

Binary target observations alone cannot reliably identify $\beta$ and every
$\gamma_j$, because both parameters control agreement with a wrong pseudo-label.
Estimate collision rates on the labeled source/meta dataset, where correctness is
known, and freeze or strongly regularize them on the held-out target. A group-level
parameter is more stable than a separate parameter per model:
$\gamma_g=\frac{\left|\{(j,i):g(j)=g,Z_i^j=0,\hat y_i\ne y_i,r_i^j=\hat y_i\}\right|}{\left|\{(j,i):g(j)=g,Z_i^j=0,\hat y_i\ne y_i\}\right|}$.

The source procedure must use exactly the same pseudo-label selector as the target
procedure. The recommended workflow is:

1. run the old kernel/prior/provenance/verifier scorer on labeled source items;
2. select one source pseudo-label per item;
3. measure $\gamma_g$ from cases where both the model and pseudo-label are wrong;
4. freeze or place a strong source-derived prior on $\gamma_g$ for target inference;
5. retain a Beta prior on $\alpha_j$ derived from source accuracy instead of using
   source accuracy only as initialization;
6. estimate target $\alpha_j$ and $\beta$ with the corrected likelihood.

This solves case 3 probabilistically while retaining binary $C_i^j$. It does not
recover information already discarded by binarization, so keeping the full result
classes remains useful for estimating collision structure and provenance effects.

### Real results after solving case 3

The collision-aware estimator was evaluated on four real datasets using 10 real
models, 150 target questions per dataset, and 500 paired item-bootstrap resamples.
Because item-level train execution classes were not saved, each target's
$\gamma_g$ was estimated with a leakage-safe leave-one-dataset-out protocol using
the other three labeled real zoo artifacts. Each target's saved train/source
accuracy remained the explicit prior for $\alpha_j$.

| Dataset | Source prior MAE ↓ | Old MAE ↓ | Uncorrected binary MAE ↓ | Case-3 MAE ↓ | Old Kendall ↑ | Case-3 Kendall ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spider | **3.73** | 13.38 | 19.59 | 11.93 | **0.71** | 0.66 |
| SQLFlow | **3.82** | **11.28** | 16.38 | 12.01 | 0.72 | 0.72 |
| BIRD | **3.47** | 17.20 | 16.32 | 17.22 | 0.60 | **0.69** |
| BIRD-MiniDev | **2.48** | 13.80 | 20.90 | 3.85 | 0.58 | **0.63** |

The case-3 correction removes the catastrophic ranking reversal of the uncorrected
binary model. SQLFlow Kendall changes from $-0.67$ to $0.72$, BIRD from $-0.47$ to
$0.69$, and BIRD-MiniDev from $-0.49$ to $0.63$.

Compared directly with old PoolEval, the results are mixed. Case 3 improves Spider
point MAE by 1.45 points, is 0.73 points worse on SQLFlow, is effectively tied on
BIRD MAE while improving its point ranking, and improves BIRD-MiniDev point MAE by
9.95 points. The BIRD-MiniDev bootstrap interval is wide, so that large improvement
is not yet statistically stable.

The paired 95% intervals for `case 3 − old` MAE are Spider
$-1.57[-3.25,0.07]$, SQLFlow $+0.64[0.03,1.17]$, BIRD
$-0.06[-0.72,0.46]$, and BIRD-MiniDev $-8.85[-14.94,1.03]$. Only the small
SQLFlow degradation excludes zero.

The anchored target estimates were $\beta=0.922$ on Spider, $\beta=0.798$ on
SQLFlow, $\beta=0.842$ on BIRD, and $\beta=0.257$ on BIRD-MiniDev. These are MAP
parameters under the approximate model, not direct empirical pseudo-label
accuracies.

The correction is therefore useful but not a general victory over old PoolEval. It
repairs the original binary model's real-data ranking failure, but the source/train
prior remains the best absolute accuracy estimator on every dataset.

### LLM-judge pseudo-label updates after case-3 EM

The saved real `gpt-5-mini` verdicts were applied after the initial case-3 fit. If
the judge selected another candidate class, that class replaced $\hat y_i$. If it
rejected all candidates, a fresh absent class replaced $\hat y_i$, making
$C_i^j=0$ for every model on that item. The affected columns of $C$ were rebuilt and
case-3 EM was rerun at cumulative budgets 3, 6, 9, and 12.

| Dataset | Case-3 MAE ↓ | MAE after 12 verdicts ↓ | Case-3 Kendall ↑ | Kendall after 12 verdicts ↑ |
| --- | ---: | ---: | ---: | ---: |
| Spider | 11.93 | **11.00** | **0.66** | 0.52 |
| SQLFlow | 12.01 | **11.51** | 0.72 | 0.72 |
| BIRD | 17.22 | **16.49** | **0.69** | 0.60 |
| BIRD-MiniDev | 3.85 | **3.69** | 0.63 | 0.63 |

Conditional paired bootstrap intervals show stable MAE improvements on Spider,
SQLFlow, and BIRD, but no stable ranking improvement. The judge changed 7, 8, 11,
and 9 pseudo-labels respectively, and issued 0, 1, 2, and 4 “none” verdicts.

These verdicts were selected by the saved old-method active acquisition order. The
original databases and questions needed for new LLM calls are unavailable, so this
is a replay after case-3 EM rather than a new case-3-specific acquisition run. The
result supports trusting “none” detections and using softer confidence weights for
candidate picks: absolute calibration improves, but unreliable picks can damage
ranking and Top-1 selection.

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
- Do not use the uncorrected binary formulation; it reverses rankings on hard real
  datasets.
- Use the collision-aware case-3 formulation when binary $C_i^j$ is required, but
  do not claim it consistently beats old PoolEval yet.
- Prefer old PoolEval or the source prior until the case-3 improvement is validated
  on additional independent model pools.
- Keep train-derived accuracy as an explicit prior in the objective instead of
  using it only to initialize $\alpha_j$.
