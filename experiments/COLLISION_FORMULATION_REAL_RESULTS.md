# Case-3 Collision-Aware Formulation on Real Text-to-SQL Data

## Conclusion

Adding the wrong-result collision parameter fixes the most serious failure of the
uncorrected binary formulation: it no longer reverses model rankings on the harder
real datasets. It does **not consistently beat old PoolEval**, however.

| Dataset | Train/source prior MAE ↓ | Old MAE ↓ | Uncorrected binary MAE ↓ | Case-3 MAE ↓ | Old Kendall ↑ | Case-3 Kendall ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spider | **3.73** | 13.38 | 19.59 | 11.93 | **0.71** | 0.66 |
| SQLFlow | **3.82** | **11.28** | 16.38 | 12.01 | 0.72 | 0.72 |
| BIRD | **3.47** | 17.20 | 16.32 | 17.22 | 0.60 | **0.69** |
| BIRD-MiniDev | **2.48** | 13.80 | 20.90 | 3.85 | 0.58 | **0.63** |

The case-3 model is much better than the uncorrected binary model on real ranking,
but the simple source/train prior remains the best absolute accuracy estimator on
all four datasets.

## What was implemented

The corrected binary model defines
$\gamma_g=P(r_i^j=\hat y_i\mid Z_i^j=0,\hat y_i\ne y_i,g(j)=g)$.

The observation probabilities become
$P(C_i^j=1\mid Z_i^j=1)=\beta$ and
$P(C_i^j=1\mid Z_i^j=0)=(1-\beta)\gamma_{g(j)}$.

For $C_i^j=1$, the E-step uses
$\tau_i^j=\frac{\alpha_j\beta}{\alpha_j\beta+(1-\alpha_j)(1-\beta)\gamma_{g(j)}}$.
For $C_i^j=0$, it uses
$\tau_i^j=\frac{\alpha_j(1-\beta)}{\alpha_j(1-\beta)+(1-\alpha_j)[1-(1-\beta)\gamma_{g(j)}]}$.

The source accuracy remains an explicit Beta anchor. If $s_j$ is its effective
sample size and $\pi_j$ is source accuracy, then the closed-form accuracy update is
$\alpha_j^{new}=\frac{\sum_i\tau_i^j+s_j\pi_j}{N+s_j}$.

This is equivalent to the MAP update under
$\alpha_j\sim\operatorname{Beta}(1+s_j\pi_j,1+s_j(1-\pi_j))$.
The stored source-prior variance implies $s_j\approx120$ for these artifacts.

## How beta is maximized

Introducing $\gamma_g$ removes the original closed-form $\beta$ update, but
$\beta$ remains a one-dimensional bounded problem. With fixed $\tau$, maximize

$Q(\beta)=\sum_{i,j}\tau_i^j[C_i^j\log\beta+(1-C_i^j)\log(1-\beta)]+(1-\tau_i^j)[C_i^j\log((1-\beta)\gamma_{g(j)})+(1-C_i^j)\log(1-(1-\beta)\gamma_{g(j)})]$.

The implementation maximizes $Q(\beta)$ on
$10^{-6}\le\beta\le1-10^{-6}$ using bounded scalar optimization. This is cheap:
there is only one optimized scalar per EM iteration, not a high-dimensional neural
or gradient-training problem.

Without a source anchor, the real likelihood often puts $\beta$ at 0 or 1. The
reported case-3 estimator therefore adds the leakage-safe source pseudo-label prior
$\beta\sim\operatorname{Beta}(1+s_\beta\beta_0,1+s_\beta(1-\beta_0))$ with
$s_\beta=120$. Its log-prior is included in the scalar objective.

## Real-data protocol

The original item-level train execution matrices are not stored, and their database
mounts are unavailable in this environment. Only aggregate target-specific train
accuracy priors remain. Consequently, target-specific train collision rates cannot
be reconstructed without regenerating the artifacts.

The experiment uses a leakage-safe leave-one-dataset-out protocol:

1. hold out one real dataset as the target;
2. use the other three labeled real zoo artifacts as the collision meta-dataset;
3. run exactly the same old pseudo-label selector on those source artifacts;
4. estimate each $\gamma_g$ with Laplace smoothing;
5. estimate source pseudo-label quality $\beta_0$;
6. freeze $\gamma_g$ on the target;
7. use the target's separately saved train/source accuracy as the explicit
   $\alpha_j$ prior;
8. estimate target $\alpha_j$ and $\beta$ without using target gold;
9. use target gold only after inference for metrics.

All four targets have 10 real models and 150 executed target questions. Spider and
BIRD use official train-derived accuracy priors. SQLFlow and BIRD-MiniDev use the
repository's deterministic disjoint labeled source partitions.

The full experiment uses 500 paired target-item bootstrap resamples per dataset.
The source collision rates and source priors remain fixed in every resample.

## Source-estimated collision rates

The seven group collision rates differ across leave-one-out source pools:

| Target held out | Source pseudo-label quality $\beta_0$ | Source $\gamma_g$ range |
| --- | ---: | ---: |
| Spider | 0.423 | 0.408–0.603 |
| SQLFlow | 0.520 | 0.517–0.659 |
| BIRD | 0.531 | 0.424–0.653 |
| BIRD-MiniDev | 0.573 | 0.471–0.656 |

The source collision estimates are based only on cells where both the model and
pseudo-label are wrong. They capture how frequently a wrong model produces exactly
the same wrong executed table as the wrong pseudo-label.

## Point results

| Dataset | Method | MAE ↓ | Flip ↓ | Kendall ↑ | Top-1 ↑ | Top-3 ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Spider | Old | 13.38 | **0.111** | **0.707** | 0 | 0.667 |
|  | Uncorrected binary | 19.59 | **0.111** | 0.691 | 0 | 0.667 |
|  | Case 3 | **11.93** | 0.133 | 0.660 | 0 | 0.667 |
| SQLFlow | Old | **11.28** | 0.133 | 0.719 | 0 | 1.000 |
|  | Uncorrected binary | 16.38 | 0.822 | -0.674 | 0 | 0.000 |
|  | Case 3 | 12.01 | 0.133 | 0.719 | 0 | 1.000 |
| BIRD | Old | **17.20** | 0.200 | 0.600 | 0 | 0.667 |
|  | Uncorrected binary | 16.32 | 0.733 | -0.467 | 0 | 0.000 |
|  | Case 3 | 17.22 | **0.156** | **0.689** | 0 | **1.000** |
| BIRD-MiniDev | Old | 13.80 | 0.178 | 0.582 | 0 | 0.667 |
|  | Uncorrected binary | 20.90 | 0.689 | -0.489 | 0 | 0.000 |
|  | Case 3 | **3.85** | **0.156** | **0.629** | **1** | 0.667 |

The correction clearly repairs the rank reversal. On SQLFlow it exactly recovers
old PoolEval's point ranking. On BIRD it improves Kendall from 0.60 to 0.69. On
BIRD-MiniDev it improves both point MAE and ranking substantially.

## Paired bootstrap uncertainty

The following intervals are `case 3 − old`. Negative MAE/Flip and positive Kendall
favor case 3.

| Dataset | ΔMAE | ΔFlip | ΔKendall |
| --- | ---: | ---: | ---: |
| Spider | -1.57 [-3.25, 0.07] | +0.009 [-0.111, 0.133] | -0.015 [-0.263, 0.234] |
| SQLFlow | **+0.64 [0.03, 1.17]** | 0.000 [-0.044, 0.044] | +0.001 [-0.091, 0.091] |
| BIRD | -0.06 [-0.72, 0.46] | -0.013 [-0.067, 0.022] | +0.026 [-0.047, 0.136] |
| BIRD-MiniDev | -8.85 [-14.94, 1.03] | -0.003 [-0.212, 0.178] | +0.007 [-0.360, 0.430] |

Only the small SQLFlow MAE degradation excludes zero. Spider's improvement nearly
excludes zero. BIRD is effectively tied. BIRD-MiniDev has a large point improvement
but a wide interval, so it is not yet a stable claim.

## Why anchoring beta matters

With fixed source $\gamma_g$ but no $\beta$ prior, scalar maximization produced
$\beta\approx1$ on Spider, SQLFlow, and BIRD, and $\beta\approx0$ on
BIRD-MiniDev. The resulting MAEs were 10.62, 10.97, 16.65, and 6.18 respectively,
but those boundary solutions are weakly identified and unsuitable as calibrated
pseudo-label-quality estimates.

Adding the source Beta anchor produced interior estimates:

| Dataset | Anchored target $\beta$ |
| --- | ---: |
| Spider | 0.922 |
| SQLFlow | 0.798 |
| BIRD | 0.842 |
| BIRD-MiniDev | 0.257 |

These values are still not direct estimates of actual target pseudo-label accuracy;
they are MAP parameters under a model whose assumptions remain approximate.

## Oracle-gamma diagnostic

For diagnosis only, collision rates were also estimated using target gold and then
fed back into target inference. This is test leakage and not a valid result.

| Dataset | Leave-one-out case-3 MAE | Oracle-target-$\gamma$ MAE |
| --- | ---: | ---: |
| Spider | 11.93 | 11.22 |
| SQLFlow | **12.01** | 12.94 |
| BIRD | 17.22 | **16.75** |
| BIRD-MiniDev | **3.85** | 5.85 |

Oracle target collision rates do not consistently improve performance. Therefore,
collision-rate transfer error is not the only remaining limitation. The shared
pseudo-label variable, conditional-independence assumptions, and compressed binary
observation still matter.

## Final assessment

- The case-3 correction is worthwhile: it fixes the uncorrected binary method's
  severe real-data ranking reversal.
- It does not establish that the binary estimator is better than old PoolEval.
- It is slightly worse on SQLFlow, tied on BIRD, suggestively better on Spider, and
  promising but unstable on BIRD-MiniDev.
- The train/source prior remains decisively best for absolute accuracy.
- Future artifacts should save item-level source execution classes so $\gamma_g$
  can be estimated on each target's actual train split rather than other datasets.
- A complete probabilistic model should represent shared pseudo-label correctness
  $H_i$ explicitly instead of marginalizing it independently for every model.

## Reproduction

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m zoo.collision_formulation_real --bootstrap 500
```

Machine-readable results are written to
`results/collision_formulation_real.json`.
