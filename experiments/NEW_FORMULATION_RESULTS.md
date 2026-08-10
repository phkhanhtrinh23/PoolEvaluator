# Closed-Form New Formulation: Implementation and Results

## Executive conclusion

The new pseudo-label agreement formulation is now implemented alongside the old
PoolEval estimator. It uses the requested two-stage pipeline:

1. apply the graded result-equivalence kernel;
2. use the old method's prior-weighted, provenance-discounted, verifier-assisted
   class score to select one pseudo-label per item;
3. construct the binary agreement matrix
   \(C_{ji}=\mathbf{1}(r_{ji}=\hat y_i)\);
4. estimate classifier accuracies \(\alpha_j\) and pseudo-label quality \(\beta\)
   with the new formulation's closed-form EM equations.

On the repository's paired simulator experiments, the new formulation is **not
better overall**. It slightly improves average ranking metrics, but substantially
worsens absolute accuracy estimation:

| Main result, 8 paired seeds | Old PoolEval | New formulation | Better |
| --- | ---: | ---: | --- |
| Accuracy MAE (points) ↓ | **3.192** | 11.586 | Old |
| Pairwise flip rate ↓ | 0.062 | **0.053** | New |
| Kendall \(\tau\) ↑ | 0.870 | **0.889** | New |
| Top-1 selection ↑ | 0.500 | **0.625** | New |
| Top-3 overlap ↑ | 0.833 | 0.833 | Tie |

The new estimator is therefore potentially useful as a **ranker**, but its raw
\(\alpha_j\) values should not currently be interpreted as calibrated test-set
accuracies.

## Implementation

The implementation is in `pooleval/new_formulation.py` and exposes:

- `agreement_em(C, alpha_init, beta_init=None)`: the standalone closed-form EM;
- `NewFormulationPoolEval(cfg)`: the complete kernel → old-score pseudo-label →
  binary agreement → closed-form EM pipeline.

The existing `PoolEval` code is unchanged.

### Pseudo-label construction

The old estimator is evaluated once on the graded-kernel observations. Its latent
class posterior is built from:

- per-model prior accuracy from the labeled train/meta dataset;
- the provenance-group discount for repeated same-group answers;
- the selected graded kernel level;
- the execution verifier bonus.

The maximum-posterior class is fixed as \(\hat y_i\). Supplying the old output to
the new estimator avoids rerunning the scorer and guarantees that the old/new
comparison uses exactly the same observation and pseudo-label source.

In the simulator, `run.prior` is the proxy for accuracy measured on a labeled
train/meta dataset. `run.true_acc` is held out and used only for reporting test
metrics. For real data, callers must populate `run.prior` from the training or
pre-built meta-dataset; the new estimator never computes this prior from test gold.

### Closed-form inference

The E-step is

\[
\tau_i^j
=\sigma\left[
\log\frac{\alpha_j}{1-\alpha_j}
+(2C_i^j-1)\log\frac{\beta}{1-\beta}
\right].
\]

The M-step uses the exact closed-form updates

\[
\alpha_j \leftarrow \frac{1}{N}\sum_i\tau_i^j,
\]

\[
\beta \leftarrow \frac{1}{NJ}\sum_{i,j}
\left[\tau_i^jC_i^j+(1-\tau_i^j)(1-C_i^j)\right].
\]

There is no numerical maximizer or gradient descent. “Closed form” applies to each
coordinate update, not to the full fixed point: EM still iterates because
\(\tau,\alpha,\beta\) depend on one another.

Probability clipping is used only to avoid `log(0)`. The implementation records the
observed-data log-likelihood, and the tests verify that it is non-decreasing.

## Experimental protocol

The comparison mirrors the existing RQ1–RQ8 simulator conditions in
`experiments/run_new_formulation_comparison.py`:

- 8 paired seeds;
- identical simulated pool for both estimators within each condition;
- identical kernel random draw;
- identical train/meta accuracy prior;
- no test-label tuning;
- old output reused as the new method's pseudo-label source.

The full machine-readable output is written to
`results/new_formulation_comparison.json` when the script runs.

## RQ1: main accuracy and ranking comparison

The old method remains far better calibrated, while the new formulation makes
slightly fewer pairwise ranking mistakes. The Top-1 difference is one additional
successful seed out of eight, so it should not be treated as conclusive with this
small sample.

The new formulation's mean diagnostics were:

| Diagnostic | Mean |
| --- | ---: |
| Actual pseudo-label accuracy | 0.835 |
| Estimated \(\beta\) | 0.696 |
| Absolute \(\beta\) error | 0.138 |
| EM iterations | 170.8 |

Seven runs converged before the 200-iteration cap; one reached the cap. Although
each M-step is closed form, convergence is not especially fast under the stringent
\(10^{-8}\) stopping tolerance.

The 13.8-point mean error in \(\beta\) is consistent with the poor absolute
accuracy calibration. The fitted binary model does not recover the actual quality
of pseudo-labels selected from multiclass SQL result classes.

## RQ2: pseudo-label scorer ablations

These ablations change components used to construct the pseudo-label. The new EM
continues to initialize \(\alpha\) from `run.prior`, because the new formulation
explicitly requires the train/meta accuracy estimate.

| Pseudo-label construction | Old MAE | New MAE | Old flip | New flip |
| --- | ---: | ---: | ---: | ---: |
| Full | **3.19** | 11.59 | 0.063 | **0.053** |
| No old-method seen-prior contribution | **3.54** | 11.64 | **0.042** | 0.047 |
| No execution verifier | **4.48** | 13.70 | **0.095** | 0.184 |
| No provenance correlation discount | **3.82** | 12.76 | 0.076 | **0.055** |
| Exact/LA0 kernel | **8.37** | 19.94 | 0.102 | 0.102 |
| Remove all old scorer components | **7.63** | 18.10 | **0.117** | 0.121 |

The verifier and graded kernel are essential for the new method because an error in
the fixed pseudo-label is propagated into every row of \(C\). Removing the verifier
raises new-method flip rate from 0.053 to 0.184. Removing graded equivalence raises
new-method MAE from 11.59 to 19.94.

## RQ3: provenance-correlated errors

| Collusion | Old MAE | New MAE | Old flip | New flip |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | **3.84** | 12.22 | 0.062 | **0.057** |
| 0.20 | **3.47** | 11.61 | 0.062 | **0.051** |
| 0.40 | **3.32** | 11.35 | 0.081 | **0.051** |
| 0.60 | **3.16** | 11.99 | **0.051** | 0.064 |
| 0.80 | **2.87** | 10.78 | 0.070 | **0.053** |
| 0.95 | **2.50** | 10.74 | 0.089 | **0.083** |

The new EM itself assumes conditionally independent binary agreements. It receives
some indirect protection from correlation because the old pseudo-label selector
discounts same-provenance votes before \(C\) is constructed. This is enough to keep
ranking competitive in most conditions, but it does not repair absolute
calibration.

## RQ4: graded kernel ladder

| Kernel | Precision | Recall | Old MAE | New MAE | Old flip | New flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LA0 | 0.999 | 0.549 | **8.37** | 19.94 | 0.102 | 0.102 |
| LA1 | 0.995 | 0.870 | **2.18** | 10.28 | 0.076 | **0.059** |
| LA2 | 1.000 | 0.940 | **3.19** | 11.59 | 0.063 | **0.053** |

Both methods are damaged by LA0's false disagreements, but the binary formulation
is much more sensitive because every missed equivalence directly changes agreement
entries in \(C\).

LA1 happens to outperform LA2 in old-method MAE in this simulator configuration.
That is a result of the configured precision/recall perturbation and eight-seed
sample, not evidence that canonicalization is generally preferable to
multi-instance execution.

## RQ5: pool-size scaling

| Models | Old MAE | New MAE | Old flip | New flip | Old Kendall | New Kendall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | **4.48** | 10.61 | 0.088 | **0.063** | 0.825 | **0.875** |
| 8 | **3.54** | 11.22 | 0.063 | 0.063 | 0.868 | 0.868 |
| 12 | **3.19** | 11.59 | 0.063 | **0.053** | 0.870 | **0.889** |
| 16 | **2.74** | 11.69 | 0.061 | **0.060** | 0.872 | **0.874** |
| 20 | **2.96** | 11.67 | 0.064 | **0.056** | 0.868 | **0.883** |

Adding models improves the old method's accuracy calibration. The new method stays
near 11–12 MAE points, showing that more agreement observations do not resolve the
binary model's level error. Its relative ranking remains strong.

## RQ6: unlabeled-item budget

| Test items | Old MAE | New MAE | Old flip | New flip |
| ---: | ---: | ---: | ---: | ---: |
| 100 | **3.62** | 11.83 | 0.125 | **0.095** |
| 250 | **3.48** | 11.49 | 0.078 | **0.061** |
| 500 | **3.00** | 10.76 | 0.047 | 0.047 |
| 1,000 | **3.52** | 11.64 | 0.044 | **0.040** |
| 1,500 | **3.54** | 10.88 | 0.051 | **0.049** |

More test items reduce ranking variance for both methods, but the new method's MAE
does not approach the old estimator's MAE. This indicates model misspecification or
identifiability error rather than insufficient sample size.

## RQ7: initialization sensitivity

The meta initialization uses each model's train/meta accuracy. The cold
initialization sets every \(\alpha_j=0.6\). Both use the same fixed pseudo-labels.

| Items | Meta MAE | Cold MAE | Meta flip | Cold flip | Meta iterations | Cold iterations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | **11.83** | 12.67 | **0.095** | 0.212 | 190.8 | **170.6** |
| 250 | 11.49 | **11.06** | 0.061 | **0.059** | 191.1 | **178.1** |
| 500 | 10.76 | **10.33** | 0.047 | **0.045** | 181.6 | **178.4** |
| 1,000 | 11.64 | **11.17** | 0.040 | **0.038** | 172.8 | **162.0** |
| 1,500 | 10.88 | **10.65** | 0.049 | 0.049 | 179.6 | **177.4** |

Initialization matters at the smallest sample size, but it does not fix the
absolute calibration issue. The cold start often terminates sooner and sometimes
has slightly lower MAE, which is another warning that the likelihood has a weakly
identified or unsuitable accuracy level for these multiclass observations.

## RQ8: workload drift

This comparison refits both estimators on every drifting batch, corresponding to
the periodic policy in the old RQ8 experiment.

| Batch | Old flip | New flip |
| ---: | ---: | ---: |
| 1 | 0.097 | 0.097 |
| 2 | **0.066** | 0.152 |
| 3 | **0.083** | 0.093 |
| 4 | 0.072 | **0.055** |
| 5 | 0.080 | **0.074** |
| 6 | 0.064 | **0.057** |
| 7 | 0.081 | 0.081 |
| 8 | **0.080** | 0.081 |
| 9 | 0.061 | **0.036** |
| 10 | **0.059** | 0.061 |

There is no consistent drift winner. The new method is markedly worse in batch 2,
better in batches 4, 6, and 9, and otherwise close.

## Why absolute accuracy degrades

The evidence points to the new formulation's conditional-agreement assumption:

\[
P(C=1\mid Z=0)=1-\beta.
\]

This is appropriate for binary labels when “classifier wrong” and “pseudo-label
wrong” force both to choose the same alternative. Text-to-SQL has many distinct
wrong SQL queries and result sets. A model and the pseudo-label can both be wrong
without agreeing, so \(1-\beta\) is not the correct probability of agreement given
model error.

The model consequently explains the observed binary agreement with distorted
values of \(\alpha\) and \(\beta\). The actual pseudo-label accuracy is 0.835, but
the mean fitted \(\beta\) is only 0.696. Since \(\alpha\) and \(\beta\) jointly
explain the same agreement rate, this level error propagates into model accuracy
estimates. Increasing \(N\) cannot correct a misspecified likelihood.

The new ranking can nevertheless be good because agreement with a strong fixed
pseudo-label remains informative about the *relative* order of models even when the
absolute probabilistic mapping is wrong.

## Recommendation

Keep the new implementation as an experimental ranker, but do not replace old
PoolEval for accuracy estimation. The next mathematically justified extension
should model the probability that two wrong multiclass results coincide, for
example with a parameter \(\gamma=P(r=\hat y\mid Z=0,\hat y\ne y)\), rather than
forcing it to equal one. A provenance-dependent \(\gamma_g\) could additionally
represent shared wrong answers from related models.

Any calibration layer should be fit only on the labeled train/meta dataset. It
must not use test truth merely to make these simulator results look better.

## Reproduction

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest -q
.venv/bin/python experiments/run_new_formulation_comparison.py --seeds 8
```

The test suite checks the equations directly, validates pseudo-label construction,
and verifies non-decreasing observed-data likelihood.
