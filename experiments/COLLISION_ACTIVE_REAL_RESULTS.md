# LLM-Judge Updates after Collision-Aware EM

## Conclusion

Applying 12 saved real `gpt-5-mini` verdicts after the initial case-3 EM lowers
absolute MAE on all four real datasets. Ranking does not consistently improve:
Spider and BIRD point Kendall decline, SQLFlow is unchanged, and BIRD-MiniDev loses
its initially correct Top-1 selection.

| Dataset | Case-3 MAE, budget 0 ↓ | Judge-updated MAE, budget 12 ↓ | Case-3 Kendall ↑ | Judge-updated Kendall ↑ |
| --- | ---: | ---: | ---: | ---: |
| Spider | 11.93 | **11.00** | **0.66** | 0.52 |
| SQLFlow | 12.01 | **11.51** | 0.72 | 0.72 |
| BIRD | 17.22 | **16.49** | **0.69** | 0.60 |
| BIRD-MiniDev | 3.85 | **3.69** | 0.63 | 0.63 |

The judge is useful as an absolute-level correction, especially when it rejects all
pool answers, but its candidate picks are too unreliable to guarantee better
ranking.

## Protocol

The requested update is performed in this order:

1. Run the old kernel/prior/provenance/verifier scorer and select one initial
   pseudo-label $\hat y_i$ per target item.
2. Construct $C_i^j=\mathbf{1}(r_i^j=\hat y_i)$.
3. Run collision-aware case-3 EM to convergence.
4. Take ambiguous items previously selected by the repository's real active
   pipeline.
5. Apply the saved `gpt-5-mini` verdict for each selected item.
6. If the judge selects another candidate, replace $\hat y_i$ with that candidate's
   executed-result class.
7. If the judge answers “none,” replace $\hat y_i$ with a fresh class absent from
   all model outputs. The complete column then becomes $C_i^j=0$ for every model.
8. Rebuild all affected columns of $C$.
9. Rerun collision-aware EM with the same leakage-free source $\gamma_g$ and source
   priors.
10. Repeat at cumulative judge budgets 3, 6, 9, and 12.

No target gold is used to change pseudo-labels. Gold execution classes are used only
after inference to compute metrics and judge diagnostics.

## Important selection limitation

The repository contains 12 saved genuine LLM verdicts per dataset, but the original
question/database mounts required to issue new judge calls are unavailable. The
saved verdicts were acquired in the ambiguity order selected by the old active
PoolEval pipeline, not a newly computed case-3 ambiguity order.

This experiment therefore answers:

> What happens when the existing real LLM corrections are applied after case-3 EM?

It does not yet answer whether a new acquisition function computed specifically
from case-3 posterior uncertainty would select better items. The bootstrap is also
conditional on the fixed judged item set and verdicts.

## Pseudo-label changes and judge behavior

| Dataset | Judged | Pseudo-labels changed | “None” verdicts | Correct “none” | Overall judge accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spider | 12 | 7 | 0 | 0 | 0.42 |
| SQLFlow | 12 | 8 | 1 | 1 | 0.08 |
| BIRD | 12 | 11 | 2 | 2 | 0.33 |
| BIRD-MiniDev | 12 | 9 | 4 | 2 | 0.17 |

The “none” behavior is the most trustworthy part: all SQLFlow and BIRD rejections
were correct, while two of four BIRD-MiniDev rejections were correct. Candidate
selection was much weaker, explaining why MAE can improve while ordering degrades.

## Budget curves

### Spider

| Judge budget | MAE ↓ | Flip ↓ | Kendall ↑ | Top-1 ↑ | $\beta$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.93 | **0.133** | **0.660** | 0 | 0.922 |
| 3 | 11.89 | 0.156 | 0.613 | 0 | 0.921 |
| 6 | 11.57 | 0.178 | 0.566 | 0 | 0.917 |
| 9 | 11.38 | 0.178 | 0.566 | 0 | 0.915 |
| 12 | **11.00** | 0.200 | 0.519 | 0 | 0.910 |

MAE improves monotonically, but ranking deteriorates as unreliable candidate
changes accumulate.

### SQLFlow

| Judge budget | MAE ↓ | Flip ↓ | Kendall ↑ | Top-1 ↑ | $\beta$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12.01 | 0.133 | 0.719 | 0 | 0.798 |
| 3 | 11.97 | 0.133 | 0.719 | 0 | 0.797 |
| 6 | 11.85 | 0.133 | 0.719 | 0 | 0.794 |
| 9 | **11.51** | 0.133 | 0.719 | 0 | 0.788 |
| 12 | **11.51** | 0.133 | 0.719 | 0 | 0.788 |

The judge lowers MAE without changing the point ranking.

### BIRD

| Judge budget | MAE ↓ | Flip ↓ | Kendall ↑ | Top-1 ↑ | $\beta$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 17.22 | **0.156** | **0.689** | 0 | 0.842 |
| 3 | 17.18 | **0.156** | **0.689** | 0 | 0.841 |
| 6 | 17.07 | **0.156** | **0.689** | 0 | 0.840 |
| 9 | 16.71 | 0.178 | 0.644 | 0 | 0.837 |
| 12 | **16.49** | 0.200 | 0.600 | 0 | 0.835 |

The first six verdicts improve MAE without affecting ranking. Later verdicts reduce
MAE further but harm the order.

### BIRD-MiniDev

| Judge budget | MAE ↓ | Flip ↓ | Kendall ↑ | Top-1 ↑ | $\beta$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.85 | 0.156 | 0.629 | **1** | 0.257 |
| 3 | 3.84 | 0.156 | 0.629 | **1** | 0.259 |
| 6 | 3.79 | 0.156 | 0.629 | **1** | 0.266 |
| 9 | 3.77 | 0.156 | 0.629 | 0 | 0.270 |
| 12 | **3.69** | 0.156 | 0.629 | 0 | 0.277 |

MAE improves slightly and Kendall is unchanged, but a later incorrect verdict
changes the Top-1 deployment choice from correct to incorrect.

## Conditional paired bootstrap

The experiment uses 400 paired item-bootstrap resamples per dataset, conditional on
the same saved judged items and verdicts. The following values are `budget 12 −
budget 0`.

| Dataset | ΔMAE | ΔFlip | ΔKendall |
| --- | ---: | ---: | ---: |
| Spider | **-0.91 [-1.72, -0.29]** | +0.015 [-0.089, 0.133] | -0.031 [-0.270, 0.182] |
| SQLFlow | **-0.50 [-1.01, -0.10]** | -0.001 [-0.067, 0.067] | +0.002 [-0.133, 0.135] |
| BIRD | **-0.72 [-1.41, -0.15]** | -0.003 [-0.089, 0.089] | +0.007 [-0.178, 0.182] |
| BIRD-MiniDev | -0.22 [-0.79, 0.38] | +0.003 [-0.044, 0.067] | -0.006 [-0.133, 0.092] |

The MAE improvement excludes zero on Spider, SQLFlow, and BIRD. No ranking change
excludes zero. BIRD-MiniDev's small MAE improvement is uncertain.

## Interpretation

Judge updates affect all models on a selected item simultaneously:

- selecting another candidate changes which models receive $C_i^j=1$;
- rejecting all candidates sets $C_i^j=0$ for every model;
- rerunning EM propagates those changes through $\tau$, $\alpha$, and $\beta$.

The results show that this mechanism improves absolute calibration. However, the
saved judge's low candidate-pick accuracy can move closely ranked models in the
wrong direction. A safer policy should:

1. trust “none” verdicts more strongly than candidate picks;
2. use soft/confidence-weighted updates for candidate choices;
3. stop before later low-confidence verdicts damage ranking;
4. recompute acquisition scores from the case-3 posterior when the original data
   mounts become available.

## Reproduction

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m zoo.collision_active_real --bootstrap 400
```

Machine-readable output is written to `results/collision_active_real.json`.
