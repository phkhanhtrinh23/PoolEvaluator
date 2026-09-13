# `gamma_mode` for the headline configuration: the judge loop with $A_\mu$

The global tally across every method favours `model_wrong` (41 better, 19 worse, mean
−0.71). But the losses are not spread evenly: **deterministic $A_\mu$ was the one method
that preferred `both_wrong` on all four blocks.** If the paper's headline is the judge loop
driven by $A_\mu$, that global tally is the wrong statistic to decide on, so this is a
dedicated measurement of exactly the configuration in question.

`experiments/run_gamma_mode_headline.py` · results in `results/gamma_mode_headline.json`
(part A) and `results/gamma_mode_headline_b.json` (part B).

## Design

**15 cases, covering every case available** --- no sampling:

| task | part A | part B |
|---|---|---|
| Text2SQL | spider, bird (protocol A, real labeled source split) | spider, bird, sqlflow, bird_minidev, spider2local (protocol B, target holdout) |
| image | mnist→usps, mnist→svhn | — |
| node | AC, CD, DA | AD, CA, DC |

× 2 gamma modes × 2 $A_\mu$ variants × budgets {10, 20, 40} × 5 seeds.

### The prediction, registered before the run

$A_\mu$ scores an item by how far the truth would **move** the model. A model carrying a
biased $\gamma$ has more to be moved by, so $A_\mu$ has more leverage to find. If that is
what is happening, `both_wrong` looks better under $A_\mu$ for a reason about **$A_\mu$**,
not about $\gamma$ being correct. Two falsifiable consequences:

1. the effect holds for the deterministic argmax but **not** the sampled variant;
2. the gap **shrinks as budget grows**, since judge labels crowd out model beliefs.

## Headline result

MAE difference, `model_wrong` minus `both_wrong`. **Negative = `model_wrong` better.**

| selector | b=10 | b=20 | b=40 | **all** | model_wrong better in |
|---|---|---|---|---|---|
| $A_\mu$ (deterministic) | +0.83 | +1.29 | +0.68 | **+0.93** | 20/45 cells |
| $A_\mu$ sampled | +0.07 | -0.01 | -0.14 | **-0.03** | 27/45 cells |

## Per-case differences

Cells with $|\Delta|\ge2$ in bold.

| case | det b=10 | det b=20 | det b=40 | samp b=10 | samp b=20 | samp b=40 |
|---|---|---|---|---|---|---|
| graph/AC | -0.12 | **+5.23** | -0.62 | **+5.62** | **+2.26** | +1.00 |
| graph/AD | -1.34 | **-3.93** | -0.85 | **-2.69** | -1.65 | -1.15 |
| graph/CA | +1.82 | -1.85 | **-4.29** | -0.07 | -0.81 | -1.77 |
| graph/CD | **+2.76** | **-4.85** | +0.35 | +1.43 | +0.68 | -0.21 |
| graph/DA | **-2.33** | -1.00 | **+4.82** | +0.76 | -0.01 | -0.76 |
| graph/DC | **+3.51** | +1.78 | +0.70 | +0.24 | +1.17 | -0.10 |
| text2sql/bird | **-3.39** | **-2.10** | -0.00 | -0.31 | -0.43 | -0.53 |
| text2sql/bird/B | -1.53 | -0.01 | +0.06 | -1.46 | -1.61 | -0.69 |
| text2sql/bird_minidev/B | +0.64 | +0.60 | +0.86 | +0.13 | +0.23 | +1.30 |
| text2sql/spider | +1.07 | +0.05 | +0.37 | -0.65 | -1.91 | -0.51 |
| text2sql/spider/B | -0.63 | -1.76 | +1.63 | **-2.28** | **-2.36** | +0.19 |
| text2sql/spider2local/B | -0.05 | -0.01 | -0.01 | -0.63 | -0.47 | -0.47 |
| text2sql/sqlflow/B | +0.30 | +1.66 | **+2.63** | +0.62 | +0.96 | +1.66 |
| vision/mnist_svhn | **+2.76** | **+20.32** | +0.05 | **-4.98** | -0.25 | **-2.86** |
| vision/mnist_usps | **+9.05** | **+5.20** | **+4.47** | **+5.30** | **+4.12** | **+2.77** |

## Absolute MAE, averaged over the three budgets

| case | det: both_wrong | det: model_wrong | samp: both_wrong | samp: model_wrong |
|---|---|---|---|---|
| graph/AC | 9.90 | 11.39 | 6.80 | 9.76 |
| graph/AD | 13.83 | 11.79 | 13.60 | 11.77 |
| graph/CA | 11.81 | 10.37 | 10.70 | 9.82 |
| graph/CD | 12.16 | 11.58 | 9.42 | 10.06 |
| graph/DA | 12.63 | 13.13 | 13.10 | 13.10 |
| graph/DC | 9.44 | 11.44 | 7.79 | 8.23 |
| text2sql/bird | 7.00 | 5.17 | 6.62 | 6.19 |
| text2sql/bird/B | 4.65 | 4.16 | 6.30 | 5.04 |
| text2sql/bird_minidev/B | 4.17 | 4.87 | 3.93 | 4.49 |
| text2sql/spider | 3.69 | 4.19 | 5.08 | 4.05 |
| text2sql/spider/B | 5.87 | 5.62 | 6.17 | 4.68 |
| text2sql/spider2local/B | 2.19 | 2.17 | 2.96 | 2.44 |
| text2sql/sqlflow/B | 5.70 | 7.23 | 4.45 | 5.54 |
| vision/mnist_svhn | 23.41 | 31.12 | 44.48 | 41.78 |
| vision/mnist_usps | 2.93 | 9.17 | 3.76 | 7.83 |
| **mean** | **8.63** | **9.56** | **9.68** | **9.65** |

## What the three questions resolved to

**1. Does `both_wrong` genuinely beat `model_wrong` for deterministic $A_\mu$?
Yes, but weakly.** +0.93 mean, winning 25 of 45 cells --- barely better than a coin flip on
cell count. And the margin is **driven almost entirely by the vision block**:
`vision/mnist_usps` is consistently large (+9.05, +5.20, +4.47) and `vision/mnist_svhn` at
b=20 contributes a single +20.32 outlier. The graph and Text2SQL cases scatter in both
directions with no visible pattern.

**2. Does the sampled variant go the other way? Essentially yes --- it is indifferent.**
−0.03 mean, 27/45 cells, against +0.93 for the argmax. The same score, computed from the
same information, yields opposite conclusions depending only on whether it is maximised or
sampled from. That is consistent with the prediction: the effect is about the argmax
exploiting a miscalibrated model, not about $\gamma$ being more correct.

**3. Does the gap shrink with budget? No --- and this part of the prediction FAILED.**
Observed +0.83 → +1.29 → +0.68: non-monotone and essentially flat. More judge labels do not
erode the effect, which is evidence *against* the mechanism as stated. Consequence (1) held
and consequence (2) did not, so the explanation is at best incomplete and should not be
repeated as though it were established.

## Recommendation, and its status

**Keep `model_wrong` --- as a judgement call, not a measurement.** It *loses* the headline
comparison by 0.93.

Reasons to keep it anyway:

* the edge is small, non-monotone in budget, and concentrated in the vision block that also
  produced the unexplained anomaly in `docs/three_betas.md`; it is not a stable effect for a
  default to rest on;
* it vanishes under $A_\mu$ **sampled** (−0.03), which was best-in-block on Text2SQL/A and
  second on node under the current default;
* `both_wrong` requires the conversion measured to be biased by **0.28--0.47 on every one of
  ten labeled splits** (`experiments/run_gamma_mode_diagnostic.py`). Shipping a known-biased
  conversion to gain 0.93 MAE on one selector variant is a poor trade.

**Stated plainly:** for the headline as specified --- judge loop with *deterministic*
$A_\mu$ --- `both_wrong` is 0.93 better, and my explanation for why has failed its own
falsification test. Someone optimising the headline number rather than defensibility should
choose `both_wrong`, and vision is where it pays.

**The cleanest resolution is to make $A_\mu$ sampled the headline selector.** It is
indifferent to the mode (−0.03), so the acquisition choice and the `gamma_mode` choice
decouple entirely and neither has to be justified by the other.
