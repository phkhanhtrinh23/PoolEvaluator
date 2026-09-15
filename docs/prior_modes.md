# Deployment modes: how cheap a prior can we get away with?

**The cost being avoided.** All four starting values -- `pi` (the alpha anchor), `beta`,
`gamma` and `e` -- are read off ONE artifact: the M models run over a labeled subset. That
run is what costs money and latency at deployment. So the deployment question is not which
prior to compute but **how many labeled items to run the models on**.

`docs/random_init.md` showed that only `pi` carries real information, and `pi` is a
per-model mean -- it needs far fewer items than the M x M matrix `e`. That suggests a cheap
middle mode: spend a small labeled subset on `pi` alone and randomise the rest.

| mode | what it measures | labeled items needed |
|---|---|---|
| `measured` | every prior, from the full labeled split | 120 / 1097-1872 / 3000-10000 |
| `pi@n` | `pi` from an n-item subsample; `beta`, `gamma`, `e` random | n |
| `random` | nothing; no anchor (`s = 0`) | 0 |

Script: `experiments/run_prior_modes.py`. Raw numbers: `results/prior_modes.json`.
Library support: `LabeledStatistics.random(M, ...)` in `pooleval/validated_em.py`.

## Table 1 -- MAE averaged over the 6 cases

Units: **accuracy percentage points**, lower is better. `b0` = no expert labels, `b10`/`b40`
= 10/40 expert-validated items. Mean over 10 draws per case, then over 6 cases.

| mode | b0 | b10 | b40 |
|---|---|---|---|
| `measured` | **18.26** | **15.16** | **8.18** |
| `pi@500` | 22.93 | 17.24 | 9.83 |
| `pi@100` | 22.06 | 17.13 | 9.61 |
| `pi@25` | 25.24 | 18.69 | 9.52 |
| `random` | 32.41 | 20.35 | 11.19 |

**Caution on spread.** The per-case tables printed by the script carry `+/- sd across the
10 draws`; the aggregate row's spread is `sd across the 6 cases`, which is dominated by how
different SVHN is from Spider and says nothing about run-to-run noise. Only the per-case
sds are draw noise, and they are large for the cheap modes (up to `+/-12.59` on SVHN).

## Table 2 -- fraction of the prior's value recovered

Scaled so `random` = 0% and `measured` = 100%.

| mode | b0 | b10 | b40 |
|---|---|---|---|
| `pi@500` | 67% | 60% | 45% |
| `pi@100` | **73%** | **62%** | 53% |
| `pi@25` | 51% | 32% | **55%** |

**100 labeled items recover ~73% of what the full split is worth at b0**, against splits of
1872-10000 items on vision and graph. That is the headline: most of the prior's value is
purchasable with two orders of magnitude less labeled data, because the only prior that
matters is a per-model mean and a per-model mean converges fast.

`pi@500` being *worse* than `pi@100` at every budget is not a real ordering -- the two are
within the draw noise of each other (`pi@500` 22.93, `pi@100` 22.06, with per-case sds of
2.5-8.8). Read them as tied. What is real is that both clearly beat `pi@25` at b0 and both
clearly beat `random` everywhere.

## Table 3 -- per-case at b40, where the averages hide the story

Units: accuracy percentage points. **Bold = matches or beats `measured`.**

| case | `measured` | `pi@100` | `pi@25` | `random` |
|---|---|---|---|---|
| text2sql/spider | 2.66 | 3.16 | 5.62 | 9.01 |
| text2sql/bird | 3.61 | 4.79 | 7.98 | 12.04 |
| vision/mnist_usps | 6.15 | **4.61** | **4.93** | **5.37** |
| vision/mnist_svhn | **13.09** | 23.57 | 20.21 | 19.10 |
| graph/AC | 8.22 | 10.15 | **8.21** | 10.91 |
| graph/DA | 15.37 | **11.40** | **10.20** | **10.73** |

The 6-case average says `measured` wins. The per-case table says it wins on **three of six**.

- **Text-to-SQL: measured wins, but there is nothing to save.** The labeled split is only
  120 items, so `pi@100` is 83% of it. The expensive-prior problem is a vision/graph
  problem, not a Text-to-SQL one.
- **vision/mnist_usps and graph/DA: the cheap modes WIN.** On DA, `pi@25` scores 10.20
  against measured's 15.37 -- a 5-point improvement from using 25 labeled items instead of
  1097. This is the distribution-shift effect from `docs/prior_gap_tables.md`: the measured
  anchor is confidently wrong (bias +18.73 on DA), and a weak or random anchor lets the
  target data speak.
- **vision/mnist_svhn: measured wins decisively** (13.09 vs 19-27) and the cheap modes are
  wildly unstable (`pi@500` is 27.15 +/- 12.59). SVHN has the largest prior error in the
  whole study (bias +85.51) yet the measured anchor still helps, because `cMAE` there is
  only 8.25 -- the ranking is right even though the level is not, and the anchor transfers
  that ranking.

## What to deploy

1. **Use `pi@100`** as the default cheap mode: ~73% of the prior's value at b0, ~53% at
   b40, from 100 labeled items instead of thousands.
2. **Randomise `beta`, `gamma` and `e` unconditionally.** `docs/random_init.md` shows beta
   costs nothing (unique closed-form M-step fixed point) and `e` ~0.2 points; `gamma` costs
   4.45 at b0 but the expert loop's `(old+temp)/2` repairs it by b40.
3. **Check for distribution shift before paying for the full split.** Where the labeled
   split does not resemble the target, a measured anchor is not merely a waste of money, it
   is actively harmful -- graph/DA and mnist_usps both do better with less.
4. **Budget for variance, not just bias.** The cheap modes are stochastic: per-case sds run
   to +/-12.59. If a single run must be trustworthy rather than an average over many, that
   spread matters more than the 1.4-point mean gap.

## Reproduce

    python experiments/run_prior_modes.py --draws 10 --modes measured pi@500 pi@100 pi@25 random
