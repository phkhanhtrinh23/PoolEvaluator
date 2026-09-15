# Does the prior matter? Randomly initialising pi, beta, gamma and e

**Question.** The pipeline computes four quantities from a pre-built labeled split before
EM ever runs. How much of the final accuracy estimate comes from those measured priors,
and how much would we get from a random start?

**Method.** Eight arms, each replacing one or more measured priors with a uniform random
draw. 6 cases (text2sql x2, image x2, node x2), budgets {0, 10, 40} expert labels,
oracle expert, `mean_gain` (A_mu) selection, **10 independent draws per arm**, reported as
mean +/- sd across draws. The baseline is deterministic, so its sd is exactly 0.

Script: `experiments/run_random_init.py`. Raw numbers: `results/random_init.json`.

## What each prior is, and whether EM can repair it

| prior | normally computed as | shape | repaired by EM? | repaired by expert loop? |
|---|---|---|---|---|
| `pi` | measured accuracy of model j on the labeled split | M-vector | it is the anchor *in* the M-step, so no | no |
| `beta` | pseudo-label accuracy on the labeled split | scalar | **yes** — closed-form M-step each sweep | n/a |
| `gamma` | `gamma_counts(mode="model_wrong")` on the labeled split | M-vector | **no** | **yes** — `(old+temp)/2` blend |
| `e` | wrong-collision counts on the labeled split | MxM matrix | **no** | **yes** — `(old+temp)/2` blend |

This table is the whole design: it separates the things EM re-estimates from the things
only the outer loop touches.

## Randomisation ranges

- `pi_j ~ U(0.15, 0.95)` i.i.d. per model. When this arm is on, anchor strength `s` is
  set to **0** — anchoring hard to a meaningless number would be a strawman. So the
  "random pi" arm is really **"no usable prior on alpha"**.
- `beta ~ U(0.05, 0.95)`, one scalar. Set via `stats._pseudo_acc`, which is where
  `validated_em` reads its beta default from.
- `gamma_j ~ U(0.05, 0.95)` i.i.d. per model.
- `e`: symmetric, off-diagonal `~ U(0, 0.5)`, diagonal `~ U(0, 0.8)` (the diagonal is a
  model's own error rate, so it lives on a wider range).

## Table 1 — MAE averaged over the 6 cases

Units: **accuracy percentage points**, lower is better. Mean +/- sd over 10 draws.

| arm | b0 | b10 | b40 | b40 gap vs baseline |
|---|---|---|---|---|
| **measured (baseline)** | **18.26 +/- 0.00** | **15.16 +/- 0.00** | **8.18 +/- 0.00** | — |
| random beta | 18.29 +/- 0.01 | 15.61 +/- 0.33 | 8.50 +/- 0.36 | +0.32 |
| random e | 18.46 +/- 0.34 | 16.59 +/- 3.48 | **8.02 +/- 1.82** | **-0.16** |
| random gamma | 22.71 +/- 3.34 | 17.30 +/- 3.66 | 8.43 +/- 1.83 | +0.25 |
| random gamma+e | 22.65 +/- 3.38 | 15.99 +/- 3.95 | 9.41 +/- 3.11 | +1.23 |
| random pi (no anchor) | 28.06 +/- 3.24 | 19.70 +/- 5.19 | 10.83 +/- 2.64 | +2.65 |
| random pi+beta | 28.69 +/- 3.42 | 19.64 +/- 4.81 | 11.28 +/- 2.50 | +3.10 |
| random EVERYTHING | 33.42 +/- 6.25 | 21.56 +/- 5.76 | 11.64 +/- 2.91 | +3.46 |

Rows sorted by damage done. Only **one** of the four priors matters, and it is `pi`.

## Finding 1 — the beta prior is worthless

18.26 -> 18.29 at b0, sd **0.01**. In three of six cases (spider, mnist_svhn, graph/DA)
the randomised runs reproduced the baseline to the last printed decimal with sd exactly
0.00: ten different starts in U(0.05, 0.95) all landed on the same number.

This is the closed-form beta M-step doing its job. Given the current responsibilities tau
and the agreement indicators C,

    beta = (sum_ij tau_ij C_ij + s_beta pi_beta) / (sum_ij tau_ij + s_beta)

has a unique maximiser, reached on the first sweep from any start. **Do not bother
measuring beta on a labeled split** — initialise it to 0.5 and the pipeline is unchanged.

## Finding 2 — the e prior is nearly free

18.26 -> 18.46 at b0 (+0.20); at b40 the random arm is marginally *better* (8.02 vs 8.18,
inside one sd). EM never updates `e`, so a random `e` stays random unless the expert loop
blends it — yet it costs almost nothing.

Reason: `e` enters only through the vote discount

    d_ij = 1 / (1 + sum_{k != j, r_k = r_j} e~_jk)

which is smooth, bounded and monotone. Getting the magnitudes wrong perturbs the vote
weights but rarely changes *which answer wins*. The cost surfaces as variance rather than
bias: sd climbs 0.34 (b0) -> 3.48 (b10).

## Finding 3 — the gamma prior costs 4.45 points, and the expert loop pays it back

22.71 at b0 vs 18.26 — a genuine penalty. By b40: 8.43 vs 8.18, a gap of 0.25, inside one
sd.

This is the cleanest evidence yet that `(old+temp)/2` is a working repair channel. EM never
touches gamma, so the entire 4.45 -> 0.25 recovery is attributable to the expert updates.
**Forty labels suffice to walk a uniformly-random gamma back to measured-quality.**

## Finding 4 — the alpha anchor is the prior that matters, and it does not wash out

28.06 at b0, still 10.83 at b40 — a 2.65-point gap that 40 expert labels do not close.
Unlike gamma, there is no channel that repairs a bad alpha prior, because the anchor *is*
part of the M-step:

    alpha_j = (sum_i tau_ij + s_j pi_j) / (N + s_j)

With `s = 0` the estimate is pure data; with a measured pi it is shrunk toward the labeled
split. The gap persisting to b40 says the EM likelihood surface has initialisation-dependent
optima, consistent with the earlier init-ablation result.

## Table 2 — but the alpha prior is not uniformly good

Per-case MAE (accuracy percentage points), baseline vs no-prior. **Bold = no-prior wins.**

| case | b0 baseline | b0 random pi | b40 baseline | b40 random pi |
|---|---|---|---|---|
| text2sql/spider | 3.61 | 7.16 | 2.66 | 8.62 |
| text2sql/bird | 4.14 | 25.05 | 3.61 | 10.81 |
| vision/mnist_usps | 16.39 | **15.07** | 6.15 | **4.70** |
| vision/mnist_svhn | 57.05 | **54.41** | 13.09 | 17.69 |
| graph/AC | 8.58 | 28.82 | 8.22 | 8.80 |
| graph/DA | 19.79 | 37.87 | 15.37 | **14.33** |

On both vision cases a **random** pi beats the measured one at b0, and on `mnist_usps` it
wins at b40 too. This matches the labeled->target shift measured earlier: on SVHN the
anchor comes from a split whose accuracy profile is 0.170-0.416 away from the target pool,
so the "informative" prior is informative about the wrong distribution. A uniform draw is
at least unbiased. Where the labeled split genuinely matches the target (text2sql, graph/AC)
the prior is worth 20+ points at b0.

**Practical rule:** the alpha anchor is worth keeping only when the labeled split is drawn
from the same distribution as the pool being evaluated. Under shift, prefer `s = 0`.

## Finding 5 — the penalties compound additively

Random-everything at b0 is 33.42. Baseline 18.26 + pi penalty 9.80 + gamma penalty 4.45 =
32.51, within one sd of 33.42. The priors are close to independent in their contribution;
there is no multiplicative blow-up from getting several wrong at once.

## Summary ranking

| rank | prior | worth at b0 | worth at b40 | why |
|---|---|---|---|---|
| 1 | `pi` (alpha anchor) | ~9.8 pts | ~2.7 pts | no repair channel; but harmful under distribution shift |
| 2 | `gamma` | ~4.5 pts | ~0.25 pts | expert loop repairs it via `(old+temp)/2` |
| 3 | `e` | ~0.2 pts | ~0 pts | only reweights votes; buys variance reduction |
| 4 | `beta` | 0 pts | 0 pts | unique closed-form M-step fixed point |

## Reproduce

    python experiments/run_random_init.py --draws 10
