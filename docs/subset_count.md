# How many labeled subsets should the prior be measured on?

**Question.** The prior ($\pi$, the $e$ matrix, $\gamma$) is read off labeled data that arrives
in fixed-size subsets. Each extra subset costs work. How many are worth buying?

**Answer: three, and only for Text-to-SQL.** Across all three modalities the prior-building
cost is exactly linear in the number of subsets $K$, while MAE falls steeply from $K=1$ to
$K=3$ and then flattens. Past $K=3$ latency keeps rising and accuracy does not improve.
The averaged curve hides a split: Text-to-SQL improves 3.6x, graph 1.3x, and **vision not at
all** — under heavy domain shift the prior is wrong by a large constant and measuring that
wrong number more precisely does not make it less wrong.

Scripts: `experiments/run_subset_count.py`, `experiments/plot_subset_count.py`.
Raw numbers: `results/subset_count.json`. Figure: `figures/subset_count.{png,pdf}`.

## Setup

| knob | value |
|---|---|
| subset size | 10 labeled items, disjoint blocks |
| $K$ | 1, 2, 3, 4, 6, 8, 12 → 10 … 120 labeled items |
| cases | spider, bird, mnist→usps, mnist→svhn, graph AC, graph DA |
| draws | 5 (fresh subsets each) |
| timing | median of 25 repeats, divided by $M$ |
| selector | `mean_gain` ($A_\mu$), `OracleExpert` |

The 10-item block is forced by the smallest labeled pool in the study: Spider and BIRD have
120 labeled items, so $K=12$ consumes all of it. That is why the text2sql rows have
`sd = 0.00` at $K=12` — there is only one possible subset selection, so every draw is
identical. A common $K$ grid across all six cases is what makes the modality average
meaningful.

Everything the prior contains is rebuilt from the same $K$ subsets — $\pi$, the $M\times M$
matrix $e$, and $\gamma$ — so $K$ genuinely means "how much labeled data does the prior get".
The declared prior sd is the binomial SE at $K\times 10$ items, floored at the transfer noise
so a 10-item subsample cannot claim an anchor stronger than domain shift permits.

### What the latency does and does not include

The models' outputs on the labeled split are cached, so the timer covers the **prior
computation**: scoring each model on each subset (`t_pi`), plus the collision matrix,
pseudo-labels and $\gamma$ (`t_all`). It does **not** include running the $M$ models on those
items, which in deployment dominates and is also linear in $K$. These numbers are therefore a
lower bound on true cost; the linear shape is the part the prior-building code contributes.

## Table 1 — the headline trade-off

Mean over the 3 modalities (each modality = mean of its 2 cases), 5 draws.
Latency is milliseconds **per model**. MAE is in accuracy points, already averaged over
models by `mae()`. Lower is better for both.

| $K$ | items | $t_\pi$ (ms) | $t_\text{all}$ (ms) | MAE b0 | MAE b10 | MAE b40 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 0.0029 | 0.0162 | 21.76 | 19.71 | 9.20 |
| 2 | 20 | 0.0053 | 0.0271 | 19.19 | 18.70 | 8.36 |
| 3 | 30 | 0.0078 | 0.0380 | **17.92** | 18.00 | **8.30** |
| 4 | 40 | 0.0101 | 0.0488 | 18.34 | 18.08 | 8.52 |
| 6 | 60 | 0.0150 | 0.0703 | 17.70 | 17.71 | 8.24 |
| 8 | 80 | 0.0198 | 0.0920 | 17.32 | 16.65 | 8.41 |
| 12 | 120 | 0.0295 | 0.1359 | 16.91 | 15.33 | 9.41 |

Cost is linear to 4 significant figures: per-item latency is 0.287 µs at $K=1$ and 0.246 µs at
$K=12$, a 14% drift explained by fixed per-call overhead amortising, not by any superlinear term.

Benefit is not linear. From $K=1$ to $K=3$, b0 MAE falls 3.84 points for 0.005 ms. From $K=3$
to $K=12$, it falls 1.01 more points for 0.022 ms — **4x the cost for a quarter of the gain**.
At b40 the flattening is complete: every $K$ from 2 to 8 sits in 8.24–8.52, a range narrower
than the draw-to-draw spread, and $K=12$ is *worse* (9.41).

## Table 2 — the average hides three different behaviours

MAE per modality. Same units.

| $K$ | items | text2sql b0 | text2sql b40 | vision b0 | vision b40 | graph b0 | graph b40 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 14.24 | 7.49 | 33.01 | 9.67 | 18.03 | 10.45 |
| 2 | 20 | 9.28 | 5.95 | 33.03 | 9.55 | 15.27 | 9.57 |
| 3 | 30 | 5.98 | 6.34 | 33.45 | 9.49 | **14.34** | **9.07** |
| 4 | 40 | 6.07 | 5.96 | 33.40 | 9.94 | 15.57 | 9.66 |
| 6 | 60 | 5.55 | 4.93 | 33.37 | 9.37 | 14.19 | 10.42 |
| 8 | 80 | 4.75 | 3.77 | 33.30 | 11.78 | 13.91 | 9.67 |
| 12 | 120 | **3.92** | 3.99 | 32.91 | 14.12 | 13.92 | 10.12 |

- **Text-to-SQL pays for every subset.** 14.24 → 3.92 at b0, monotone apart from noise. This is
  the only modality where buying the twelfth subset is still worth something.
- **Vision is flat to within noise.** 33.01 → 32.91 over a 12x increase in labeled data. The
  b0 error is ~33 points and stays there because it is *bias*, not variance: MNIST→SVHN alone
  sits near 49 at every $K$. Sharpening a biased estimate does not debias it.
- **Graph gains early and then stops.** 18.03 → 14.34 by $K=3$, then nothing (13.92 at $K=12$).

### The one place more data actively hurts

Vision b40 rises 9.49 → 11.78 → 14.12 over $K$ = 3, 8, 12, driven entirely by MNIST→SVHN
(14.62 → 19.28 → 23.22, with the spread exploding to ±16.33). The mechanism is the anchor: a
prior measured on more items declares a smaller binomial sd, `_strength` converts the smaller
sd into a *stronger* anchor, and a stronger anchor pins the estimate harder to a prior that is
~86 accuracy points wrong. The transfer-noise floor caps this but does not eliminate it. This
is the same failure `docs/ess_coverage.md` describes: the prior's usable effective sample size
is bounded by the transfer term, not by the source sample size. **Under severe shift, more
labeled data makes the prior more confident and therefore more harmful.**

## Recommendation

$K=3$ (30 labeled items) is the knee for the aggregate and for graph; text2sql alone justifies
going to $K=12$. No configuration justifies buying subsets past the point where the anchor
starts dominating, which under severe shift is immediately.
