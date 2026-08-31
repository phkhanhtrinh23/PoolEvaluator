# Porting PoolEval beyond Text2SQL: image and node classification

Two real, non-simulated ports, with the model pools taken from the papers that
define each task's label-free evaluation setting:

| domain | setup | source | models |
|---|---|---|---|
| image classification | MNIST → USPS, MNIST → SVHN | MetaEvaluator, Pham et al., KDD 2026 ([arXiv 2605.23595](https://arxiv.org/abs/2605.23595)) | ResNeXt-50-32x4d, RegNetY-8GF, ConvNeXt-Tiny, ViT-Tiny, DeiT-Small |
| node classification | ACMv9 / Citationv1 / DBLPv7, six cross-graph transfers | GNNEvaluator, Zheng et al., NeurIPS 2023 ([arXiv 2310.14586](https://arxiv.org/abs/2310.14586)) | GCN, GraphSAGE, GAT, GIN, MLP |

Each architecture family is trained with three seeds, so **a family is a
provenance group**: same inductive bias, correlated errors — the structure
`n_groups` models in the Text2SQL pool.

```bash
python experiments/run_domain_graph.py          # 6 transfers, ~15 min on one GPU
python experiments/run_domain_vision.py         # 2 shifts,  ~90 min on one GPU
python experiments/run_domain_diagnostics.py --kind graph|vision
```

## What ports unchanged, and what does not

`run_em` only ever tests observations for **equality**, so it is task-agnostic.
Only three things are domain-specific:

| component | Text2SQL | vision / graph |
|---|---|---|
| observation kernel | result-set equivalence, graded LA0/LA1/LA2 | predicted label, **exact match — the kernel is the identity** |
| prior | accuracy on a seen benchmark | held-out **source** accuracy |
| verifier | execution on the live DB | held-out model + TTA (vision); feature-only LR smoothed over the target graph (graph) |

`pooleval/domains/adapter.py` maps predictions into the repo's class convention
(`0` = correct) by `pred == gold ? 0 : 1 + pred`. This is an exact isomorphism of
the per-item agreement partition, so the estimator sees precisely the agreement
structure of the raw predictions.

**One failure mode disappears.** In Text2SQL the candidate set is whatever tables
the pool produced, so the truth can be absent from it entirely. With a closed
K-class label space the truth is always one of K candidates, and an external
channel can always name it. **The gauge trap does not disappear** — that is what
the numbers below are about.

## Results

Node classification, mean over six transfers (`results/domain_graph.json`):

```
method                       MAE     bias     rho     tau
DoC                       0.0546  +0.0381  +0.543  +0.424
B1 Independent            0.1309  +0.1269  +0.850  +0.718
B4 Agreement-on-line      0.1336  +0.1269  +0.846  +0.707
PoolEval                  0.1531  +0.1531  +0.782  +0.621
PoolEval (no anchors)     0.1641  +0.1641  +0.890  +0.768
PoolEval (learned verif.) 0.1660  +0.1660  +0.879  +0.752
B3 Dawid--Skene           0.1763  +0.1763  +0.883  +0.749
B2 Majority/self-cons.    0.1771  +0.1771  +0.862  +0.735
ATC-MC                    0.2049  -0.0272  -0.304  -0.210
ATC-NE                    0.2164  -0.0344  -0.370  -0.257
```

Image classification, MNIST → USPS (`results/domain_vision.json`):

```
B3 Dawid--Skene           0.0370  +0.0370  +0.989
PoolEval (no anchors)     0.0474  +0.0288  +0.836
B2 Majority/self-cons.    0.0664  +0.0422  +0.943
PoolEval (learned verif.) 0.0686  +0.0362  +0.789
ATC-NE                    0.1853  +0.1742  -0.186
DoC                       0.2167  +0.2167  -0.007
PoolEval                  0.2420  +0.0002  -0.746
B1 Independent            0.2918  +0.2918  -0.403
```

MNIST → SVHN is a **degenerate regime**: true target accuracy is 0.062–0.355
with K = 10, so most of the pool is *below chance*. Every method listed above
fails there, with bias between +0.46 and +0.85 and negative rank correlation.

> **Correction (later work).** An earlier version of this document attributed
> that failure to the Dawid–Skene identifiability theorem and called it
> unsalvageable. That was too strong. It is a *uniform-error* failure, not an
> identifiability failure: under this shift six models collapse to emitting one
> class for 96–98% of items, and one-coin DS reads their mutual agreement as
> accuracy (scoring them 0.96–0.98 against a true 0.06). Giving each model a
> confusion matrix instead makes a collapsed model visibly uninformative — its
> confusion rows become identical — and recovers the case: MAE 0.5598 → 0.1528
> and rho −0.886 → **+0.829**. See [`multiclass_ds.md`](multiclass_ds.md) §5.
> The identifiability symmetry is still real; it just was not what was binding
> here.

## Three defects the ports exposed

### 1. The prior is switched off by large N

Precision fusion weights the prior by `1/prior_sigma^2` against `1/a_sigma^2`
with `a_sigma = sqrt(a(1-a)/N)`. Vision and graph targets have N in the thousands:

```
pool               N   a_sigma   w_prior
mnist_usps      2007    0.0112     2.48%
mnist_svhn      5000    0.0071     1.01%
graph AD        5484    0.0068     0.92%
graph CA        9360    0.0052     0.54%
```

The prior receives **under 1%** of the weight on the graph transfers, so removing
it entirely changes MAE by 0.0013 (0.1531 → 0.1544) — while `fusion=prior_only`
scores 0.1309, the best of every PoolEval variant. The one anchor that could
correct a correlated pool is silenced exactly where the pool is most correlated.

The root cause is that `a_sigma` is the **binomial standard error of the agreement
rate** — pure sampling noise. It is not the error of `a_agree` as an estimate of
true accuracy, which is dominated by *systematic* consensus error and does not
shrink with N at all. Fix: calibrate the consensus variance against something
other than 1/N, or floor the prior's weight.

### 2. `verifier_strength` is an undeclared MAE-vs-ranking knob

The bonus is added on the **accuracy scale** while every other term is a
reliability bounded in (0, 1), so at the default 2.0 it outvotes several whole
models. Raising it shrinks every model toward the verifier: common-mode bias
falls, between-model spread collapses.

```
graph                          vision
strength   MAE      rho        strength   MAE      rho
    0.00  0.1639  +0.890           0.00  0.3035  -0.018
    0.25  0.1657  +0.906           0.25  0.3080  -0.013
    2.00  0.1531  +0.782           2.00  0.3988  -0.815   <- default
    4.00  0.1231  +0.031           4.00  0.4702  -0.918
```

On MNIST → USPS the default drives PoolEval to MAE 0.2420 with rho −0.746 — a
*reversed* ranking — while plain Dawid–Skene on the same predictions gets 0.0370
and rho +0.989. The verifier there (a held-out CNN + TTA) is 8% accurate, i.e.
below chance, and the fixed bonus lets it dominate anyway.

**Mitigation implemented:** `cfg.verifier_mode = "learned"` appends the verifier
as one extra voter in its own provenance group and lets EM estimate its
reliability like any pool member, weighting it by `w = a_verifier` on the same
scale as everyone else. This removes the hyper-parameter and recovers most of the
loss (vision MAE 0.3988 → 0.3134, rho −0.815 → −0.048; graph rho +0.782 →
+0.879). The default stays `"fixed"` so published Text2SQL numbers reproduce; on
the Text2SQL simulator, where the verifier is genuinely reliable and the constant
was tuned for it, `"learned"` costs 0.0320 → 0.0411 MAE.

It inherits the fundamental limit: on MNIST → SVHN the estimated verifier
reliability is **0.98 against a true 0.06**, because EM measures agreement with
the pseudo-groundtruth and the pseudo-groundtruth is itself wrong.

### 3. The posterior cannot self-diagnose

```
pool          latent_hat acc  mean max-post   true bias
graph DA              0.5747         0.9035     +0.2029
graph CD              0.6973         0.9244     +0.1452
mnist_usps            0.6114         0.7785     +0.0002
mnist_svhn            0.0614         0.9407     +0.4589
```

On MNIST → SVHN the hard latent answer is correct on **6%** of items while the
posterior's mean confidence is **94%**. Across the graph transfers,
`corr(confidence, |bias|) = +0.045` — a working self-diagnostic would be strongly
negative. Confidence carries no information about the estimator's own error, and
on the worst pool it is *positively* correlated with it.

**Consequence for Active PoolEval:** entropy-based acquisition cannot find these
items. This independently confirms, on real predictions in two new domains, the
design choice at `active.py:141` that gives entropy only a 0.05 coefficient and
puts the weight on `corr_risk`.

## Reading of the results

- **Consensus beats confidence on ranking; confidence beats consensus on level.**
  On graph, DoC has the best MAE (0.0546) and the worst usable ranking
  (rho +0.543, top-1 never correct); the consensus family clusters at rho
  +0.78–0.89. They fail in orthogonal directions, which is an argument for
  combining them rather than choosing.
- **Every consensus method carries a large positive bias** (+0.15 graph, +0.25
  vision). Models under shift fail *together*, consensus reads that as
  correctness, and the estimate inflates. This is the gauge trap measured on real
  predictions, not simulated.
- **PoolEval's anchors are tuned for the Text2SQL regime** (N ≈ 400, a reliable
  execution verifier, an informative seen prior). All three assumptions break in
  these domains, and with the anchors disabled PoolEval matches or beats plain
  Dawid–Skene while keeping the group-discount gain (graph: 0.1641 / rho +0.890
  versus DS 0.1763 / +0.883).
- **Vision breaks the prior in a way Text2SQL does not.** Every model reaches
  ≈0.99 on MNIST, so the seen prior is a constant: B1 scores MAE 0.2918 and B4's
  regression is degenerate (rho NaN). Source accuracy carries *zero* ranking
  information, so consensus is the only available signal — the strongest case for
  a pool, and the exact opposite of the graph transfers where the prior is the
  single best-ranking method.

## The collision-aware formulation on the new domains

`Trinh_proof.tex` ("Collisions: Which Closed Forms Survive") drops the hidden
assumption `P(C=1 | Z=0) = 1-β`, which says a wrong model and a wrong
pseudo-label *always* produce the same wrong answer. That is forced for K=2 and
false everywhere else. Replacing it with `P(C=1 | Z=0) = (1-β)γ_g` keeps the
E-step and the α M-step in closed form and costs only the β M-step, which becomes
a concave one-dimensional maximization (`pooleval/new_formulation.py:200-210`).

The closed-label-space ports are the natural place to test it, because there `γ`
has an exact analytic null: under independent errors over `K` classes a wrong
model lands on the pseudo-label's wrong class with probability `1/(K-1)`.

`pooleval/domains/collision.py` estimates `γ_g` on the **source validation
split** — never the target — by running the identical pipeline (PoolEval →
argmax pseudo-label → count wrong-model / wrong-pseudo-label matches per
provenance group), the domain analogue of the leave-one-dataset-out estimate in
`zoo/collision_formulation_real.py`.

### Collisions are real, large, and not close to either extreme

| domain | K | null `1/(K-1)` | γ̂ source | γ̂ target (oracle) | ratio to null |
|---|---|---|---|---|---|
| node (6 transfers) | 5 | 0.250 | **0.863** | 0.803 | 3.4× |
| image (2 shifts) | 10 | 0.111 | **0.924** | 0.605 | 8.3× |

Both extremes in the literature are wrong. `γ=1` (the pre-collision binary
model) overstates collision; `γ=1/(K-1)` (independent errors, what a naive
multi-class Dawid–Skene assumes) understates it by 3–8×. The pool's wrong
answers really do collide — which is exactly the shared-error signal
MetaEvaluator and GNNEvaluator go after with learned shift descriptors, here read
straight off the pool with no meta-training.

### The correction works, in 8 of 8 runs

| | node, mean MAE | image, mean MAE |
|---|---|---|
| NF binary EM (γ=1) | 0.2502 | 0.4173 |
| NF collision (γ=1/(K−1)) | 0.2146 | 0.4055 |
| **NF collision (γ from source)** | **0.2106** | **0.4072** |
| NF collision (γ oracle, diagnostic) | 0.2311 | 0.4072 |

Going from γ=1 to source-measured γ lowers MAE in **every single run** — all six
graph transfers (−0.034 to −0.046) and both vision shifts (−0.019, −0.001). The
proof's correction is not cosmetic: on graph it removes 16% of the binary model's
error.

### Two honest caveats

**γ does not transfer as well as it needs to.** The means in the table above look
close on graph, but per group they are not: correlation between source and target
γ is only **+0.262** across the 30 graph groups, and the vision estimate is
nearly flat (0.873–0.973) while the truth is wildly heterogeneous
(0.119–0.974) — ConvNeXt and the two transformer families collide at completely
different rates under shift, and the source split cannot see it. The mechanism is
visible in the diagnostics: source pseudo-labels are near-perfect (β̂ = 0.995 on
MNIST), so the *only* eligible both-wrong pairs are the genuinely ambiguous
items, which are precisely the ones everybody gets wrong the same way. 467
eligible pairs, selected for collusion. Under shift, errors spread to ordinary
items where mistakes are idiosyncratic, and γ falls.

**Oracle γ scores *worse* than source γ on MAE, in both domains.** That is not a
bug and it is worth stating plainly: MAE here is dominated by a common positive
bias, and a larger γ makes an observed agreement weaker evidence of correctness,
pushing every α̂ down. Since γ̂_source > γ_target, the overestimate cancels bias
it was not modelling. On *ranking* the oracle does win on graph (rho +0.791 vs
+0.775). The correct reading is that γ mostly sets the **level**, not the order.

### The result that matters most: don't do the binary reduction here at all

Every member of the binary family loses to the multiclass estimator it is built
on top of — node: best NF 0.2106 vs PoolEval 0.1531; image: best NF 0.4055 vs
Dawid–Skene 0.2992.

This is structural, not a tuning failure. `pooleval/latent.py` never collapses
the pool: it keeps the full multiclass observation matrix, so a collision is
*directly observed* — two models emitting the same wrong class land on the same
key of the per-item score. The binary formulation reduces the pool to
`C[j,i] = 1{model j agrees with the pseudo-label}` and throws away *which* wrong
answer each model gave; `γ_g` is one scalar per group trying to summarise what
the full matrix records exactly. Fitting γ recovers a large part of that loss
(16% on graph) but cannot recover all of it.

So the experiment supports the proof and bounds its scope:

- The collision correction is **necessary** wherever the binary reduction is
  used — γ=1 is badly wrong for K>2, and it is wrong in a measurable direction.
- The binary reduction is **worth making only when the answer space is
  unbounded**. In Text2SQL an "answer" is an execution result table, there is no
  finite class set to index, and collapsing to agreement-with-pseudo-label is a
  genuine simplification. In image and node classification the class set is
  closed and small, the full matrix is free, and reducing to a binary matrix is
  pure information loss.

Reproduce with `python experiments/run_domain_{graph,vision}.py`; γ diagnostics
land in the `gamma` field of `results/domain_{graph,vision}.json`.

**Follow-up.** [`multiclass_ds.md`](multiclass_ds.md) takes the conclusion of
this section seriously and asks what the right closed-label-space estimator
actually is. Short answer: textbook multiclass Dawid–Skene, closed form in both
steps — but its *implied* collision rate `1/(K-1)` is wrong by 1.9×–6.4× on
these same pools, and fixing that with per-model confusion matrices beats every
estimator in the tables above on vision (MAE 0.0872, rho +0.914, top-1 correct in
both shifts).
