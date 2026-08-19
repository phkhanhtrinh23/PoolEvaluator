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

MNIST → SVHN is reported in the JSON but is a **degenerate regime**: true target
accuracy is 0.062–0.355 with K = 10, so most of the pool is *below chance*. Every
label-free method fails there with bias between +0.46 and +0.85 and negative rank
correlation. This is not a tuning problem — it is the Dawid–Skene identifiability
theorem. The likelihood is invariant under flipping the latent labels and
replacing each `a_m` with its complement, so "the crowd is accurate" and "the
crowd is anti-accurate" fit the data equally well; only the assumption that
annotators beat chance breaks the tie, and here that assumption is false.

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
