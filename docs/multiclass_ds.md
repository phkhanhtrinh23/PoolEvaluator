# Estimating model accuracy without labels when the answer is one of K classes

This note explains, from scratch, what changes when you move PoolEval from
Text2SQL to image and node classification, and what the right estimator is once
you get there. It is written to be readable without having read the proof.

Everything here is measured on the real trained pools from
[`docs/domain_ports.md`](domain_ports.md) — 15 GNNs across six citation-network
transfers, and 15 image classifiers across two digit shifts. Reproduce with:

```bash
python experiments/run_ds_assumption.py     # the assumption tests below
python experiments/run_domain_graph.py      # the estimator comparison
python experiments/run_domain_vision.py
```

---

## 1. The problem

You have `M` models and `N` test items. **You do not have the labels.** Each
model emits one answer per item, so all you hold is an `M × N` grid:

```
             item 1   item 2   item 3   ...
model 1        cat      dog      cat
model 2        cat      dog      fox
model 3        cat      bird     cat
```

You want each model's accuracy. The only signal available is *who agrees with
whom*. The whole field is variations on one idea: **guess the truth from the
consensus, score each model against that guess, then use the new scores to make a
better guess.** That loop is EM.

The danger is equally simple: if the models are wrong *together*, the consensus
is confidently wrong and every model looks good. That is the **gauge trap**, and
it is the thing every design decision below is fighting.

---

## 2. Why Text2SQL needs a "collision rate" and classification does not

In Text2SQL an answer is an executed SQL result table. There is no finite list of
possible answers — the space is effectively unbounded. So the formulation in
`Trinh_proof.tex` reduces each model to a single yes/no per item:

```
C[j,i] = 1  if model j agrees with the current pseudo-label on item i
         0  otherwise
```

That reduction is convenient, but it **throws away which answer was given**. And
that creates an ambiguity you cannot resolve from `C` alone:

| what really happened | what `C` records |
|---|---|
| model right, pseudo-label right | `C = 1` |
| model wrong, pseudo-label wrong, **same** wrong answer | `C = 1` ← same symbol! |
| model right, pseudo-label wrong | `C = 0` |
| model wrong, pseudo-label wrong, **different** answers | `C = 0` ← same symbol! |

Rows 1 and 2 are indistinguishable, as are rows 3 and 4. To write down the
likelihood you must supply the missing piece as a parameter:

> **γ_g** = P(a wrong model and a wrong pseudo-label give the *same* wrong answer)

That is exactly the collision-aware correction. `γ = 1` says two wrong answers
always coincide — forced when there are only two classes, false otherwise.

**In classification you never have to make that trade.** The class set is closed
and small (K = 5 for the citation graphs, K = 10 for digits), so you can keep the
full `M × N` grid of labels. Then:

- both wrong on the **same** class → two models land on the same entry, their
  votes reinforce, and you *see* it;
- both wrong on **different** classes → two distinct entries, the mass splits,
  and you *see* that too.

Nothing is marginalised away, so nothing needs a parameter. γ is a nuisance
parameter *created by* the binary reduction, not a fact about the world.

This is measurable, and it is the first result of this note. Every binary variant
loses to a multiclass estimator built on the same data:

| | node, mean MAE | image, mean MAE |
|---|---|---|
| binary EM, γ = 1 | 0.2502 | 0.4173 |
| binary EM, γ measured on source | 0.2106 | 0.4072 |
| **multiclass (best below)** | **0.1531** | **0.0872** |

Fitting γ does **not** close that gap. Under a controlled ablation (anchors and
β initialisation held fixed, only γ varied) source-measured γ beats γ=1 in just
1 of 8 runs, and the whole γ range spans under 0.006 MAE — because γ enters only
as `(1−β)γ`, and with β ≈ 0.90 that term is swamped by `αβ`. γ earns its keep
where the pseudo-label is weak, which is Text2SQL's regime (MAE 18.3 → 11.3
there), not this one. See `experiments/run_gamma_ablation.py`.

**If your answer space is enumerable, do not binarise** — and if you do binarise
here, γ will not rescue it.

---

## 3. One-coin Dawid–Skene, derived

This is the classic multiclass estimator (Dawid & Skene, 1979). Two sets of
unknowns:

- `z_i ∈ {1..K}` — the true label of item `i` (latent)
- `α_j ∈ (0,1)` — the accuracy of model `j` (what we want)

and one modelling assumption, the "one coin":

```
P(model j says k | truth is c)  =  α_j            if k = c
                                =  (1 − α_j)/(K−1) if k ≠ c
```

In words: *model j is right with probability α_j, and when it is wrong it picks
uniformly at random among the K−1 other classes.*

### The E-step (guess the truth)

By Bayes' rule, with `p_c` the class prior:

```
τ_i(c)  ∝  p_c · Π_j P(obs[j,i] | z_i = c)
```

Split that product into the models that voted for `c` and those that did not:

```
Π_j (…)  =  [ Π_j (1−α_j)/(K−1) ]  ·  Π_{j: obs[j,i] = c}  α_j(K−1)/(1−α_j)
              └── same for every c, cancels ──┘
```

The first bracket does not depend on `c`, so it vanishes when you normalise.
Taking logs leaves something remarkably simple:

```
log τ_i(c)  =  log p_c  +  Σ_{j : obs[j,i] = c}  w_j        where  w_j = log[ α_j(K−1) / (1−α_j) ]
```

**It is a weighted vote.** Each model adds a weight `w_j` to whichever class it
picked, and you softmax. The whole model reduces to: *how many votes is model j
worth?*

And the derived weight has a property worth pausing on:

| α_j (K=5) | weight `w_j` | meaning |
|---|---|---|
| 0.90 | +3.58 | strong vote |
| 0.60 | +1.79 | ordinary vote |
| **0.20 = 1/K** | **0.00** | chance-level model, **ignored entirely** |
| 0.15 | −0.35 | worse than chance, votes *against* its own answer |

`w_j = 0` exactly at `α_j = 1/K`, because `α(K−1)/(1−α) = 1 ⟺ α = 1/K`. A model
at chance contributes nothing; a model below chance is *informative in reverse*.
This falls out of the model — nobody chose it.

> Note for this codebase: `pooleval/latent.py:61-66` instead sets the vote weight
> to `clip(α, 0.05, 0.99)`, which is always positive and never zero. That was a
> deliberate choice to avoid a runaway feedback loop, but it means sub-chance
> models keep a positive vote when the correct weight is negative. On
> MNIST→SVHN, where most models are *below* chance, that is not a small detail.

### The M-step (rescore the models)

```
α_j  =  (1/N) Σ_i τ_i( obs[j,i] )          p_c  =  (1/N) Σ_i τ_i(c)
```

"How often did model j's answer land on what we now believe the truth is." Both
are ratios of expected counts — **closed form**, no optimiser, no root-finding.

So the answer to *"is it closed form?"* is **yes, and more cleanly than the
binary case**: the binary+γ model loses its closed form in β and needs a
polynomial root or a 1-D maximisation; the multiclass model keeps closed forms
everywhere. Implementation: `pooleval/domains/multiclass_ds.py:one_coin_ds`.

---

## 4. The hidden assumption, and why it is false

That one line `(1 − α_j)/(K−1)` is doing enormous work. It asserts **two** things:

1. **Uniformity** — a model's errors spread evenly over the K−1 wrong classes.
2. **Independence** — two wrong models therefore collide with probability
   exactly `1/(K−1)`.

Note the consequence: **one-coin DS has a γ after all.** It just never says so.
Its implied collision rate is `1/(K−1)` — 0.250 for the graphs, 0.111 for digits.

Both claims are testable by pure counting: take the raw predictions and the
withheld gold, no EM, no pseudo-labels, nothing that could be an artefact of a
method. `experiments/run_ds_assumption.py` does exactly that:

| | K | DS assumes | **measured** | ratio |
|---|---|---|---|---|
| **node** — mass on a model's single favourite error | 5 | 0.250 | **0.465** | 1.9× |
| **node** — collision, models of *different* provenance | 5 | 0.250 | **0.702** | 2.8× |
| **node** — collision, models of the *same* provenance | 5 | 0.250 | **0.840** | 3.4× |
| **image** — mass on a model's single favourite error | 10 | 0.111 | **0.627** | 5.6× |
| **image** — collision, *different* provenance | 10 | 0.111 | **0.412** | 3.7× |
| **image** — collision, *same* provenance | 10 | 0.111 | **0.706** | 6.4× |

Read the three rows per domain in order, because they separate the two failures:

- Errors are **not uniform** — a model puts 1.9×–5.6× more mass on its favourite
  mistake than chance allows. Digits are confusable in specific ways (4↔9), and
  citation classes overlap in specific ways.
- Because of that alone, even models from *different* families collide at
  2.8×–3.7× the assumed rate. **Non-uniformity by itself breaks independence.**
- And *same-family* models collide more still — 0.840 vs 0.702, 0.706 vs 0.412.
  Shared architecture adds correlated error **on top of** shared task difficulty.

So there are two distinct violations, and they need two distinct fixes.

---

## 5. Fix #1: full Dawid–Skene (a confusion matrix per model)

Drop uniformity. Give every model its own `K × K` table:

```
P(model j says k | truth is c)  =  π_j[c, k]
```

- **E-step:** `log τ_i(c) = log p_c + Σ_j log π_j[c, obs[j,i]]`
- **M-step:** `π_j[c,k] = Σ_i τ_i(c)·1[obs[j,i]=k] / Σ_i τ_i(c)` — row-normalised
  expected counts, **still closed form**.

Now the model can learn "this ResNeXt calls 4s 9s" instead of pretending the
error is spread evenly. Implementation: `multiclass_ds.py:full_ds`.

### It helps — in 8 of 8 runs

| | node, mean MAE | node, ρ | image, mean MAE | image, ρ | image top-1 |
|---|---|---|---|---|---|
| DS one-coin (exact) | 0.1754 | +0.889 | 0.2977 | +0.055 | 0.50 |
| **DS full (confusion)** | **0.1658** | **+0.898** | **0.0872** | **+0.914** | **1.00** |

Modest on graph (−5.5%, every transfer), enormous on vision (**3.4× lower MAE**,
and the ranking goes from useless to near-perfect). That gap is exactly what the
assumption table predicts: non-uniformity is 1.9× on graph but 5.6× on vision, so
there is far more for a confusion matrix to recover on vision.

### The case that makes it vivid: MNIST → SVHN

This is the shift where *every* other method collapses. True accuracies are
0.062–0.355, i.e. mostly **below** the 0.10 chance line. Here is what the two
estimators say, model by model:

| model | true | one-coin says | full says | fraction of items given its top class |
|---|---|---|---|---|
| ConvNeXtT-s1 | 0.355 | 0.083 | **0.667** | 0.243 |
| ConvNeXtT-s0 | 0.330 | 0.118 | **0.667** | 0.195 |
| ViT-Tiny-s0 | 0.094 | 0.799 | 0.230 | 0.780 |
| ResNeXt50-s1 | 0.066 | **0.959** | 0.143 | 0.957 |
| ResNeXt50-s0 | 0.063 | **0.974** | 0.138 | 0.982 |
| RegNetY8GF-s0 | 0.062 | **0.984** | 0.155 | 0.978 |

Look at the last column. Under this shift the ResNeXt and RegNet models have
**collapsed**: they emit the same single digit for 96–98% of the test set. They
are barely better than a constant function.

One-coin DS scores them at **0.96–0.98 — nearly perfect.** Why? Six models
collapsed onto the same class, so they agree with each other almost always, and
unanimous agreement is the only thing one-coin DS knows how to read. It is the
gauge trap in its purest observable form: *a pool of constant predictors is
perfectly self-consistent.*

Full DS scores them at 0.14–0.16, and correctly puts the three ConvNeXts on top.
The reason is visible in the confusion matrix: a collapsed model has **near-
identical rows** — it says `k*` whatever the truth is — so `π_j[c, k*]` is high
for *every* `c`, its vote carries no information about `c`, and the E-step
discounts it automatically. Quantitatively, the spread of a model's confusion
rows correlates with its true accuracy at **ρ = +0.982**.

The downstream effect: pseudo-label accuracy rises from **0.062 → 0.307**, and
the estimator's ranking from **ρ = −0.886 → +0.829**.

`docs/domain_ports.md` originally recorded this shift as unsalvageable —
"all methods fail, this is the identifiability theorem, empirically." That
conclusion was too strong. It was a *uniform-error* failure, not an
identifiability failure, and dropping uniformity largely fixes it.

---

## 6. What full DS still does not fix

It relaxes uniformity but keeps **conditional independence**: given the true
label, models are assumed to err independently. The assumption table says that is
false — same-provenance models collide at 0.840 vs 0.702 on graph, 0.706 vs 0.412
on vision. Three seeds of one architecture are not three independent witnesses.

And this shows in the results. On graph, where correlation is the dominant
violation, PoolEval's **group discount** — which explicitly counts same-group
models voting the same class and shrinks them toward one vote — still wins on
MAE:

| node, mean over 6 transfers | MAE | ρ |
|---|---|---|
| PoolEval (group discount, anchors on) | **0.1531** | +0.782 |
| DS full (confusion matrices) | 0.1658 | **+0.898** |
| DS one-coin | 0.1754 | +0.889 |

The two fixes are complementary and **no estimator here implements both**:

| violation | fixed by | closed form? |
|---|---|---|
| errors are not uniform | full DS confusion matrices | ✅ yes |
| models err *together* | PoolEval's group discount `u_g` | ❌ no — it is a weighted-vote heuristic, not a likelihood |

That is the open problem this note leaves behind: a confusion-matrix model with a
provenance-group structure, which would fix both at once. It would not stay
closed form — correlated-error models generally do not — but the concavity
argument that rescued the binary β-step may transfer.

---

## 7. Which estimator wins in which regime

The choice between PoolEval's single reliability scalar and full DS's confusion
matrices is a bias/variance trade, and one quantity sets the crossover:
**observations per free parameter.**

| | free parameters | grows with K? |
|---|---|---|
| PoolEval (anchor, collision-aware) | `M` α + 1 β + `G` group loadings (+ `G` γ, *fixed* from source) | **no** |
| full DS | `M·K(K−1) + (K−1)` | **quadratically** |

So full DS should win when data is plentiful relative to `K`, and lose when the
target set is small, `K` is large, or rare classes starve individual confusion
rows. All three are measured by `experiments/run_regime_scenarios.py`.

> One simplification worth stating up front: **γ is nearly inert in these
> domains** (§2), so "PoolEval with anchor and collision-awareness" behaves
> essentially like "PoolEval with anchor" here. The anchor is what matters.

### Scenario 1 — small unlabelled target set

Real pool, items subsampled, 5 repetitions (MAE / ρ):

| N | obs per DS-full param | PoolEval | DS one-coin | DS full |
|---|---|---|---|---|
| 50 | 0.56 | 0.1937 / −0.077 | **0.0427 / +0.960** | 0.0898 / +0.798 |
| 100 | 1.11 | 0.2229 / −0.412 | **0.0472 / +0.949** | 0.0709 / +0.838 |
| 200 | 2.22 | 0.1942 / −0.234 | 0.0447 / +0.933 | **0.0287 / +0.965** |
| 1000 | 11.1 | 0.2260 / −0.541 | 0.0351 / +0.989 | **0.0220 / +0.997** |
| 2007 | 22.3 | 0.2420 / −0.746 | 0.0356 / +0.996 | **0.0216 / +1.000** |

*(MNIST→USPS, K=10, M=15, 1350 full-DS parameters.)*

**The crossover is at roughly 2 observations per parameter.** Below it the
confusion matrices are noise that the E-step then trusts. On the graph pool
(K=5, only 300 parameters) full DS never overfits even at N=50 — so the
crossover is driven by `K`, not by `N` alone.

Note the shape of PoolEval's anchor: its weight is `s/(N+s)`, so it dominates
exactly when data is scarce and fades when it is not (0.36% at N=8935). That is
structurally a small-N device, which is the right design — see
[`domain_ports.md`](domain_ports.md) §1, where the same property reads as a
*defect* because the Text2SQL anchors were tuned for N ≈ 400.

### Scenario 2 — many classes

`K` beyond what these datasets offer, so this one is **simulated**: N fixed at
400, M=9, three provenance groups, accuracies 0.20–0.50, within-group collusion
0.6.

| K | obs per DS-full param | PoolEval | DS one-coin | DS full |
|---|---|---|---|---|
| 5 | 20.0 | 0.0238 | 0.0578 | **0.0052** |
| 10 | 4.44 | 0.0197 | 0.0244 | **0.0019** |
| 20 | 1.05 | 0.0664 | 0.0257 | **0.0014** |
| 50 | 0.16 | 0.0783 | 0.0285 | **0.0134** |
| 100 | 0.04 | 0.0835 | **0.0312** | 0.0802 |

Full DS wins comfortably up to K≈50 and then falls off a cliff — MAE 0.0134 →
0.0802 between K=50 and K=100 — while one-coin barely moves (0.0244 → 0.0312)
because its parameter count does not depend on `K` at all. Laplace smoothing
holds full DS up further than raw counting would; it is still winning at 0.16
observations per parameter.

### Scenario 3 — imbalanced classes

The clearest failure. Real graph pool, resampled to a power law with **N held
fixed at 800**, so nothing changes except the class distribution:

| imbalance | rarest class n | PoolEval ρ | DS one-coin ρ | DS full ρ |
|---|---|---|---|---|
| 1.0× | 160 | +0.741 | +0.928 | **+0.940** |
| 11.3× | 40 | +0.752 | **+0.920** | +0.873 |
| 134.8× | 5 | +0.659 | **+0.861** | +0.607 |
| 385.5× | 2 | +0.669 | **+0.925** | **+0.559** |

**Full DS's ranking collapses (+0.940 → +0.559) while one-coin is flat.** The
mechanism: the row `π_j[c,·]` for a rare class is estimated from 2–5 items, but
it is then used in the E-step for *every* item where any model votes that class.
A few starved rows poison the whole posterior. A single scalar `α_j` cannot
fragment that way, so it is immune.

The same pattern appears on vision (full DS +0.999 → +0.857, one-coin +0.989 →
+0.899), though less sharply.

### Scenario 4 — a heterogeneous pool (one model per family)

This one has a clean theoretical answer, and the code confirms it exactly.

PoolEval's *only* structural advantage over one-coin DS is the provenance
discount

```
disc_m = 1 / (1 + u_g · (n_{g,k} − 1))
```

where `n_{g,k}` counts models of group `g` voting for class `k`. **If every model
comes from a different family, every group is a singleton, so `n_{g,k} ≡ 1` and
`disc_m ≡ 1`.** The discount multiplies by one. It does nothing.

Verified directly — turning `use_correlation` on and off in a one-model-per-family
pool:

| pool | one per family | 3+2 near-clones |
|---|---|---|
| graph A→C | `max|on−off| = 0.00e+00` **inert** | 4.48e-02 active |
| graph D→A | `0.00e+00` **inert** | 5.27e-02 active |
| svhn→mnist | `0.00e+00` **inert** | 1.89e-02 active |

Bit-for-bit identical. So in a fully heterogeneous pool PoolEval collapses to a
weighted-vote approximation of one-coin DS — and a *worse* one, because its vote
weight is `clip(α, 0.05, 0.99)` rather than the derived `log[α(K−1)/(1−α)]` (§3).

Full DS, meanwhile, loses nothing and *gains*: its conditional-independence
assumption is closest to true precisely when no two models share a backbone. The
one violation it cannot represent is the one that just disappeared.

Measured at fixed M=5, varying only composition (MAE / ρ):

| pool | composition | PoolEval | DS one-coin | DS full |
|---|---|---|---|---|
| graph A→C | 5 families × 1 | 0.0929 / +0.300 | 0.1317 / +0.900 | 0.1237 / **+1.000** |
| | 2 families (3+2) | 0.1409 / +0.367 | 0.1784 / +0.726 | 0.1736 / +0.726 |
| graph D→A | 5 families × 1 | 0.1840 / +0.100 | 0.2202 / **+1.000** | 0.2075 / **+1.000** |
| | 2 families (3+2) | 0.2159 / −0.233 | 0.2669 / +0.267 | 0.2607 / +0.200 |
| svhn→mnist | 5 families × 1 | 0.0840 / +0.900 | 0.1155 / +0.900 | 0.1061 / +0.900 |
| | 2 families (3+2) | 0.1243 / +0.367 | 0.1457 / +0.833 | 0.1409 / +0.967 |

Two things to read off:

- **Heterogeneity helps everyone.** Every method improves going from 2 families
  to 5. Five independent witnesses beat five near-clones — which is a pool-design
  lesson independent of estimator choice.
- **It helps the DS variants most.** Their ranking goes to ρ = +0.90–1.00 in the
  heterogeneous pools, while PoolEval's stays poor (+0.100 to +0.300 on graph).
  PoolEval keeps an MAE edge, but that is bias-shaving again, not signal.

**So: full DS wins, and by more than in a homogeneous pool.** A heterogeneous
pool removes PoolEval's reason to exist and simultaneously repairs full DS's
weakest assumption.

The corollary is worth stating for pool design: if you can *choose* your pool,
choose one model per family — it improves every estimator, and it means you no
longer need the provenance machinery at all. The group discount is insurance
against a pool you did not get to design.

> **Do not read the MAE column on its own here.** In every one of these rows MAE
> equals the bias exactly — every method over-estimates every model — so MAE is
> measuring the size of a shared offset, not how well an estimator separates
> models. Strip the offset and PoolEval loses 3 of 4 configurations. §8 works
> through why.

*(Caveat: with M=5, Spearman is computed on five points and is therefore coarse —
+0.100 and +0.300 differ by one swap. The inertness check is the exact part.)*

### Summary

| regime | use | why |
|---|---|---|
| small N | **one-coin** (or PoolEval) | full DS needs ≳2 observations per parameter |
| large K | **one-coin** (or PoolEval) | full DS is O(K²) parameters; data is O(N) |
| imbalanced classes | **one-coin** | starved rare-class rows drive the E-step everywhere |
| large N, small K, balanced | **full DS** | it can afford the parameters, and non-uniform errors are real |
| heterogeneous pool (1 per family) | **full DS** | PoolEval's group discount is provably inert; DS's independence assumption becomes true |

**Two caveats.** First, PoolEval's *baseline* ranking on these pools is lower to
begin with (ρ +0.74 vs +0.93 balanced), so "robust to imbalance" partly means
"already worse". In the robust regimes **one-coin DS, not PoolEval, is usually
the better choice** — it is as robust and starts from a much better baseline.
Second, MAE and ranking diverge: full DS keeps an MAE edge even where its ranking
collapses. If you want a *number*, full DS; if you want an *order* under
imbalance, one-coin.

## 8. Why PoolEval under-spreads and DS over-spreads

This is the single most important thing to understand before reading any MAE
column in this document, because it explains a result that otherwise looks
contradictory: **PoolEval wins MAE almost everywhere and loses ranking almost
everywhere.**

### The observation: MAE here *is* bias

Decomposing the scenario-4 numbers (fixed M=5):

| pool / composition | method | MAE | bias | **MAE with bias removed** | ρ |
|---|---|---|---|---|---|
| graph A→C, 5 fams | PoolEval | 0.0929 | +0.0929 | 0.0683 | +0.300 |
| | DS one-coin | 0.1317 | +0.1317 | **0.0356** | +0.900 |
| | DS full | 0.1237 | +0.1237 | 0.0456 | **+1.000** |
| svhn→mnist, 5 fams | PoolEval | 0.0840 | +0.0840 | 0.0451 | +0.900 |
| | DS one-coin | 0.1155 | +0.1155 | **0.0325** | +0.900 |
| | DS full | 0.1061 | +0.1061 | 0.0352 | +0.900 |
| svhn→mnist, 3+2 | PoolEval | 0.1232 | +0.1232 | 0.0180 | +0.100 |
| | DS one-coin | 0.1399 | +0.1399 | 0.0094 | +0.900 |
| | DS full | 0.1362 | +0.1362 | **0.0092** | **+1.000** |

**MAE equals bias to four decimals in every row.** Every method over-estimates
every model — that is the gauge trap, and it is common to all of them. So the
MAE column measures only *the size of a shared offset*; it says nothing about
whether an estimator can tell models apart. Remove the offset and PoolEval loses
3 of 4 configurations.

The remaining question is why the offsets differ, and the answer is spread.

### Spread against the truth

| pool | true spread | PoolEval + anchor | PoolEval no anchor | DS one-coin | DS full |
|---|---|---|---|---|---|
| graph A→C | 0.466 | 0.501 | 0.595 | 0.597 | 0.634 |
| graph D→A | 0.308 | 0.413 | 0.514 | 0.565 | 0.612 |
| svhn→mnist | 0.527 | 0.357 | 0.542 | 0.569 | 0.594 |

Three mechanisms produce this, and they can be measured separately.

### Mechanism 1 — the vote-weight transform (compresses PoolEval)

Both estimators aggregate votes; they weight them differently.

| α | PoolEval `clip(α, 0.05, 0.99)` | DS `log[α(K−1)/(1−α)]`, K=5 |
|---|---|---|
| 0.05 | 0.050 | **−1.558** |
| **0.20 = 1/K** | 0.200 | **0.000** |
| 0.40 | 0.400 | +0.981 |
| 0.70 | 0.700 | +2.234 |
| 0.99 | 0.990 | +5.981 |

PoolEval's weight is the accuracy itself, clipped: the best model can out-vote
the worst by at most **19.8×**. DS's weight is a log-odds, so the same two models
differ by 7.54 in log space — a vote ratio of **e^7.54 ≈ 1881×**, roughly 95×
more separation.

And note the two rows in bold. DS's weight is **exactly zero at chance accuracy**
(`α(K−1)/(1−α) = 1 ⟺ α = 1/K`) and **negative below it** — a sub-chance model
votes *against* its own answer. PoolEval's floor is +0.05, always positive.

The consequence for spread: with a bounded weight, weak models keep pulling the
consensus toward their answers. Everyone is then scored against a target that
sits between the good models and the bad ones, so good models score lower than
they should and bad ones higher. **The estimates are squeezed toward each
other.** DS's aggressive weighting lets good models dominate the consensus, which
preserves — and then amplifies — the separation.

This is not an accident, and the code says so. From `pooleval/latent.py:61-66`:

> ```
> # reliability weight = estimated accuracy, always positive and bounded in
> # (0,1). Bounded (not logit) avoids the high-a -> peaked-consensus -> higher-a
> # runaway; keeping it strictly positive avoids the opposite collapse where
> # sub-0.5 models get zero weight and the consensus loses all information.
> ```

PoolEval **deliberately damps** the feedback loop that DS runs at full strength.
The trade documented here is exactly the one that comment is making, measured.

### Mechanism 2 — the anchor (pulls spread toward the *prior's* spread)

The anchor fuses the data estimate with the source prior. It does not compress
unconditionally — it pulls the estimated spread toward whatever spread the prior
has:

| pool | prior spread | true spread | effect |
|---|---|---|---|
| graph A→C | 0.524 | 0.466 | slight inflation (0.501) |
| graph D→A | 0.502 | 0.308 | inflation (0.413) |
| svhn→mnist | 0.404 | 0.527 | **compression** (0.357) |
| MNIST→USPS | **0.012** | 0.565 | **severe compression** |

The last row is the pathological case, and it explains PoolEval's negative ρ on
that pool. Every model reaches ≈0.99 on MNIST, so the prior is nearly a constant
— and worse, it is *anti-correlated* with target accuracy:

| pool | ρ(prior, true target accuracy) |
|---|---|
| graph A→C | +0.781 |
| graph D→A | +0.924 |
| svhn→mnist | +0.832 |
| **MNIST→USPS** | **−0.625** |

**Shrinkage preserves ranking only if the prior ranks correctly.** On graph it
roughly does, so the cost is modest. On MNIST→USPS it ranks *backwards*, so
shrinking toward it actively destroys the ordering. Turning the anchor off
recovers both the spread and the ranking every time:

| pool | ρ with anchor | ρ without |
|---|---|---|
| graph A→C | +0.864 | **+0.964** |
| graph D→A | +0.654 | **+0.814** |
| svhn→mnist | +0.868 | **+0.982** |

### Mechanism 3 — EM feedback (inflates DS)

DS's over-spread is not present at initialisation; the EM loop builds it. A model
slightly above average gets more weight, the consensus moves toward it, so it
agrees with the consensus more, so its α rises, so it gets more weight again.
Watching that run:

| EM iterations | est. spread (graph A→C) | best/worst vote ratio | ρ |
|---|---|---|---|
| 1 | 0.543 | 15× | +0.939 |
| 2 | 0.588 | 20× | +0.939 |
| 3 | 0.594 | 21× | +0.939 |
| 5 | 0.596 | 22× | +0.961 |
| 200 | 0.597 | 22× | +0.961 |

*(true spread 0.466)*

The spread grows monotonically, crosses the truth between iterations 1 and 2, and
settles above it. Same on svhn→mnist: 0.465 → 0.569 against a true 0.527.

**But notice the ρ column: it improves as the spread inflates** (+0.939 →
+0.961). The feedback is monotone in the initial signal, so it exaggerates
differences without reordering them. That is the crux — **over-spreading is a
level error, not an ordering error.**

### Why this decides which metric to trust

| | spread | MAE | ranking |
|---|---|---|---|
| PoolEval (anchored) | under | **good** | poor |
| DS | over | poor | **good** |

A method that compresses toward the middle gets closer to every model on average
— that is why PoolEval wins MAE — but it flattens the very differences you need
to choose a model. A method that exaggerates differences overshoots the level but
keeps the order.

The asymmetry that matters in practice: **a constant offset is removable the
moment you have one labelled anchor; a wrong ranking is not recoverable at all.**
So if the goal is model selection, read ρ and DS wins. If the goal is to report
an accuracy number, note that all three are +0.08 to +0.19 too high — none is
usable uncorrected, and the MAE ordering among them is a comparison of failures,
not a comparison of successes.

The obvious synthesis — DS's log-odds weighting for ordering, plus a
bias-correction for the level — is not implemented here, and is a cleaner target
than either estimator alone. Reproduce all of the above with
`experiments/run_regime_scenarios.py`.

## 9. What to actually use

| your setting | use | why |
|---|---|---|
| answer space unbounded (Text2SQL) | binary EM **with** collision-aware γ | you must reduce; γ=1 is wrong for anything but K=2 |
| closed K classes, models may collapse under shift | **full DS** | non-uniformity dominates; recovers MNIST→SVHN |
| closed K classes, pool has near-clones / shared backbones | **PoolEval with group discount**, anchors off | correlation dominates |
| you have reliable source-domain accuracies | add them as a Beta prior | but see `domain_ports.md` §1 — large N silences it |

Short version of the whole note:

0. MAE and ranking disagree throughout, because every estimator shares the same
   positive bias and MAE only measures its size. PoolEval under-spreads (bounded
   vote weight + shrinkage to the prior) so it wins MAE; DS over-spreads
   (log-odds weight + EM feedback) so it wins ranking. Offsets are correctable,
   orderings are not — see §8.

1. Don't binarise if you can enumerate the classes — γ only exists to repair
   information the reduction destroyed.
2. The multiclass model is closed form in both steps, and *derives* the vote
   weight (zero at chance, negative below) that heuristics usually guess.
3. One-coin DS is not neutral: it silently assumes a collision rate of `1/(K−1)`,
   which is wrong by 1.9×–6.4× on real pools.
4. Confusion matrices fix the uniformity half, cheaply and closed form, and turn
   one catastrophic failure case into a solved one.
5. Nothing here fixes correlated error except the group discount, which is not
   yet a likelihood. That is the next piece of work.
