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

Fitting γ recovers a lot of what the reduction cost — 16% of the error on
graph, in 8 of 8 runs — but never all of it. **If your answer space is
enumerable, do not binarise.**

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

## 7. What to actually use

| your setting | use | why |
|---|---|---|
| answer space unbounded (Text2SQL) | binary EM **with** collision-aware γ | you must reduce; γ=1 is wrong for anything but K=2 |
| closed K classes, models may collapse under shift | **full DS** | non-uniformity dominates; recovers MNIST→SVHN |
| closed K classes, pool has near-clones / shared backbones | **PoolEval with group discount**, anchors off | correlation dominates |
| you have reliable source-domain accuracies | add them as a Beta prior | but see `domain_ports.md` §1 — large N silences it |

Short version of the whole note:

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
