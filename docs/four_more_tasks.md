# Four more tasks: link prediction, image captioning, knowledge-graph completion

This note reports four new experiments and explains what they mean. It assumes
you have read nothing else in this repository.

Reproduce with:

```bash
python experiments/run_domain_linkpred.py    # link prediction
python experiments/run_domain_caption.py     # image captioning
python experiments/run_domain_kgc.py         # knowledge-graph completion (2 datasets)
```

---

## 1. The problem all four tasks share

You have several trained models. You want to know **how accurate each one is on
some new data** — but that new data has **no labels**, and labelling it is
expensive. So you cannot just measure accuracy.

The only thing you can do is look at what the models *say* and compare them to
each other.

The classroom picture used throughout this repo:

> Fifteen students sit a test. **The answer key is lost.** You rebuild the key
> from what the students agreed on, then grade everyone against your guess.

Models are the students, the unlabelled data is the test, the guessed key is
called the **pseudo-label**, and the loop of *guess key → grade students → give
the good ones more say → guess a better key* is the **EM algorithm**.

The obvious danger: **if the class is wrong together, the guessed key inherits
the mistake and everybody gets credit for it.** Every method below over-estimates
accuracy for exactly this reason. It has a name in this repo: the *gauge trap*.

### The methods being compared

| method | one-line description |
|---|---|
| **B1 Independent** | ignore the pool entirely; predict the accuracy the model had on labelled source data. A "no consensus" control. |
| **B2 Majority** | plain majority vote builds the key; grade against it. |
| **B3 / DS one-coin** | Dawid–Skene: like majority vote, but better models get more voting power, learned by EM. Each model gets **one number** (its accuracy). |
| **DS full** | Dawid–Skene with a whole **confusion matrix** per model — it can learn "this model mistakes 4 for 9" rather than assuming errors are spread evenly. |
| **PoolEval** | this repo's estimator: DS-style voting plus a *provenance discount* (models from the same family are treated as fewer independent votes) plus an *anchor* toward source accuracy. |
| **NF binary / NF collision** | the formulation from `Trinh_proof.tex`: collapse the pool to "agrees with the key / doesn't", then model the chance two wrong answers coincide with a parameter γ. |
| **DoC / ATC** | ignore the pool; predict accuracy from how *confident* each model is. |

### The two numbers reported

- **MAE** — average distance between estimated and true accuracy. Lower is better.
- **ρ (Spearman)** — did you rank the models correctly? +1 is perfect order,
  0 is random, negative is backwards.

They often disagree, and when they do, **ρ is usually what matters**: a constant
offset can be removed later with a few labelled examples, a wrong ordering
cannot. (This is worked through in [`multiclass_ds.md`](multiclass_ds.md) §8.)

---

## 2. The four tasks, and why these four

The tasks were chosen to span one axis: **how many possible answers are there?**
That number, called `K`, turns out to decide almost everything.

| task | the question asked | K | data |
|---|---|---|---|
| **Link prediction** | "is there an edge between these two nodes?" | **2** | ACMv9 / Citationv1 / DBLPv7, 6 cross-graph transfers |
| **Image captioning** | "describe this image" | **unbounded** — a sentence | 1000 COCO images |
| **KGC (FB15k-237)** | "(head, relation, ?) — which entity?" | **14,541** | FB15k-237 |
| **KGC (WN18RR)** | same | **40,943** | WN18RR |

---

## 3. Link prediction (K = 2)

Pool: GCN, SAGE, GAT, GIN, MLP encoders × 3 seeds = 15 models with a dot-product
decoder. Train on one citation graph's edges, score node pairs on another.
Positives and sampled negatives in equal numbers, so chance accuracy is 0.50.

**Results (mean over 6 transfers):**

| method | MAE | ρ |
|---|---|---|
| DoC | **0.0523** | +0.002 |
| B1 Independent | 0.1241 | +0.148 |
| B4 Agreement-on-line | 0.1241 | +0.579 |
| **DS full (confusion)** | 0.1466 | **+0.602** |
| ATC-MC / ATC-NE | 0.1814 | −0.269 |
| DS one-coin | 0.1983 | −0.636 |
| PoolEval | 0.2045 | −0.329 |
| B3 Dawid–Skene | 0.2074 | −0.593 |
| B2 Majority | 0.2084 | −0.582 |
| NF collision (γ=null) | 0.2292 | −0.329 |
| NF collision (γ=source) | 0.2292 | −0.329 |
| NF binary EM (γ=1) | 0.2780 | −0.330 |

### What to notice

**Almost every consensus method ranks models BACKWARDS** (ρ from −0.27 to −0.64).
This is the worst showing for consensus anywhere in this repo, and the reason is
worth understanding.

With only two answers, agreeing with the crowd is *cheap*. Two models that both
guess "edge" agree perfectly, whether or not there is an edge. Chance agreement
between two random models is 50%, so the agreement signal is almost entirely
noise. Add that the models are close in quality (accuracies cluster in
0.55–0.68) and there is simply very little for consensus to find.

**DS full is the only consensus method that ranks usefully** (+0.602). It can
represent "this model says *edge* far too often" — a bias a single accuracy
number cannot express — and with K=2 its confusion matrix is only 2×2, so there
is plenty of data to fit it.

**The collision correction is exactly, provably useless here — as predicted.**
`Trinh_proof.tex` observes that when there are only two answers, a wrong model
and a wrong key *must* give the same wrong answer. So the collision rate γ is
forced to 1. Measured on the data: **γ = 0.999**, and the "independent errors"
value `1/(K−1)` is also **1.000**. Every γ setting is the same setting, and the
table confirms it — `γ=null` and `γ=source` are identical to four decimals
(0.2292 both). This is the one place in the repo where the correction is
guaranteed to buy nothing, and it buys nothing.

*(DoC wins MAE with ρ = +0.002 — it predicts the accuracy *level* well and orders
the models at chance. It is not a usable model-selection method here.)*

---

## 4. Image captioning (K unbounded)

Pool: four pretrained captioners — ViT-GPT2, BLIP-base, GIT-base, BLIP-large —
each decoded three ways (one beam-search, two sampled) = 12 models. A fifth
model, GIT-large, is held out as an independent verifier. 1000 COCO images:
300 keep their human captions (that is where the prior comes from), 700 have
theirs withheld for scoring.

### The new problem: what does "agree" even mean?

For classification, two models agree when they output the same class. Two
captions are never *identical*, but they can mean the same thing. So captioning
needs an **equivalence kernel** — the same job the SQL execution-result
comparison does in Text2SQL.

Ours: strip punctuation and stopwords, then call two captions equivalent when
their content words overlap enough (Jaccard ≥ threshold). Per image, all
captions plus the human references are clustered by that relation. A model is
**correct** when its caption lands in the same cluster as a human reference.

The threshold is a strictness dial, so everything is reported at four settings
(0.2, 0.3, 0.4, 0.5) rather than one arbitrary choice.

**Results (mean over 4 thresholds):**

| method | MAE | ρ |
|---|---|---|
| B4 Agreement-on-line | **0.0108** | +0.998 |
| B1 Independent | 0.0158 | +0.981 |
| PoolEval | 0.0303 | +0.995 |
| PoolEval (learned verif.) | 0.0347 | +0.998 |
| DS one-coin | 0.0362 | +0.987 |
| PoolEval (no anchors) | 0.0381 | +0.996 |
| B2 Majority | 0.0393 | **+1.000** |
| B3 Dawid–Skene | 0.0394 | +0.998 |
| **NF collision (γ=source)** | **0.0480** | +0.995 |
| NF collision (γ=oracle) | 0.0493 | +0.995 |
| NF collision (γ=null) | 0.0538 | +0.995 |
| NF binary EM (γ=1) | 0.0656 | +0.995 |
| **DS full** | **not applicable** — see §6 | |

### What to notice

**This is the one task where the collision correction works as designed.** Read
the last four rows in order:

```
γ = 1            0.0656      "two wrong captions always coincide"    (false)
γ = 1/(K−1)      0.0538      "wrong captions never coincide"          (also false)
γ = measured     0.0480      the truth, measured on labelled source data
```

Monotone improvement, and the measured value beats both extremes. Measured
γ ≈ 0.38–0.48 against an independence value of ≈0.074 — so **roughly 5–6× more
collision than independent errors would produce**. Two captioners really do make
the same mistake often (they were trained on similar data and describe the same
salient object), and measuring that helps.

It also **transfers well**: γ measured on the 300 source images (0.477) closely
matches the truth on the 700 target images (0.380). This is the property that
failed badly in the image-classification port.

Why does it work here when it did nothing for node/image classification? Because
γ only enters the model through the product `(1−β)γ`, where β is the key's
accuracy. Where the key is very good (β ≈ 0.9), `(1−β)` is tiny and γ has no
room to matter. Here the kernel thresholds put β between 0.67 and 0.99, and the
harder thresholds give γ real leverage.

**Honest caveats.** (a) Ranking is nearly saturated — almost every method scores
ρ > 0.98 — because the pool has a wide, clean quality spread (true accuracy
0.35–0.78). Ordering is easy here; only MAE separates the methods. (b) The
winner, B4, uses no consensus at all. (c) The kernel is deliberately simple; a
semantic-similarity model would cluster better, and the numbers would move.

---

## 5. Knowledge-graph completion (K = 14,541 and 40,943)

A knowledge graph stores facts as triples: *(Paris, capital-of, France)*. The
task hides the third slot and asks the model to name it: *(Paris, capital-of, ?)*.
The answer is one of **every entity in the graph** — 14,541 for FB15k-237,
40,943 for WN18RR. Accuracy is "did the top-ranked entity match", i.e. Hits@1.

Pool: TransE, DistMult, ComplEx, RotatE — four genuinely different scoring
functions — × 3 seeds = 12 models. Prior from the labelled validation split.

**FB15k-237 (K = 14,541):**

| method | MAE | ρ |
|---|---|---|
| B1 Independent | **0.0071** | +0.979 |
| B4 Agreement-on-line | 0.0243 | +0.797 |
| PoolEval (learned verif.) | 0.0500 | +0.951 |
| PoolEval | 0.0792 | +0.930 |
| B3 Dawid–Skene | 0.0837 | +0.916 |
| PoolEval (no anchors) | 0.0850 | +0.916 |
| NF collision (γ=null) | 0.0912 | +0.930 |
| NF binary EM (γ=1) | 0.1185 | +0.930 |
| DS one-coin | 0.1331 | +0.979 |
| B2 Majority | 0.1364 | +0.918 |
| NF collision (γ=oracle) | 0.3361 | +0.469 |
| NF collision (γ=source) | 0.3699 | +0.469 |
| **DS full** | **not applicable** — 18.9 GB | |

**WN18RR (K = 40,943):**

| method | MAE | ρ |
|---|---|---|
| B1 Independent | **0.0046** | +0.988 |
| PoolEval (learned verif.) | 0.0671 | +0.988 |
| PoolEval (no anchors) | 0.0672 | +0.988 |
| PoolEval | 0.0790 | +0.771 |
| B4 Agreement-on-line | 0.0811 | +0.403 |
| NF collision (γ=oracle) | 0.1192 | +0.949 |
| DS one-coin | 0.1193 | +0.403 |
| B2 Majority | 0.1210 | +0.403 |
| NF binary EM (γ=1) | 0.1252 | +0.771 |
| NF collision (γ=source) | 0.1301 | +0.771 |
| NF collision (γ=null) | 0.1408 | +0.771 |
| B3 Dawid–Skene | 0.1806 | +0.395 |
| **DS full** | **not applicable** — 149.9 GB | |

### What to notice

**The independence assumption fails by three orders of magnitude.** If two models
picked wrong answers independently from 14,541 entities, they would essentially
never pick the *same* wrong entity — the chance is `1/14540 = 0.000069`. Measured,
they collide **14.2%** of the time. That is **2058×** the assumed rate. On WN18RR
it is **3683×**.

Put plainly: **when these models are wrong, they are wrong in the same specific
way.** They all guess the popular entity, the plausible-looking entity, the one
that fits the relation type. Errors are not scattered — they are concentrated.

**Yet the collision correction makes things worse here** (γ=1 gives 0.1185,
measured γ gives 0.3699). This looks contradictory but is not. Measured γ = 0.142
is *small*, which tells the estimator "collisions are rare, so agreement is
strong evidence of being right" — and accuracy estimates inflate (bias +0.34).
Using γ = 1 is *wrong*, but wrong in the direction that cancels the gauge trap's
inflation. It is right for the wrong reason.

**Honest caveat: this setup has no distribution shift.** The validation and test
splits come from the same graph, so a model's validation accuracy almost exactly
predicts its test accuracy. That is why B1 — which just reports the source
accuracy and ignores the pool entirely — wins by a wide margin. These runs test
**large K**, not robustness to shift. A shifted variant (train on frequent
relations, test on rare ones) would test both.

---

## 6. Can full Dawid–Skene be applied? — the direct answer

**Only to one of the four tasks: link prediction.** And the three failures have
**two different causes**, which are worth keeping separate.

| task | K | full DS? | why |
|---|---|---|---|
| **Link prediction** | 2 | ✅ **yes** | 2×2 matrix = 2 free parameters per model. Trivially affordable, and it wins the ranking (+0.602 when everything else is negative). |
| **Captioning** | unbounded | ❌ **meaningless** | ids are per-item and do not persist |
| **KGC FB15k-237** | 14,541 | ❌ **infeasible** | 18.9 GB, 211,426,140 parameters/model |
| **KGC WN18RR** | 40,943 | ❌ **infeasible** | 149.9 GB, 1,676,288,306 parameters/model |

### Cause A — too big (knowledge-graph completion)

Full DS gives each model a `K × K` table: "when the truth is entity *c*, how
often does this model answer entity *k*?" With K = 40,943 that is 1.68 **billion**
numbers per model, times 12 models — 149.9 GB. The code refuses with that
arithmetic rather than crashing:

```
full Dawid--Skene needs 149.9 GB for 12 models x 40943^2 confusion cells
(budget 2.0 GB). The estimator is not applicable at this label-space size.
```

This is a **machine limit, not a conceptual one**. The model is perfectly well
defined; you just cannot hold it. And even if you could, you could not *fit* it:
3000 test items cannot estimate 1.68 billion parameters.

Note this is the *real-data confirmation* of a prediction that had only been
simulated. [`multiclass_ds.md`](multiclass_ds.md) §7 predicted full DS collapses
once it falls below about one observation per parameter, somewhere between K=50
and K=100. Here K is 14,541 and 40,943.

### Cause B — meaningless (image captioning)

This one is worse, and more interesting.

A confusion matrix entry `π_j[c, k]` only means something if **class *c* is the
same thing on every item**. For digits it is: "class 4" is the digit four on
every image. For captioning it is not. Captions are grouped into clusters *per
image*, and cluster numbers are assigned locally — "cluster 2" on image 5 has
nothing whatever to do with "cluster 2" on image 6. There is no quantity for the
matrix to estimate.

The dangerous part: **the code will happily run anyway and give you a number.**
Per-image clustering produces at most ~14 clusters, so `full_ds` sees a
well-formed K=14 problem and returns MAE 0.0314, which looks respectable.

Here is the proof it is meaningless. Cluster ids are assigned in order of first
appearance, and model captions are listed first — so "cluster 0" really means
"whatever model 0 said". Permuting the order the models are listed in, with
**identical images, captions and references**, changes the answer:

| model order | full DS MAE | one-coin DS MAE |
|---|---|---|
| identity | 0.0435 | 0.0449 |
| reversed | **0.0243** | 0.0492 |
| shuffled | 0.0280 | 0.0490 |
| shuffled (other seed) | 0.0269 | 0.0460 |

Full DS swings by **1.8×** on the same data. One-coin barely moves — because it
only ever tests ids for *equality*, which is permutation-invariant, whereas full
DS *indexes a matrix by id*, so arbitrary ids mean arbitrary matrices.

`ds_estimates` now refuses when a task declares `unbounded=True`, and prints the
reason instead of a number.

### The general rule

> Full Dawid–Skene needs a label space that is **small enough to enumerate** and
> **stable across items**. Lose the first and it is unaffordable; lose the second
> and it is not even wrong — it is undefined.

One-coin DS and PoolEval survive both failures because they only ever ask *"did
these two models give the same answer?"* — a question that needs no class list
at all.

---

## 7. What the four tasks together show

**(a) K decides how much the collision correction matters.**

| task | K | independence rate `1/(K−1)` | measured γ | ratio | effect of correcting |
|---|---|---|---|---|---|
| Link prediction | 2 | 1.000 | 0.999 | **1.0×** | exactly zero — provably |
| Captioning | unbounded | ≈0.074 | 0.38–0.48 | 5–6× | **helps, correctly ordered** |
| KGC FB15k-237 | 14,541 | 0.000069 | 0.142 | 2058× | hurts (γ=1 cancels bias) |
| KGC WN18RR | 40,943 | 0.000024 | 0.090 | 3683× | hurts |

At K=2 the correction is guaranteed to do nothing, and does nothing. As K grows
the independence assumption fails by more and more — up to 3683×. But a *large
violation* is not the same as *large benefit*: what actually decides whether
measuring γ helps is the key's accuracy β, because γ only ever appears as
`(1−β)γ`.

**(b) Models fail together in every single domain.** Every consensus method
over-estimates in every task. The size of the collision-to-independence ratio —
1× to 3683× — is the direct measurement of it.

**(c) Consensus is not always worth having.** In two of four tasks the winner
ignores the pool completely (DoC on link prediction, B1 on both KG datasets).
Consensus earns its keep when models are diverse and the answer space is big
enough that agreement is informative. With K=2 agreement is nearly free, and with
no distribution shift the source prior already knows the answer.

**(d) The estimator's assumptions should be checked, not assumed.** Two results
here came from testing an assumption rather than trusting it: that γ = 1 is exact
at K=2 (it is), and that a confusion matrix means something for captions (it does
not, and the code was silently producing a plausible number until it was tested).
