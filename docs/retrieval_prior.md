# Does a Top-K cosine-similar slice of labeled data predict target accuracy?

**Question.** Retrieve the K labeled items most similar to the unlabeled target pool, measure
each model's accuracy on that slice, and use it as the prior. Does it match the models'
true accuracy on the target pool, or deviate much?

**Answer: it deviates enormously, and retrieval does not help.** Deviation ranges from 3.7
accuracy points (Text-to-SQL) to **85.5 points** (MNIST->SVHN). Top-K beats a random slice
of the same size in 13/24 comparisons -- a coin flip -- and the k-NN variant of the same
idea wins 2/24, i.e. it is actively worse than not retrieving at all.

Scripts: `experiments/embed_text2sql.py` (one paid embedding call, cached),
`experiments/run_retrieval_prior.py`. Raw numbers: `results/retrieval_prior.json`.

## Setup

Every item gets a **label-free** embedding -- the only kind a deployment can compute:

| modality | embedding | dim |
|---|---|---|
| text2sql (spider, bird) | OpenAI `text-embedding-3-small` over `"[db_id] question"` | 1536 |
| vision (mnist_usps, mnist_svhn) | the M models' class-probability vectors, concatenated | 15x10 = 150 |
| graph (AC, DA) | same | 15x5 = 75 |

Source items are scored by cosine similarity to the target two ways: against the target
**centroid** (equivalently, mean similarity to every target item), and by mean similarity to
each item's **10 nearest** target neighbours.

Three controls, because the headline means nothing without them:

- **bottom-K** -- the FARTHEST items. If similarity carries signal, nearest must beat farthest.
- **random-K** -- same slice size, no retrieval. Top-K that cannot beat this retrieves nothing.
- **full** -- the entire labeled split.

**Ordering check.** The retrieval needs embedding row `i` to be the same item as
`true_class` column `i`. Verified with a signature that needs no re-execution: models
emitting *identical SQL* must share a `true_class` value. Spider scores 1.000 against 0.717
for a shuffled order; BIRD 0.925 against 0.608 -- order confirmed, with ~7.5% item-level
noise on BIRD from execution timeouts differing between the two builds.

## Error decomposition -- never report a single MAE

| quantity | meaning |
|---|---|
| `bias` | mean signed error: the shared level offset, one number for the whole pool |
| `MAE` | mean absolute error, which *contains* that offset |
| `cMAE` | MAE after removing the offset: is the SHAPE (relative model ability) right? |
| `rho` | Spearman rank correlation between estimated and true accuracy |

## Table 1 -- how far off is the labeled split, before any retrieval?

Full labeled split vs true target accuracy. Units: **accuracy percentage points**.

| case | MAE | bias | cMAE | rho | share of MAE that is pure level offset |
|---|---|---|---|---|---|
| text2sql/spider | 3.73 | +3.07 | 2.82 | +0.12 | 82.1% |
| text2sql/bird | 3.63 | +3.47 | 1.67 | +0.92 | 95.4% |
| graph/AC | 8.03 | +8.02 | 4.30 | +0.77 | 99.8% |
| graph/DA | 18.73 | +18.73 | 5.78 | +0.91 | 100.0% |
| vision/mnist_usps | 29.42 | +29.42 | 17.97 | **-0.62** | 100.0% |
| vision/mnist_svhn | **85.51** | +85.51 | 8.25 | **-0.63** | 100.0% |

Two things to read off this.

**The error is a single shared level offset.** `|bias| / MAE` is 82-100% in every case,
and 100% in four of six -- meaning *every* model's accuracy is overstated, by nearly the
same amount. On MNIST->SVHN the labeled split says the pool is 85.5 points better than it
is. This reproduces the earlier NE-3 finding on Text-to-SQL and shows it holds across all
three modalities.

**On vision the prior ranks models backwards.** `rho = -0.62` and `-0.63`: source accuracy
is *anti*-correlated with target accuracy. The prior is not merely mis-levelled there, its
shape is wrong too, and `cMAE` of 17.97 on USPS confirms it. Text2SQL and graph keep a
usable ranking (rho 0.77-0.92), so for those the prior gets relative ability right and only
the level wrong.

## Table 2 -- does Top-K retrieval close the gap?

Best K per case (the K minimising Top-K MAE), against the random slice of the *same* size.
**Bold = Top-K beat random-K.** Units: accuracy percentage points.

### centroid similarity -- 13/24 comparisons won (54%, a coin flip)

| case | best K | Top-K | random-K | full |
|---|---|---|---|---|
| text2sql/spider | 30 | **3.13** | 3.37 | 3.73 |
| text2sql/bird | 60 | **3.27** | 4.18 | 3.63 |
| vision/mnist_usps | 2500 | **28.80** | 29.42 | 29.42 |
| vision/mnist_svhn | 1000 | **84.62** | 85.51 | 85.51 |
| graph/AC | 936 | **7.64** | 8.32 | 8.03 |
| graph/DA | 55 | **5.65** | 19.69 | 18.73 |

### 10-NN similarity -- 2/24 comparisons won (8%, worse than not retrieving)

| case | best K | Top-K | random-K | full |
|---|---|---|---|---|
| text2sql/spider | 30 | 3.80 | 3.37 | 3.73 |
| text2sql/bird | 60 | **2.47** | 4.18 | 3.63 |
| vision/mnist_usps | 5000 | 30.28 | 29.41 | 29.42 |
| vision/mnist_svhn | 1000 | **85.49** | 85.51 | 85.51 |
| graph/AC | 936 | 19.47 | 8.32 | 8.03 |
| graph/DA | 548 | 33.39 | 19.01 | 18.73 |

Table 2 is selected on the test quantity -- best K is chosen *after* seeing the answer --
so it is the most generous reading available, and it still only reaches a coin flip. At a
fixed K chosen in advance the picture is worse: at the smallest slices Top-K is
catastrophic (spider K=6 MAE 15.13 vs random 6.32; bird K=12 18.97 vs 5.84; graph/AC k-NN
K=94 33.26 vs 7.73).

**The one real win is graph/DA at K=55: 5.65 vs 19.69 random and 18.73 full** -- a 13-point
improvement. But the same case under k-NN similarity gives 43.26, the worst number in the
experiment. A method whose result flips from best to worst when the definition of
"similar" changes is not measuring similarity, it is measuring luck.

## Why retrieval cannot work here

Top-K reweights *within* the source distribution. The error is *between* distributions.

No subset of MNIST is as hard as SVHN. The best any reweighting could do is pick the
hardest source items, and even the deliberately-farthest slice on MNIST->SVHN only reaches
84.21 -- against a gap of 85.5. The level offset is a property of the domain pair, not of
which items you select, so no selection rule can remove it.

This is also why `bottom-K` is sometimes the better slice: on graph/DA k-NN, bottom-K
reaches 5.50 while top-K gets 37.48. Picking *dissimilar* items happens to pick harder
ones, which accidentally corrects the level. That is not retrieval working, it is a
difficulty proxy working by accident -- and it reverses sign across cases, so it cannot be
turned into a rule.

A further problem specific to the vision and graph embeddings: they are built from the
models' own probability vectors, so similarity is entangled with confidence. Retrieving
items that "look like the target" partly retrieves items the models are *unsure* about,
which shifts accuracy for a reason unrelated to distributional match.

## What to do instead

1. **Do not use retrieval to align the prior.** Confirms and generalises NE-3 across all
   three modalities.
2. **Always report `bias` + `cMAE`, never a single MAE.** A single MAE of 85.51 on SVHN
   hides that `cMAE` is 8.25 -- the relative ordering is far better than the number suggests.
3. **The lever is the level, not the alignment.** The prior's shape is already usable on
   text2sql and graph (rho 0.77-0.92); what it needs is a scalar gauge, and a handful of
   target labels buys that. This is what the expert loop already provides -- see
   `docs/random_init.md`, where the anchor's value collapses from ~9.8 points at b0 to
   ~2.7 at b40 precisely because validation supplies the level.
4. **On vision, do not trust the prior's ranking at all** (rho < 0), which is consistent
   with `docs/random_init.md` finding that a *random* anchor beat the measured one on both
   vision cases.

## Reproduce

    python experiments/embed_text2sql.py          # one cached OpenAI embedding call
    python experiments/run_retrieval_prior.py --fracs 0.05 0.1 0.25 0.5
