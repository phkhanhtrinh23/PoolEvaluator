# NE-3: Retrieval-aligned priors from SynSQL-2.5M

**Status:** measured (real models, real execution). Code: [`synsql/`](../synsql/README.md).
Console: `results/synsql_prior_console.txt`. Raw results: `results/synsql_prior.json`,
`results/synsql_retrieval_dry.json`, `results/synsql_level_budget.json`,
`results/synsql_cross_prior.json`.

## The problem this attacks

PoolEval-SQL is advertised as *label-free*. It is — on the target's **outputs**. But the
estimator needs a **seen prior** `p_m` to fix the gauge (cross-model agreement
identifies the *ranking* but not the *level*; see the paper's RQ2 and the review's §A2),
and in every experiment so far that prior was measured on **a labeled split of the
target benchmark itself**:

| target | where its prior came from | target labels used |
| --- | --- | --- |
| spider | `spider train` split | yes |
| bird | `bird train` split | yes |
| bird_minidev, sqlflow, spider2local | held-out slice of the same pool | yes |

A pool operator pointed at a new customer database has none of that. So the anchor —
the single most load-bearing input after the kernel — was quietly importing exactly the
labels the method claims not to need. **NE-3 replaces it with a prior that uses zero
target labels.**

## The method

> Split SynSQL-2.5M into subsets of ~1K → for an unlabeled target, retrieve the top-k
> most aligned subsets → measure the pool on those subsets → use that as the prior.

1. **Ingest.** One streaming pass over the 9.3 GB `data.json` → 2,544,390 records, all
   with an executable SQLite database (16,583 DBs extracted from `databases.zip`).
2. **Partition** into subsets of 1000 (three variants, below).
3. **Probe** a diverse cover of *S* = 32 candidate subsets **once**, offline, with the
   M = 10 model pool: 25 items each → 8,000 generations, executed against the live
   SynSQL DBs for true per-model correctness.
4. **Retrieve.** Represent every item by TF-IDF over *question tokens ++ schema tokens
   (table/column names)* — the only things an operator has for an unlabeled target. A
   set is the L2-normalized mean (a kernel mean embedding), so
   `d(S,T) = 1 − cos(φ(S), φ(T))` and `‖φ(S) − φ(T)‖` is exactly the linear-kernel MMD.
5. **Prior** = pooled per-model EX over the top-k = 5 retrieved subsets (5 × 25 = 125
   labeled items, budget-matched to the `n_source = 120` in-split prior it replaces).

**The economy that makes it deployable:** a subset is probed *once* and reused by every
target. The 8,000 generations are a fixed offline cost; each additional unlabeled target
then gets a prior for **zero** additional inference.

## Why the partitioning decides everything

The literal instruction — *split randomly into ~1K subsets* — is implemented
(`partition_random`) and is a **degenerate control**. Every random subset is an unbiased
i.i.d. sample of the *same* corpus, so all subsets share one distribution and one pool
accuracy up to O(1/√1000) noise. Retrieval then has nothing to select on. This is a
property of the split, not a failure of the retriever.

So three partitionings are built and compared:

| name | construction |
| --- | --- |
| `random` | i.i.d. shuffle → chunks of 1000 (the degenerate control) |
| `db` | pack whole `db_id`s (a SynSQL `db_id` *is* a domain, e.g. `forestry`) |
| `kmeans` | MiniBatchKMeans over question TF-IDF, cut into 1000s (used for the main run) |

Measured separability — std of `d(S,T)` across subsets, per target:

| partition | spider | bird | bird_minidev | sqlflow | spider2local |
|---|---|---|---|---|---|
| `random` | 0.0060 | 0.0047 | 0.0050 | 0.0096 | 0.0082 |
| `db` | 0.0165 | 0.0117 | 0.0130 | 0.0268 | 0.0234 |
| `kmeans` | **0.0638** | **0.0530** | **0.0578** | **0.1089** | **0.0842** |

Random subsets are ~10× less separable than clustered ones. Everything below therefore
uses the **most favourable** partitioning (`kmeans`) — so the results are an upper
bound on what this idea delivers, not a strawman.

---

# Results

Setup: M = 10 real LLMs, 5 unlabeled targets (150 items each), S = 32 candidate
subsets probed at 25 items = **8,000 generations executed on live SynSQL DBs**.
Probed pool EX per subset spans **0.412–0.624** (mean ≈ 0.54), so the corpus really
does contain subsets of differing difficulty.

## R1. Retrieval does not predict prior quality — at all

The core claim to test: *does a subset being closer to the target make it a better
prior?* Correlations across the 32 candidate subsets:

| target | corr(d, prior MAE) | corr(d, centered MAE) | corr(d_struct, prior MAE) | corr(d, subset EX) |
|---|---|---|---|---|
| spider | −0.082 | +0.114 | −0.170 | +0.080 |
| bird | −0.040 | −0.094 | +0.149 | −0.034 |
| bird_minidev | −0.059 | −0.304 | +0.134 | −0.060 |
| sqlflow | +0.052 | −0.207 | +0.176 | +0.036 |
| spider2local | +0.009 | −0.144 | +0.216 | +0.009 |

Every correlation is ≈ 0, and several have the *wrong sign*. The last column is the
diagnosis: **topical distance carries essentially no information about how well the
pool performs on a subset** (|r| ≤ 0.08). Retrieval is selecting on a quantity that is
statistically unrelated to the thing the prior needs to estimate.

Nor is this a defect of TF-IDF. I added a second retriever (`StructuralRetriever`)
using only *difficulty* proxies visible on an unlabeled target — question length,
#tables, #columns, and cue-word rates for aggregation / grouping / superlative /
comparison / temporal / negation / join. It is **worse** (`d_struct` correlations are
positive, i.e. nearer subsets are slightly *harder* to predict), and `synsql-topk-struct`
loses to plain `synsql-topk` on 4 of 5 targets.

The `synsql-far` control settles it: pooling the **farthest** subsets beats the nearest
on prior MAE for 3 of 5 targets (bird 20.5 vs 23.9; bird_minidev 26.7 vs 33.7;
spider 16.8 vs 17.1). **Alignment is not what matters.**

## R2. Prior error is *entirely* level, not shape

Decomposing prior error into a signed level offset (`bias`) and the residual after
removing it (`centered MAE`) — aggregate over the 5 targets:

| prior | prior MAE ↓ | bias | centered MAE ↓ | PoolEval MAE ↓ | Kendall ↑ |
|---|---|---|---|---|---|
| insplit (target-labeled) | **3.62** | +2.50 | 2.80 | 12.98 | **0.68** |
| synsql-topk | 28.60 | +21.78 | 3.64 | 25.23 | 0.62 |
| synsql-topk-struct | 29.66 | +23.18 | 3.89 | 26.03 | 0.55 |
| synsql-soft | 27.59 | +19.83 | 3.57 | 24.47 | 0.60 |
| synsql-random | 27.42 | +20.56 | 3.29 | 24.47 | 0.54 |
| synsql-far | 27.24 | +20.51 | 2.99 | 24.76 | 0.62 |
| synsql-all | 26.75 | +19.82 | **2.98** | 24.22 | 0.49 |
| synsql-oracle (bound) | 14.10 | +12.27 | 3.09 | 19.40 | 0.60 |
| **synsql-topk + true level** | **3.64** | 0.00 | 3.64 | **11.68** | 0.56 |

Read the `bias` and `centered MAE` columns together: for every SynSQL condition,
`prior MAE ≈ |bias|` and the centered MAE is **2.98–3.89 — statistically on par with
the in-split prior's 2.80**, which costs a fully labeled 120-item in-domain split.

> **A label-free corpus prior already recovers the pool's relative ability. The entire
> error is a single shared level offset — i.e. exactly the gauge.**

Per target the offset is −17.1 (spider), +23.9 (bird), +33.7 (bird_minidev), +15.4
(sqlflow), +52.8 (spider2local). Note the **sign flips**: SynSQL is harder than Spider
and easier than BIRD/Spider2.0-local. No retrieval can fix that, because SynSQL's own
difficulty range (EX 0.41–0.62) does not span the targets' (EX 0.05–0.74). Even the
`oracle` row — the best 5 of 32 subsets chosen *using target labels* — still leaves
+12.3 mean bias, and +39.1 on spider2local.

## R3. So buy back only the level — it costs ~10–40 labels

If the level is the whole problem, label a handful of target items and shift the
retrieved prior so its mean matches the pool's observed accuracy on them. Everything
else stays label-free (`synsql/level_budget.py`, 200 resamples per point, **zero extra
generation**):

| target | j=0 | j=2 | j=5 | j=10 | j=20 | j=40 | j=80 | in-split (120 labeled) |
|---|---|---|---|---|---|---|---|---|
| spider | 17.07 | 24.04 | 14.24 | 9.87 | 7.01 | 4.97 | 3.80 | 3.73 |
| bird | 23.93 | 22.52 | 14.25 | 10.59 | 7.37 | 5.31 | 3.69 | 3.47 |
| bird_minidev | 33.73 | 22.22 | 13.18 | 8.87 | 6.52 | 5.37 | 4.37 | 2.48 |
| sqlflow | 15.44 | 24.16 | 14.71 | 10.88 | 7.98 | 5.51 | 4.03 | 3.82 |
| spider2local | 52.84 | 7.29 | 6.04 | 5.59 | 5.43 | 5.45 | 5.45 | 4.58 |

*(prior MAE ↓)*

**~40 target labels buys what a 120-item labeled in-domain split buys** (prior MAE
5.0–5.5 vs 2.5–4.6), and 10 labels already removes 60–90 % of the gap. Downstream,
the j=40 prior gives PoolEval MAE ≤ the in-split prior on **4 of 5** targets (spider
13.05 vs 13.38; bird 15.32 vs 17.20; sqlflow 10.17 vs 11.28; spider2local 5.39 vs
9.24; bird_minidev 14.91 vs 13.80 is the exception).

Caveat worth stating: **j=2 is worse than j=0** on spider and sqlflow (24.0 and 24.2 vs
17.1 and 15.4). A level estimated from 2 items is noisier than the corpus offset it
replaces. A too-small anchor is worse than none.

## R4. One downstream number that looks good is a coincidence

On spider, `synsql-all` gives PoolEval MAE **5.28** against the in-split prior's 13.38 —
apparently a 2.5× improvement from a *worse* prior. It is not a real gain. The earlier
zoo run showed Spider's single-vendor pool over-agrees (86 % agreement vs 74 % correct)
and PoolEval-SQL therefore **over**-estimates accuracy by ~13 points; the SynSQL prior
is biased **downwards** by ~17 on Spider, and the two errors cancel. The same cancellation
runs the wrong way on BIRD/minidev/spider2local, where the upward-biased prior compounds
the inflation (PoolEval MAE 26.4 / 30.4 / 43.0 vs 17.2 / 13.8 / 9.2). Two large errors
cancelling is not a method.

## R5. The level problem is not SynSQL's fault — no out-of-domain source fixes it

The comparison so far is corpus-vs-*the target's own labeled split*, which leaves out the
middle ground: what if the prior comes from a **different real benchmark**? Every
dataset's source-split prior is already measured, so the full source × target matrix
costs **zero generation** (`synsql/cross_prior.py`).

Raw prior MAE, rows = prior source, `*` = in-domain (the diagonal):

| prior from | spider | bird | bird_minidev | sqlflow | spider2local |
|---|---|---|---|---|---|
| spider | **3.73\*** | 40.47 | 50.27 | 34.93 | 72.33 |
| bird | 34.10 | **3.47\*** | 13.10 | 2.60 | 35.17 |
| bird_minidev | 47.52 | 10.12 | **2.48\*** | 15.65 | 21.75 |
| sqlflow | 29.18 | 8.38 | 18.02 | **3.82\*** | 40.08 |
| spider2local | 65.52 | 28.12 | 18.32 | 33.65 | **4.58\*** |
| SynSQL top-k | 17.07 | 23.93 | 33.73 | 15.44 | 52.84 |

Centered MAE (level removed), summarised:

| prior source | spider | bird | bird_minidev | sqlflow | spider2local |
|---|---|---|---|---|---|
| in-domain | 2.82 | 1.83 | 2.48 | 2.28 | 4.58 |
| best cross-benchmark | 1.89 | 2.65 | 2.98 | 1.64 | 3.50 |
| mean cross-benchmark | 3.66 | 4.13 | 4.35 | 3.48 | 4.38 |
| SynSQL top-k | 3.04 | 2.83 | 3.68 | 3.21 | 5.45 |

Two things follow, and the first partially rehabilitates the corpus idea:

1. **The level catastrophe is a property of being out-of-domain, not of SynSQL being
   synthetic.** Spider→BIRD is off by 40.5, Spider→Spider2.0-local by 72.3 — both *worse*
   than SynSQL. SynSQL beats the mean cross-benchmark prior on 2 of 5 targets (spider
   17.1 vs 44.1, sqlflow 15.4 vs 21.7). As a prior corpus it is competitive with, and
   sometimes better than, borrowing another real benchmark.
2. **Relative ability transfers from essentially anywhere. The level transfers from
   nowhere.** Centered MAE sits in 1.8–4.4 for *every* source — in-domain, cross-benchmark,
   or synthetic corpus — while raw MAE spans 2.5–72.3 purely through the offset. This is
   R2 generalised: the gauge is the only thing an in-domain split actually buys.

Two confounds to respect before citing this matrix. **`sqlflow` is BIRD-derived** (it
reuses the BIRD databases), which is why bird→sqlflow scores 2.60 — better than sqlflow's
own in-domain prior. That is leakage, not transfer, and the same caution applies to the
bird↔bird_minidev cells. **`spider2local` has N=24 and true EX 0.05**, so it is noisy as
both source and target, and as a *source* its prior is nearly all zeros, which is why its
row has the worst centered MAE in the table (7.2–8.0).

---

# What this means for the paper

1. **The "label-free" claim needs an asterisk, and now has a price tag.** The seen prior
   was silently importing a labeled split of the target benchmark. It cannot be replaced
   by an auxiliary corpus, however well aligned: R1 shows alignment is uninformative and
   R2 shows the residual is a pure gauge offset. The honest statement is *label-free up
   to a scalar gauge*, and R3 prices that gauge at **~10–40 labels**, down from a full
   labeled split.
1b. **SynSQL is a defensible prior corpus — just not for the reason one would assume.**
   R5 shows it is competitive with (and on 2 of 5 targets better than) borrowing another
   real benchmark, and that the level gap afflicts every out-of-domain source equally.
   What does *not* survive is any claim that domain coverage or retrieval alignment is
   what makes it work: the shape transfers from anywhere, aligned or not.
2. **This is a positive result for the Active PoolEval extension already in this repo.**
   The gauge needs ~10–40 target labels — precisely the budget the judge-in-the-loop
   module (`pooleval/active`, `zoo/judge.py`) operates at. The natural composition is:
   retrieved corpus prior for the *shape* (free, amortized over all targets) + a small
   judge budget for the *level*. That is a cleaner story than either piece alone.
3. **Report the bias/shape decomposition, not just MAE.** It is what separates "the prior
   knows which model is better" (it does, everywhere) from "the prior knows how good they
   are" (it doesn't, without an in-domain anchor). The paper's current single-MAE
   reporting hides this entirely.
4. **Do not cite the Spider `synsql-all` improvement.** See R4.

## Caveats

- One real pool (M=10, OpenAI-only), one seed per target, N=150 per target. Top-1 is
  degenerate at this scale (0.00 everywhere) and is not reported.
- 32 probed subsets of 354; the candidate cover is target-blind, so a bigger budget
  could find better subsets — but the `oracle` row bounds that gain and it is small
  relative to the level offset.
- SynSQL gold is synthetic; ungradable items (gold fails to execute) are dropped.
  Gradability was 25/25 on essentially every probed subset.
- The `random` partitioning is reported for completeness but was not probed: R1 already
  shows retrieval fails on the *most* separable partitioning, so it cannot succeed on
  the least separable one.


## Reproduce

```bash
python -m synsql.ingest                                # 9.3 GB -> index      (~30 s)
python -m synsql.subsets --sample 400000               # 3 partitionings
python -m synsql.run_prior --dry                       # separability, no API spend
python -m synsql.run_prior --partition kmeans --probe --workers 12
python -m synsql.level_budget                          # gauge label budget (no API)
python -m synsql.cross_prior                           # source x target matrix (no API)
python -m synsql.report                                # markdown tables
```
