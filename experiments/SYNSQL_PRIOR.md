# NE-3: Retrieval-aligned priors from SynSQL-2.5M

**Status:** measured (real models, real execution). Code: [`synsql/`](../synsql/README.md).
Console: `results/synsql_prior_console.txt`; raw: `results/synsql_prior.json`,
`results/synsql_retrieval_dry.json`.

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

<!--RESULTS-->

## Reproduce

```bash
python -m synsql.ingest                                # 9.3 GB -> index      (~30 s)
python -m synsql.subsets --sample 400000               # 3 partitionings
python -m synsql.run_prior --dry                       # separability, no API spend
python -m synsql.run_prior --partition kmeans --probe --workers 12
python -m synsql.report                                # markdown tables
```
