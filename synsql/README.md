# Retrieval-aligned priors from SynSQL-2.5M

PoolEval-SQL is label-free *on the target* — except for one thing. The **seen prior**
is the anchor that fixes the gauge (agreement alone identifies the ranking but not the
absolute level), and in every run so far that prior came from **a labeled split of the
target benchmark itself** (`spider train`, `bird train`, …). That is the one input a
deployed pool operator facing a new database does not have. The prior was doing real
work while quietly importing target labels.

This package removes that dependency. The prior is estimated on **SynSQL-2.5M subsets
retrieved to match the unlabeled target**, using only the target's questions and
schemas — no target gold SQL, ever.

```
python -m synsql.ingest                                   # 9.3 GB -> compact index
python -m synsql.subsets --sample 400000                  # 3 partitionings into ~1K subsets
python -m synsql.run_prior --dry                          # retrieval analysis, no API spend
python -m synsql.run_prior --partition kmeans --probe     # probe + full comparison
```

## Method

1. **Partition** SynSQL-2.5M into subsets of ~1000 items (`subsets.py`).
2. **Probe** a diverse cover of *S* candidate subsets once with the model pool
   (`prior.py`): generate SQL, execute against the SynSQL SQLite DB, record per-model
   correctness. This is the only cost, and it is **paid once for all targets**.
3. **Retrieve**, for an unlabeled target *T*, the top-*k* subsets with smallest
   distance to *T* (`retrieve.py`).
4. **Prior** = pooled per-model execution accuracy over those *k* probes, with a
   binomial σ. Plugs straight into `PoolRun.prior` / `PoolRun.prior_sigma`.

### Representation and distance

Only what an operator actually has:

```
phi(item) = TF-IDF( question tokens  ++  schema tokens: table & column names )
phi(set)  = L2-normalized mean of phi over the set        (kernel mean embedding)
d(S,T)    = 1 - cos( phi(S), phi(T) )
```

Because `phi(set)` is a kernel mean embedding, `||phi(S) - phi(T)||` **is** the
linear-kernel MMD — so the retrieval score is a proper two-sample distance, not an
ad-hoc heuristic. Both are computed; cosine is primary.

### Three partitionings (and why it matters)

| name | how | role |
| --- | --- | --- |
| `random` | i.i.d. shuffle → chunks of 1000 | **degenerate control**: every subset is an unbiased sample of the *same* corpus, so all subsets have the same distribution and the same pool accuracy up to O(1/√1000) noise. Retrieval has nothing to select on. |
| `db` | pack whole `db_id`s (a `db_id` is a domain, e.g. `forestry`) | domain-coherent, genuinely different subsets |
| `kmeans` | MiniBatchKMeans on question TF-IDF, cut into 1000s | maximally separated; the partitioning used for the main run |

The literal reading of "split randomly into 1K subsets" is the `random` row, and it
**cannot work** — this is a property of the split, not of the retriever. The measured
distance spread across subsets (std of `d(S,T)`) makes this concrete; see
`experiments/SYNSQL_PRIOR.md`.

## Honest protocol notes

- **Candidates are chosen without looking at the targets** (`pick_candidates`:
  k-means over subset centroids → medoids). Probing a cover selected *after* seeing
  the target would leak target information into the budget allocation.
- **Budget-matched comparison.** `TOP_K × N_PROBE = 5 × 25 = 125` labeled items, the
  same order as the `n_source = 120` in-split prior it is compared against.
- **Conditions** all draw from the *same* probed candidates, so they differ only in
  *which* subsets are pooled: `synsql-topk` (nearest), `synsql-random`, `synsql-far`
  (farthest — the control that shows distance is what matters), `synsql-all`
  (everything, i.e. more labels but no alignment), and `insplit` (the old
  target-labeled prior, an upper reference rather than a competitor).
- **SynSQL gold is synthetic.** Items whose gold SQL fails to execute are marked
  ungradable (`-1`) and dropped from the prior.

## Files

| file | role |
| --- | --- |
| `config.py` | paths + experiment knobs |
| `ingest.py` | one streaming pass over the 9.3 GB `data.json` → compact index (line-prefix field extraction, skips the huge `cot` field) |
| `subsets.py` | the three partitionings; reservoir-samples the index so partitioning fits in memory |
| `retrieve.py` | TF-IDF representation, subset kernel mean embeddings, cosine/MMD distances, top-k |
| `prior.py` | probe subsets with the pool (cached, resumable, threaded), pool probes into a prior |
| `run_prior.py` | orchestration: spread analysis → candidate cover → probing → per-target comparison |

Results and analysis: [`../experiments/SYNSQL_PRIOR.md`](../experiments/SYNSQL_PRIOR.md).
