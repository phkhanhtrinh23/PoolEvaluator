# Active PoolEval-SQL on real model zoos across 5 datasets

Real-data validation of Active PoolEval-SQL. For each dataset we generate a pool of
**10 real LLMs** (OpenAI family, 7 provenance groups — see `zoo/config.py`), execute
their SQL on the live databases to build a real `PoolRun`, then compare **label-free
PoolEval-SQL** against **Active PoolEval-SQL**, which spends a small **gpt-5-mini judge**
budget (12 questions) on the items the submodular selector deems most ambiguous.

The judge is label-free: it reads the question, the DB schema (only the tables the
candidate SQLs reference), and each distinct executed candidate result, then picks the
correct one **or answers "none"** when no candidate is right — the candidate-coverage
fix, on real data. Gold labels are used only to *score* the final ranking, never by the
judge.

---

## Datasets

| Dataset | What it is | DB backend | N (target) |
| --- | --- | --- | --- |
| **Spider** | classic cross-domain Text-to-SQL (dev) | local SQLite | 150 |
| **SQLFlow** | augmented/complex SQL over BIRD databases | local SQLite | 150 |
| **BIRD** | large real databases, harder questions (dev) | local SQLite | 150 |
| **BIRD-MiniDev** | curated hard subset of BIRD | local SQLite | 150 |
| **Spider 2.0-lite (local)** | enterprise analytics, external-knowledge docs, very hard | local SQLite | 24* |

*Spider 2.0-lite has 135 local instances but only ~34 ship a gold SQL usable here, so
N is small (treat its numbers as indicative). Its BigQuery/Snowflake instances (credentials
under `Spider2/.../evaluation_suite`) are a separate cloud runner and are not in this
local table.

---

## Metric definitions

Every number compares the estimator's per-model accuracy vector against the models'
**true execution accuracy (EX)** on that dataset. The judge budget is **12 questions**
in all runs (hybrid-submodular selection).

| Column | Meaning | Better |
| --- | --- | --- |
| **trueEX** | mean true execution accuracy of the pool (dataset difficulty; low = models mostly wrong) | — |
| **pure MAE / active MAE** | mean abs. error of estimated vs true accuracy, ×100 (points) — pure PoolEval vs Active | ↓ |
| **ΔMAE** | active − pure MAE (negative = Active is more accurate) | ↓ (negative) |
| **pure Ken / active Ken** | Kendall-τ rank correlation of estimated vs true ranking | ↑ |
| **pure T1 / active T1** | Top-1 correct: does the estimator rank the truly-best model #1? (0/1) | ↑ |
| **none** | of the 12 judged items, how many gpt-5-mini declared **"none of the pool is correct"** (candidate-coverage events detected on real data) | — |

---

## Results (10-model pool, gpt-5-mini judge, 12-question budget)

Sorted easy → hard by trueEX:

| Dataset | N | trueEX | pure MAE | active MAE | ΔMAE | pure Ken | active Ken | pure T1 | active T1 | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spider | 150 | 0.74 | 13.38 | **10.96** | **−2.42** | 0.71 | **0.75** | 0 | 0 | 0 |
| SQLFlow | 150 | 0.42 | 11.28 | **10.23** | **−1.04** | 0.72 | 0.72 | 0 | 0 | 1 |
| BIRD | 150 | 0.37 | 17.27 | **16.46** | **−0.81** | 0.60 | 0.56 | 0 | 0 | 2 |
| BIRD-MiniDev | 150 | 0.27 | 13.80 | **12.73** | **−1.07** | 0.58 | 0.58 | 0 | 0 | 4 |
| Spider 2.0-lite | 24 | 0.05 | 9.24 | **8.54** | **−0.70** | 0.79 | **0.85** | 0 | **1** | 3 |

---

## Analysis

**1. Active validation lowers ranking error (MAE) on every dataset.** With only 12
judged questions, ΔMAE is negative across all five (−0.7 to −2.4 points). The judge
never hurt the absolute accuracy estimate on any real dataset.

**2. The "none" rate tracks difficulty — candidate coverage is a real phenomenon, not a
simulator artifact.** On Spider (trueEX 0.74) the judge never says "none" (the correct
result is almost always in the pool). As pools get weaker, "none" rises: SQLFlow 1 →
BIRD 2 → Spider 2.0 3 → BIRD-MiniDev 4. These are real cases where *every model is
wrong* and consensus alone could never recover the truth.

**3. The biggest ranking win is on the hardest dataset — exactly where the theory
predicts.** On **Spider 2.0-lite**, where the pool solves only **5%** of questions, the
correct answer is missing constantly. There Active PoolEval **improves Kendall 0.79 →
0.85 and fixes the Top-1 deployment decision (0 → 1)** — the candidate-coverage fix
paying off on the hardest real benchmark. Spider also gains rank quality (τ 0.71 →
0.75) because the judge confirms/repairs the close calls.

**4. Honest negative: BIRD Kendall dips slightly (0.60 → 0.56).** BIRD is large and
noisy; 12 labels out of 150 is thin, and on a low-signal dataset a couple of judged
items can nudge the rank correlation the wrong way even as MAE improves. This is the
expected variance of a *tiny* label budget, and it is why we report all datasets rather
than cherry-picking.

**5. Real datasets are not adversarially constructed, so effects are modest.** Unlike
the simulator's constructed candidate-gap DGP (where Active goes from 10% → 85% correct
deployments — see [`ACTIVE_POOLEVAL.md`](ACTIVE_POOLEVAL.md)), natural workloads have far
fewer near-clone-collusion traps. The real-data takeaway is that the pipeline
**generalizes across five benchmarks and four DB layouts**, the judge's "none" path
**fires on genuinely-hard questions**, and a **12-question budget consistently reduces
ranking error** — most on the hardest data.

---

## Reproduce

All model generations are cached on disk (`zoo_artifacts/gen_cache/`), so re-runs are
cheap and resumable. Real runs call the OpenAI API (pool generation + the 12 gpt-5-mini
judge calls); use `--mock` for an oracle judge at zero judge cost.

```bash
# 1. Simulator experiments (fast, free, reproducible) — the headline candidate-gap result
python experiments/run_active.py --seeds 20

# 2. Real multi-dataset zoo + gpt-5-mini judge (local-SQLite datasets)
python -m zoo.run_multi --datasets spider sqlflow bird bird_minidev spider2local \
       --n-target 150 --n-source 120 --budget 12 --model gpt-5-mini

#    single dataset / oracle judge (no judge API cost) / smaller & faster:
python -m zoo.run_multi --datasets bird --budget 12 --mock
python -m zoo.run_multi --datasets spider2local --n-target 60 --n-source 30 --budget 12

# 3. Single-dataset judge-in-the-loop on the pre-built Spider PoolRun
python -m zoo.run_active --budget 12                 # real gpt-5-mini
python -m zoo.run_active --budget 12 --mock          # oracle, no API cost
```

Outputs: `results/zoo_multi_<dataset>.json` (per dataset), `results/zoo_multi_all.json`
(combined table above).

### Dataset locations (this machine)

| Dataset | Questions / gold | Databases |
| --- | --- | --- |
| Spider | `data_FusionSQL/spider/sft_spider_*_text2sql.json` | `data_FusionSQL/spider/database` |
| BIRD | `data/sft_bird_*_text2sql.json` | `data/sft_data_collections/bird/**/dev_databases` (the `data_FusionSQL/bird` copies are git-LFS stubs) |
| BIRD-MiniDev / SQLFlow | HF arrow caches under `datasets/` | reuse BIRD databases |
| Spider 2.0-lite (local) | `Spider2/spider2-lite/spider2-lite.jsonl` + `evaluation_suite/gold/sql` | `Spider2/spider2-lite/resource/databases/spider2-localdb` |

Registry + loaders: [`zoo/datasets.py`](../zoo/datasets.py). The loader auto-introspects
each SQLite schema, so adding a dataset is one registry entry.

### Not included

- **Spider 2.0 BigQuery/Snowflake** instances need cloud execution (credentials exist
  under `Spider2/.../evaluation_suite`). The Python clients are installed; a cloud runner
  is future work (per-query cost + latency), separate from this free local table.
- **LiveSQLBench** uses test-case-based evaluation with **no single gold SQL string**
  (`sol_sql` is empty), so it needs a different scoring harness and is excluded.
