# Real Text-to-SQL model zoo → PoolRun

This package replaces the paper's **simulator** with a **real** evaluation pipeline: it
generates SQL from a pool of real LLMs, executes it on the live Spider databases, and
builds the `PoolRun` that the shipped estimator (`pooleval/`) and baselines consume —
so PoolEval-SQL is finally measured on real model outputs, not on data drawn from its
own generative assumptions (the circularity flagged in the review).

```
python -m zoo.run --n-target 150 --n-source 120
```

Outputs: `zoo_artifacts/poolrun_dev.npz` (the real PoolRun) and
`results/zoo_spider_main.json` (ranking of PoolEval-SQL vs baselines vs true EX).

## Pipeline

| stage | module | what it does |
| --- | --- | --- |
| load | `data_spider.py` | Spider dev/train (FusionSQL SFT format); remaps stale DB paths to local SQLite; deterministic subset spread across databases |
| prompt | `prompt.py` | schema-rich / minimal / few-shot prompt styles (style = within-group near-clone knob) |
| generate | `clients.py`, `generate.py` | OpenAI chat with **on-disk cache + retries** (fully resumable); extracts one SQL statement from each response |
| execute | `execute.py` | runs pred + gold on SQLite with a per-query timeout; order-insensitive-unless-`ORDER BY`, type-normalized result comparison (LA1 canonical), arity-aware |
| build | `build.py` | result → equivalence class (`0`=matches gold, `>0`=shared wrong result, `<0`=exec error); execution verifier; source-split seen prior → assembles `PoolRun` |
| evaluate | `run.py` | PoolEval-SQL + B1–B4 on the real `PoolRun`, scored against true dev EX |

## The zoo (manifest in `config.py`)

M=10 members over **7 provenance groups** (= base checkpoint), with three groups
carrying a within-group **near-clone pair** (same base, different prompt/decoding) so
the true independent count ≈ 7 < M — the correlated-pool regime PoolEval-SQL targets.

| group (base) | members | role |
| --- | --- | --- |
| gpt-4o | schema, minimal | near-clone pair |
| gpt-4o-mini | schema, fewshot | near-clone pair |
| gpt-4.1 | schema | singleton |
| gpt-4.1-mini | schema | singleton |
| gpt-4.1-nano | schema | singleton |
| gpt-4-turbo | schema | singleton |
| gpt-3.5-turbo | schema, fewshot | weak-base near-clone pair (a colluding low-accuracy clique) |

## How the real PoolRun maps to the estimator's inputs

| PoolRun field | real source |
| --- | --- |
| `true_class[m,i]` | executed result of model m vs gold on item i (0=correct; shared id for identical wrong results = the real agreement signal) |
| `true_acc[m]` | real execution accuracy (EX) on the dev target — what error is measured against |
| `group[m]` | base checkpoint (ground-truthed by construction) |
| `prior[m]`, `prior_sigma[m]` | model's **real EX on the Spider train split** (a labeled *source* domain) → the "seen calibration" transferred to the dev target; train/dev share no databases, so this is a genuine cross-domain prior under shift |
| `verifier_guess[i]` | execution-plausibility: most-supported non-empty executable result; abstains if all empty/failed |

`Config(real_data=True)` makes the kernel the identity, so both the estimator and the
baselines read these **measured** equivalence classes directly (no synthetic
perturbation).

## Honest deviations from the paper (read before citing numbers)

1. **OpenAI-only pool.** The `TOGETHER_API_KEY` on this machine returns HTTP 403, so the
   open-model groups (Qwen/DeepSeek/Llama/CodeLlama) could not be queried. The zoo is
   built from the OpenAI family instead: distinct base checkpoints are the provenance
   groups, prompt/decoding variants are the near-clones. This is a *narrower* diversity
   than the paper's seven distinct pretrained bases — error correlation across OpenAI
   bases is plausibly higher than across truly independent vendors, which if anything
   makes the setting *harder* for the joint estimator (less independent evidence).
2. **Realistic (not idealized) verifier.** The verifier is execution-grounded and
   therefore *pool-correlated*: it cannot certify a query that runs but returns wrong
   rows. This is the realistic model the review argued for (NE-1), not the simulator's
   independent oracle — expect the verifier to help less than in the paper.
3. **Spider first.** BIRD's questions/gold are unfetched Git-LFS stubs on this machine;
   only its databases are present. The pipeline is benchmark-agnostic (`data_spider.py`
   is the only Spider-specific piece); a BIRD loader returning the same item dicts wires
   it in once the data is available.
4. **One real pool, one seed.** Unlike the simulator's 5–20 pool resamples, this is a
   single real pool. CIs should come from item-level bootstrap (a follow-up), not from
   pool resampling.

Results and interpretation: see `experiments/REAL_ZOO_RESULTS.md`.
