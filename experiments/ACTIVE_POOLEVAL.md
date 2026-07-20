# Active PoolEval-SQL — label-efficient, judge-in-the-loop evaluation (NEW)

Not part of the original paper. This extension **formalizes and experimentally
measures the candidate-coverage limitation** of label-free consensus, and shows that a
small, *targeted* expert budget resolves the cases unlabeled agreement fundamentally
cannot. It keeps PoolEval-SQL exactly as shipped and adds a thin active layer.

```bash
python experiments/run_active.py --seeds 20          # simulator (fast, reproducible)
python -m zoo.run_active --budget 12                 # REAL Spider zoo + gpt-5-mini
python -m zoo.run_active --budget 12 --mock          # same, oracle judge, no API cost
```

Code: [`pooleval/active.py`](../pooleval/active.py) (algorithm),
[`pooleval/latent.py`](../pooleval/latent.py) (`run_em(..., constraints=)`),
[`experiments/run_active.py`](run_active.py) (simulator experiments),
[`zoo/judge.py`](../zoo/judge.py) + [`zoo/run_active.py`](../zoo/run_active.py) (real
gpt-5-mini judge). Tests: [`tests/test_active.py`](../tests/test_active.py).

---

## 1. The limitation, made precise

PoolEval-SQL infers each item's latent correct answer as the anchored, group-discounted
**consensus over the classes the pool actually produced**. In [`latent.py`](../pooleval/latent.py)
the candidate set for item *i* is literally `np.unique(obs[:, i])`, and accuracy is
`mean(obs == argmax posterior)`. Two failure modes follow, neither fixable by any
amount of unlabeled agreement:

- **Candidate coverage.** If **no model is correct**, the true class is *absent* from
  the candidate set. Consensus must pick some produced (wrong) class and credits the
  models that emitted it.
- **Near-clone collusion under a fooled verifier.** When a provenance clique (same base
  checkpoint, different prompt/decoding) unanimously emits the **same wrong result**,
  and that result executes plausibly (so the verifier is fooled too), consensus looks
  *confident* while being wrong. The clique's estimated accuracy inflates and the
  deployment ranking can invert.

This is deeper than the collusion the paper already discusses: the correct answer is
not merely down-weighted, it is **not in the hypothesis space at all**. The tell-tale
signals a practitioner might reach for — high disagreement, high posterior entropy —
are exactly *absent* on these items, because a colluded consensus is peaked and
confident. That is why they are dangerous, and why naive active learning misses them.

### The `corr_risk` signal

The one label-free signal that *does* fire is **provenance-structural**: the winning
class is a plurality carried by too few independent groups. We score each item

```
corr_risk_i = 1 − (#distinct provenance groups backing the winning class) / (plurality size)
```

which is ~0 on benign items (a correct plurality drawn from many groups) and large
exactly on near-clone-collusion items — *regardless* of how confident the posterior
looks. This is the acquisition signal the paper's provenance model uniquely provides.

---

## 2. Active PoolEval-SQL

A strong **label-free judge** (gpt-5-mini: reads the question, executes each candidate
SQL, aligns result-to-question; it is *not* given a gold label) is expensive, so we
spend it only where it matters:

1. **Run PoolEval-SQL** on all unlabeled items → latent posteriors, accuracies, ranking.
2. **Select the most valuable items by greedy submodular maximization.** The objective is
   ```
   f(A) = Σ_{i∈A} value_i  +  λ · Σ_i value_i · max_{j∈A} sim(i,j)
   ```
   a **modular** value term (validate high-`corr_risk` items) plus a **facility-location**
   coverage term (avoid spending two labels on redundant near-duplicate items). `f` is
   monotone submodular, so lazy greedy (Minoux/CELF) attains **≥ (1−1/e)·OPT**
   (Nemhauser–Wolsey–Fisher 1978; the bound is tight). For *independent* candidate-gap
   traps the modular term dominates and `f` reduces to top-value (greedy is then exactly
   optimal); the facility term only earns its keep when items genuinely generalize.
3. **Judge the selected items.** The verdict is the correct result's equivalence class —
   possibly a **fresh class no model produced** ("none of these is correct").
4. **Pin each as a hard EM constraint** `P(z_i = judge answer) = 1` and **re-solve with
   incremental (warm-started) EM** — not a restart.
5. **Stop** when the decision is stable (top-1 win-prob > 0.95 and a stable top-3) or the
   budget is spent.

**Abstention.** When the winning consensus is backed by fewer than two provenance groups
(near-clone bloc) or `M_eff` is very small, the estimator reports *“insufficient
independent evidence — expert validation required”* rather than a confident wrong pick.

---

## 3. Experiments (simulator, 20 seeds)

A controlled **candidate-gap DGP** ([`dgp_candidate_gap`](run_active.py)): a 3-member
near-clone clique (true acc ≈ 0.36) is made to win under label-free consensus by traps
on ρ≈24 % of items — the whole clique emits one shared wrong result, **no model is
correct**, and the verifier is fooled. One independent singleton (true acc 0.70) *should*
rank #1. `Top1` = deployment picks the genuinely-best model (↑); N = 400, budgets are
% of N.

**Pure PoolEval-SQL fails as predicted:** it inflates the clique from true **0.357 →
0.582**, ranking a clique member #1 (Top1 ≈ 0.10 across seeds), MAE 6.03, Flip 0.44.

### Top-1 recovery vs judge budget (fraction of correct deployment decisions, ↑)

| strategy | 0 % | 1 % | 5 % | 10 % |
| --- | --- | --- | --- | --- |
| random | 0.10 | 0.10 | 0.20 | 0.30 |
| uncertainty (entropy) | 0.10 | 0.10 | 0.10 | **0.15** |
| disagreement | 0.10 | 0.15 | 0.40 | 0.80 |
| ranking-impact | 0.10 | 0.10 | 0.20 | 0.55 |
| **hybrid submodular (ours)** | 0.10 | **0.20** | **0.50** | **0.85** |

### MAE (accuracy error ×100, ↓) and Kendall-τ (↑) at 10 % budget

| strategy | MAE ↓ | Flip ↓ | Kendall ↑ |
| --- | --- | --- | --- |
| random | 5.50 | 0.43 | 0.13 |
| uncertainty | 5.84 | 0.43 | 0.13 |
| disagreement | 4.01 | 0.32 | 0.34 |
| ranking-impact | 4.96 | 0.38 | 0.22 |
| **hybrid submodular (ours)** | **3.80** | **0.30** | **0.39** |

**Findings.**

- **Uncertainty sampling is the *worst* strategy** (Top1 0.10→0.15) — a direct,
  measured confirmation that the dangerous items look *confident*, so entropy-driven
  acquisition avoids exactly what needs validating.
- **Provenance-aware submodular acquisition is best at every budget**, reaching the
  correct deployment decision **85 % of the time at a 10 % label budget** (vs 30 % random,
  15 % uncertainty), and cutting ranking error roughly in half (MAE 6.0→3.8).
- **The advantage is the acquisition, not the labels.** Random labeling barely moves the
  ranking because it rarely lands on a trap; `corr_risk`-led selection lands on traps
  ~100 % of the time (see [`tests/test_active.py`](../tests/test_active.py)).

### Abstention (experiment C)

The per-item near-clone-collusion flag identifies the trap items with **precision 1.00,
recall 1.00**. Note that the pool's own `M_eff` stays ≈ M on this DGP: because the
collusion makes the clique look *correct* (high posterior), the shared-*wrong*-answer
statistic behind `M_eff` cannot see it. The **structural per-item flag**, which reads
the observation directly, catches what the posterior-based `M_eff` misses — motivating
reporting both.

---

## 4. Real Spider zoo with a gpt-5-mini judge

[`zoo/run_active.py`](../zoo/run_active.py) runs the same loop on the **real** PoolRun
(`zoo_artifacts/poolrun_dev.npz`, M=10 OpenAI models, N=150 Spider-dev items). The judge
is the real **gpt-5-mini** ([`zoo/judge.py`](../zoo/judge.py)): for each selected item it
sees the question, schema, and every distinct executed candidate result, and returns the
correct one *or* "none". On a real, non-adversarial workload the effect is naturally
modest — but the mechanism is confirmed on live data:

- With **8 gpt-5-mini calls** (the 8 most `corr_risk`-ambiguous items), ranking error
  drops **MAE 13.38 → 11.21**; with 12 (mock oracle) **→ 10.73**.
- On **3 of 8 items the judge answered "none"** — correctly detecting that *no* pool
  candidate is right and injecting a fresh correct class no model produced. This is the
  candidate-coverage fix firing on real Spider data, not in simulation.

Cost is bounded by `--budget` (all calls cached/resumable). Use `--mock` to exercise the
full pipeline with an oracle judge at zero API cost.

---

## 5. Honest limitations

- **Universal (all-groups) delusion is still unsolvable, and we don't claim otherwise.**
  If *every* provenance group agrees on the same wrong answer, `corr_risk` is low and no
  label-free signal points to the item; only random/exhaustive validation or a prior
  would help. The abstention flag is the honest response there.
- **The simulator DGP is controlled** (fixed accuracies, verifier deterministically
  fooled on traps) so the flip margin is a clean function of budget. It is a *stress
  test* of candidate coverage, not a claim about natural trap frequency.
- **The real-zoo judge is a selection judge** (choose a produced result or "none"); it
  does not synthesize a brand-new SQL. "None" already supplies the correct *class* for
  ranking/accuracy; a synthesis judge (write + execute new SQL) is a drop-in upgrade of
  `RealJudge.query`.
- **The judge is assumed reliable** (oracle-accurate in simulation). A fallible judge
  degrades gracefully — a wrong pin is one mislabeled item — but this is not yet swept.
