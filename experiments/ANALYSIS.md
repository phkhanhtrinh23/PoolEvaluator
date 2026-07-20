# Deeper analysis of Active PoolEval-SQL on real model zoos

Five follow-up studies on the real multi-dataset runs (Spider, SQLFlow, BIRD,
BIRD-MiniDev, Spider 2.0-lite; 10-model pools, gpt-5-mini judge, 12-question budget).
The first three reuse the **saved PoolRuns and the real judge's logged decisions** (no
new API calls); the last two are the budget sweep and the synthesis-judge ablation.

```bash
python -m zoo.analyze       --datasets spider sqlflow bird bird_minidev spider2local  # 1,2,3
python -m zoo.budget_sweep  --datasets spider sqlflow bird bird_minidev spider2local --mock  # 4
python -m zoo.run_multi     --datasets spider2local bird --synthesize                  # 5
```

---

## 1. Prior ablation (with vs without the seen prior)

Re-runs pure and active with `Config(use_prior=True/False)`. MAE ×100 (↓).

| Dataset | with-prior pure | with-prior active | no-prior pure | no-prior active |
| --- | --- | --- | --- | --- |
| Spider | 13.38 | **10.96** | 16.67 | 14.07 |
| SQLFlow | 11.28 | **10.23** | 17.93 | 16.60 |
| BIRD | 17.20 | **16.38** | 28.13 | 26.27 |
| BIRD-MiniDev | 13.80 | **12.73** | 27.53 | 26.07 |
| Spider 2.0-lite | 9.24 | **8.54** | 18.75 | 15.42 |

**Finding — the prior is load-bearing on real data.** Removing it inflates MAE by
**+3 to +14 points** (BIRD 17→28, MiniDev 14→28, Spider2 9→19). No-prior *active* (with
12 judged items) never reaches with-prior *pure*. This is the opposite of the
simulator, where at high budget no-prior slightly won: on real cross-domain shift the
source-domain calibration carries most of the absolute-level signal, and the judge
budget refines it. **Keep the prior on.**

---

## 2. Judge reliability vs withheld gold

For each of the 12 judged items we score gpt-5-mini's verdict against the true gold
(used only here, for measurement). Definitions:

- **judge-acc** — fraction of verdicts that are right (a "none" when truly no model is
  correct, or a pick/synth that resolves to the correct class).
- **none-precision** — when the judge says "none", how often that's true (candidate
  coverage really failed).
- **pick-acc** — when the judge *picks* a candidate, how often it's the correct one.
- **gap-rate** — fraction of judged items where no pool model is actually correct.

| Dataset | judge-acc | none-precision | pick-acc | gap-rate |
| --- | --- | --- | --- | --- |
| Spider | 0.42 | – | 0.42 | 0.17 |
| SQLFlow | 0.08 | **1.00** | 0.00 | 0.58 |
| BIRD | 0.33 | **1.00** | 0.20 | 0.67 |
| BIRD-MiniDev | 0.17 | 0.50 | 0.00 | 0.75 |
| Spider 2.0-lite | 0.50 | **1.00** | 0.33 | 0.75 |

**Finding — gpt-5-mini is a reliable "none"-detector but a poor picker.** On the hard
datasets `none-precision ≈ 1.0` (when it declares "no model is correct", it's right) but
`pick-acc ≈ 0.0–0.33` (choosing among plausible-but-wrong candidates, it usually picks
wrong). Because most judged items are true gaps (gap-rate 0.58–0.75), the reliable
"none" verdicts are what help; **the poor picks are what caused the small BIRD Kendall
dip.** Actionable: trust "none", discount picks (soft constraints — future work).

---

## 3. Item-level bootstrap CIs (is the MAE gain real?)

400 bootstraps resampling items with replacement; re-run pure and active EM on each.
ΔMAE = active − pure (negative = active better).

| Dataset | ΔMAE | 95% CI | P(active better) |
| --- | --- | --- | --- |
| Spider | −2.44 | [−4.23, −0.63] | **1.00** |
| SQLFlow | −0.86 | [−1.78, −0.15] | **1.00** |
| BIRD | −0.95 | [−1.99, −0.17] | **0.99** |
| BIRD-MiniDev | −0.91 | [−1.82, −0.33] | **1.00** |
| Spider 2.0-lite | −0.80 | [−1.91, +0.01] | 0.97 |

**Finding — the accuracy improvement is statistically significant on 4/5 datasets**
(CI excludes 0). Spider 2.0-lite is borderline (its CI grazes 0) purely because N=24 —
the point estimate still favors active with P=0.97.

---

## 4. Judge-budget sweep (oracle judge — the selection ceiling)

Active at 0 / 5 / 10 / 20 % budget with a **perfect** judge (isolates *selection +
budget* from *judge reliability*, which §2 already measured). MAE ×100 (↓).

| Dataset | 0% | 5% | 10% | 20% |
| --- | --- | --- | --- | --- |
| Spider | 13.38 | 12.33 | 10.73 | **8.67** |
| SQLFlow | 11.28 | 10.33 | 10.11 | **8.62** |
| BIRD | 17.27 | 16.74 | 15.15 | **13.21** |
| BIRD-MiniDev | 13.80 | 13.34 | 12.43 | **10.44** |
| Spider 2.0-lite* | 9.24 | 9.24 | 9.00 | **8.47** |

*Spider2 budgets are 0/1/2/5 (N=24). Kendall also rises at higher budget (e.g. BIRD
0.60→0.69, MiniDev 0.58→0.68 at 20%).

**Finding — with a good judge, more budget monotonically lowers MAE on every dataset.**
This is the *ceiling*; the real gpt-5-mini judge (§2) captures a fraction of it because
of its picking errors. The gap between this table and the real-judge results is exactly
the room a better judge would recover.

---

## 5. Synthesis judge (let gpt-5-mini WRITE the answer, not just select)

Motivated by §2 (poor picker): instead of choosing among candidates, the judge may
write its own SQL when none is correct; we execute it and map the result to the class
space (`zoo/judge.py`, `synthesize=True`). Compared at the same 12-question budget.

| Dataset | | judge-acc | active MAE | active Ken | synth SQL → correct |
| --- | --- | --- | --- | --- | --- |
| Spider 2.0-lite | selection | 0.50 | **8.54** | 0.85 | – |
| | synthesis | 0.33 | 8.83 | 0.85 | **0 / 3** |
| BIRD | selection | 0.33 | **16.46** | 0.56 | – |
| | synthesis | 0.17 | 16.46 | 0.56 | **0 / 3** |

**Finding — synthesis does not help (and slightly hurts) on hard benchmarks.** Of every
SQL gpt-5-mini wrote, **none recovered a correct answer** (0/3 on both): it cannot solve
questions the whole pool failed. Worse, offering "write your own" made the judge *commit*
more (pick/write) instead of abstaining, which **eroded the reliable "none" detection**
and lowered judge-acc (0.50→0.33, 0.33→0.17). Synthesis would only pay off where the
judge is *more capable than the pool* **and** candidate gaps are frequent — a narrow
regime not seen here. **Robust configuration: conservative "none"-detection + prior +
submodular selection; do not enable synthesis on hard data.**

---

## Overall conclusions

1. **The prior does most of the absolute-accuracy work on real data; the judge budget
   refines it** — and the refinement (ΔMAE) is statistically significant on 4/5 datasets.
2. **The judge's value is its abstention, not its choices.** gpt-5-mini reliably flags
   "no model is correct" (candidate coverage) but is a weak picker/writer on hard SQL.
3. **Selection + budget has real headroom** (oracle sweep is monotone), so the highest-
   leverage next step is a **more reliable verdict**, not more labels — via **soft,
   confidence-weighted constraints** that trust "none" and discount uncertain picks,
   rather than synthesis (which backfired).

### Suggested next steps (grounded in the above)
- **Soft constraints** weighted by verdict type/confidence (trust "none", discount picks)
  — directly targets the BIRD dip and the §4 ceiling gap.
- **A stronger judge** (larger model, self-consistency, retrieval of column docs) — §4
  says the ceiling is worth it; §5 says raw synthesis alone is not the way.
- **Multiple pools/seeds** for cross-pool CIs, and the **Spider 2.0 BigQuery/Snowflake**
  instances (credentials wired) to enlarge the hardest, most-gap-prone regime.
