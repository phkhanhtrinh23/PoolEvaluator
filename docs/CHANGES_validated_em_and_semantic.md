# Change log: validated EM, execution-suite equivalence, answer matching

Everything added or changed relative to the repository at commit `7c35250`, covering
both pieces of work:

* **Part A — validated EM.** `e` and `gamma` measured on a pre-built labeled split
  instead of fitted; the i-EM judge loop with information-gain selection.
* **Part B — execution-suite equivalence.** For Text-to-SQL, deciding when two answers
  are the same by running them on a suite of database instances instead of one, so `e`,
  `gamma` and correctness stop absorbing coincidental collisions.

---

## 0. The headline: nothing existing was modified

```
$ git diff --stat 7c35250
 pooleval/__init__.py | 15 +++++++++++++--
 1 file changed, 13 insertions(+), 2 deletions(-)
```

That diff is **one import block and two `__all__` entries**. No estimator, no EM, no
kernel, no experiment script from the old version was touched. Every other change is a
new file.

This is not an accident of tidiness — it is the central structural claim of both parts:

* Part A changes **what two nuisance parameters are**, not how anything is maximised.
  The $\alpha$ M-step is character-for-character the one in `Trinh_proof.tex`.
* Part B changes **which answers count as equal**, and every estimator in the repo
  consumes that fact through exactly one operation (`obs[m,i] == obs[m',i]`).

So both plug in underneath the existing machinery rather than beside it.

### New files

| file | lines | part | what it is |
|---|---|---|---|
| `pooleval/validated_em.py` | 815 | A | the whole formulation: measured statistics, EM, information gain, experts, loop |
| `tests/test_validated_em.py` | 394 | A | 28 tests, each re-deriving a claim numerically |
| `experiments/run_validated_em.py` | 450 | A | the three-domain experiment (Text2SQL, vision, graph) |
| `experiments/report_validated_em.py` | 92 | A | turns the result JSON into markdown tables |
| `docs/validated_em.md` | 289 | A | the derivation, the pipeline, the diagnostics |
| `pooleval/semantic.py` | 256 | B | domain-agnostic meaning-clustering and semantic entropy |
| `zoo/semantic_sql.py` | 354 | B | Text2SQL: database test suite + SQL AST tie-breaker |
| `tests/test_semantic.py` | 185 | B | 16 tests, pinned on the two directions of error |
| `experiments/run_semantic_kernel.py` | 269 | B | exact vs semantic kernel, through the whole stack |

Test suite: **97 passing** (was 81).

### Modified file, in full

`pooleval/__init__.py` — re-exports the new symbols so `from pooleval import ...` works:

```python
from .validated_em import (LabeledStatistics, correctness_em, JudgeExpert,
                          NoisyExpert, OracleExpert,
                          excess_collision, gamma_counts, gamma_from_counts,
                          information_gain, latent_posterior, run_validation,
                          set_entropy, validated_em, vote_discount,
                          wrong_collision_matrix)
from . import metrics, kernel, theory, validated_em
```

`pooleval/semantic.py` and `zoo/semantic_sql.py` are imported by path, not re-exported,
because the kernel is chosen per experiment rather than globally.

---

# Part A — validated EM

## A1. What changed conceptually

| | old (`Trinh_proof.tex`, `pooleval/new_formulation.py`) | new (`pooleval/validated_em.py`) |
|---|---|---|
| correlated error | provenance **groups** declared by hand; loading $u_g$ fitted from the target | matrix $e \in [0,1]^{M\times M}$ **measured** on labeled data |
| $\gamma$ | $\gamma_{g(j)}$ = collision rate, a **free parameter** | $\gamma_j = P(C=0\mid Z=0)$, **measured and frozen** |
| $\alpha$ M-step | closed form | closed form — **identical formula** |
| $\beta$ M-step | **not** closed form; `scipy.optimize.minimize_scalar` | **closed form restored** |
| labels | none | expert validates one item at a time, chosen by information gain |
| statistics over time | static | refreshed after each judge call, `new = (old + temp)/2` |

## A2. Measuring the statistics

| symbol | function | replaces |
|---|---|---|
| $e_{jk}$ | `wrong_collision_counts`, `wrong_collision_matrix` | `run.group` + `_estimate_loadings` in `pooleval/latent.py` |
| $\tilde e$ | `excess_collision(e, group)` | the $u_g$ loading |
| $\gamma_j$ | `gamma_counts`, `gamma_from_counts` | `gamma_group` in `collision_agreement_em` |
| $P(C{=}0\mid Z{=}0)$ | `gamma_to_conditional` | $d_j(\beta) = 1-(1-\beta)\gamma^{\text{coll}}_j$ |
| $\beta$ | `pseudo_label_quality` | the fitted `beta` |
| all of the above, live | `LabeledStatistics` | *(new — nothing had state before)* |

**`e[j,k]`** = fraction of labeled items where models $j$ and $k$ return the *same*
answer and it is *wrong*. Execution errors carry distinct negative ids, so two crashes
never count as a collision.

**`excess_collision`** subtracts the cross-group off-diagonal mean and zeroes the
diagonal, so only correlation *above chance* is penalised. This is the same baseline
`pooleval/latent.py::_estimate_loadings` already subtracts.

**Three readings of $\gamma$**, all implemented because the spec admits all three:

| mode | $Z=0$ conditions on | note |
|---|---|---|
| `both_wrong` *(default)* | model wrong **and** pseudo-label wrong | the event actually counted; equals $1-\gamma^{\text{coll}}$; needs the $\beta$ conversion |
| `model_wrong` | model wrong | matches $Z^j_i$ as defined in the `.tex`; plugs in with no conversion |
| `pseudo_wrong` | pseudo-label wrong | the literal reading of the spec sentence |

`gamma_to_conditional` closes the loop between them:

$$P(C=0\mid Z=0)=\beta+(1-\beta)\gamma^{\text{both}}_j = 1-(1-\beta)\gamma^{\text{coll}}_j = d_j(\beta).$$

**`LabeledStatistics`** holds them as live state. `add_validated(answers, truth,
consensus_label)` folds in one expert-revealed label and applies

$$\text{new} = \tfrac12(\text{old} + \text{temp})$$

where `temp` is the statistic recomputed on the validated target items. Entries whose
`temp` is undefined (a model with no eligible validated item yet) keep their old value,
so the blend is a no-op there rather than a pull toward the $1/2$ fallback.
`update_rule="counts"` pools the validated items into the source counts instead,
weighting every labeled item equally — implemented so the two can be compared.

Validated items enter $\gamma$ through the pseudo-label the consensus **would have**
produced without the pin. Using the pinned label would make every wrong model disagree
by construction and drive $\gamma\to1$.

## A3. Where `e` enters — a strict generalisation of the old discount

`vote_discount(column, e_excess)` gives a model's vote at one item the weight

$$\text{disc}_j=\Big(1+\sum_{k\neq j,\ r^k_i=r^j_i}\tilde e_{jk}\Big)^{-1}.$$

Set $\tilde e_{jk}=u_g$ within group and $0$ across and the sum collapses to
$u_g(n_{g,\ell}-1)$ — **exactly** `pooleval/latent.py`'s $1/(1+u_g(n_{g,\ell}-1))$.
Verified over 50 random pools in
`test_vote_discount_reproduces_the_group_loading_formula_exactly`. Provenance grouping
is the special case where correlation is *declared* rather than *measured*.

## A4. The EM

`validated_em(obs, stats, prior, prior_strength, ...)` — one sweep is:

1. latent E-step → $U(o,\ell)$ from correlation-discounted votes, pinned one-hot on validated items;
2. $C_{ij} = \mathbf 1\{r^j_i = \arg\max_\ell U(i,\ell)\}$;
3. correctness E-step → $\tau = P(Z{=}1\mid C,\alpha,\beta,\gamma)$, with $\tau$ **observed** (0/1) on validated items, since the truth is known there;
4. M-step, both maximisers closed form.

```
alpha_j = (sum_i tau_ij + s_j pi_j) / (N + s_j)          # identical to the .tex
beta    = (sum_ij tau_ij C_ij + s_b pi_b) / (sum_ij tau_ij + s_b)   # newly closed form
```

`correctness_em(C, gamma, prior, prior_strength, ...)` is the same fit with the
pseudo-labels frozen — the drop-in replacement for `collision_agreement_em`, and the
one whose log-posterior trace is guaranteed monotone.

`LatentPlan` caches everything about the E-step that does not move between sweeps.
The per-item candidate lists are ragged, so they are stored flat: one slot per (item,
candidate class), `offsets` marking item boundaries, `slot[m,i]` saying where model
$m$'s vote lands. A whole E-step is one `np.add.at` plus segmented reductions — no
Python loop over items. This matters because information gain re-solves the model once
per (candidate item, candidate answer) pair; the vectorised version took a vision pool
from unusable to **5 judge calls in 0.6 s**.

## A5. The judge loop

```
solve EM to convergence
repeat until budget spent:
    IG(o) = H(P) - sum_l U(o,l) H(P_l)          # Hung et al. Eq. (9)
    o*    = argmax IG                            # Eq. (10)
    a     = expert chooses among o*'s candidate answers
    refresh e and gamma with the revealed label  # before the warm start
    if a == pseudo_label[o*]:  keep going, no re-solve
    else:                      pin U(o*,.) one-hot, warm-start EM to convergence
```

| function / class | role |
|---|---|
| `item_entropy`, `set_entropy` | $H(o)$ and $H(P)=\sum_o H(o)$ (Eqs. 7) |
| `information_gain` | Eq. (9), warm-started, `plan` and `report_posterior=False` reused across all hypotheticals |
| `run_validation` | the loop; `select` ∈ `info_gain` / `entropy` / `random` |
| `OracleExpert` | picks the correct answer when the pool produced it; `allow_none=True` lets it reject every candidate |
| `NoisyExpert` | same, right only `accuracy` of the time |
| `JudgeExpert` | adapter for a **real LLM** — wraps `zoo.judge.RealJudge` |

`clamp_on_confirm` (default `False`) follows the spec literally: a confirming answer
changes nothing and triggers no re-solve. Setting it `True` gives Hung et al.'s Eq. (4)
behaviour, pinning every validated item. `update_stats_on_confirm` (default `True`)
refreshes `e`/`gamma` on every judge call, confirmation included.

## A6. What Part A does **not** change

* `pooleval/latent.py`, `inference.py`, `new_formulation.py`, `active.py`, `kernel.py`, `config.py` — untouched.
* The $\alpha$ M-step, the Beta anchor $\text{Beta}(1+s\pi, 1+s(1-\pi))$, and the ESS convention $s+2$ — unchanged.
* `PoolEval`, `NewFormulationPoolEval`, `CollisionAwareNewFormulationPoolEval` — still run, and are the baselines in every table.

---

# Part B — semantic answer equivalence

## B1. The problem

Every estimator consumes one fact: *are these two answers the same?* On an open output
space, exact match answers that question wrongly, and wrongly in a **directional** way:

| quantity | exact-match bias | consequence |
|---|---|---|
| $e$ | **under**counts | near-clones failing the same way in different words look independent → the vote discount never fires → the correlated-error trap reopens |
| $\gamma$ | **over**counts (→1) | EM concludes wrong models reliably disagree with wrong pseudo-labels → it over-trusts agreement |
| $Z$ | mis-graded | "correct" is also an equality test against gold |

So the fix belongs in the **observation kernel**, not the model.

## B2. Relation to the Nature paper

Farquhar, Kossen, Kuhn & Gal, *Nature* **630**:625–630 (2024) has two separable halves:

* **(a) clustering generations by meaning** via bidirectional entailment — *this is what transfers*, with M models' answers in place of M samples from one model;
* **(b) entropy over those clusters** as an uncertainty score — *this arrives for free*: once classes are meaning-clusters, the $H(o)$ already used to pick what the expert validates **is** a discrete semantic entropy, weighted by reliability-discounted vote mass instead of sample frequency.

Combined with the measured result that $IG(o)\approx H(o)$ (Spearman 1.000 / 0.995),
the acquisition rule becomes *"validate the highest-semantic-entropy item"*.

## B3. `pooleval/semantic.py` — domain-agnostic

| function | purpose |
|---|---|
| `greedy_meaning_clusters(n, equivalent, order)` | first-match against cluster representatives — the paper's clustering; **cannot chain** |
| `clusters_from_similarity(S, threshold, order)` | the same with `equivalent(i,j) = S[i,j] >= threshold` |
| `connected_components(adjacency)` | transitive closure — **for comparison only**, it over-merges |
| `refine(coarse, fine)` | product of two partitions: can split, can never merge |
| `gated_clusters(separator, similarity, threshold, decided)` | hard evidence decides; the soft signal only reaches what it cannot |
| `cluster_probabilities(labels, weights)` | $p(c)$, uniform (the paper's *discrete* variant) or reliability-weighted |
| `semantic_entropy`, `discrete_semantic_entropy` | $-\sum_c p(c)\log p(c)$ |
| `to_repo_classes(labels, gold_cluster)` | into the repo convention, `0 == correct` |
| `canonical_relabel` | stable renumbering for comparison |
| `partition_quality(predicted, gold)` | pairwise precision / recall of an induced equivalence |

**Why greedy and not connected components.** Bidirectional entailment is not transitive,
so the relation is not an equivalence relation and a closure has to be chosen.
Components chain — $a\equiv b$, $b\equiv c$ merges $a$ with $c$ — and in a framework
that reads agreement as evidence, merging is the dangerous direction. First-match
cannot chain. The sweep `order` is pinned so the kernel is reproducible.

**Why `partition_quality` exists.** Coarsening equality is a risk, not automatically a
win: a kernel that over-merges *manufactures consensus*. Low **precision** means
answers that differ were merged — the failure that corrupts everything downstream. Low
recall merely leaves signal on the table. Measure both before adopting a kernel.

## B4. `zoo/semantic_sql.py` — Text-to-SQL

### The test suite (`build_variants`, `informative_instances`, `execution_signature`)

Two genuinely equivalent queries agree on **every** database instance. So evaluating
both on extra, row-sampled instances can only **split** a class the single instance
merged; it can never split a truly equivalent pair. **Sound by construction.** This is
the pairwise-equivalence half of test-suite accuracy (Zhong, Yu & Klein, EMNLP 2020) —
and it is what `kernel.py`'s LA2, *"multi-instance (re-run on t constraint-preserving
sub-instances)"*, has only ever **simulated** via a precision parameter. In real-data
mode the kernel was the identity; this makes LA2 real for the first time.

* Sampling is a deterministic salted hash of the row id, never `RANDOM()` — otherwise the equivalence relation itself would be non-deterministic.
* Foreign keys are deliberately not repaired: a dangling reference makes a join drop rows, which is precisely the perturbation that separates two queries that coincided on the full instance. Realism is not required of a test suite; discrimination is.
* `informative_instances` keeps only instances where the **gold** query still returns rows. Without it, pruning eventually makes every query return empty, all of them collide, and the variant manufactures the agreement it was added to disprove.
* Size caps (`--max-db-mb`, `--budget-mb`) skip oversized databases, which then fall back to single-instance evidence — still sound, just less discriminating. Coverage is reported.

### The AST tie-breaker (`ast_features`, `feature_cosine`, `ast_similarity_matrix`)

`sqlglot` parse → `simplify` → bag of node types, table/column/literal multisets and
clause flags. Unparseable SQL falls back to character trigrams so a garbled generation
lands somewhere instead of crashing the kernel.

**The AST is a tie-breaker, never a decider.** Measured:

| pair | cosine | truth |
|---|---|---|
| `WHERE x > 5` vs `WHERE 5 < x` | 1.0000 | same meaning ✓ |
| `WHERE x=1 AND y=2` vs `WHERE y=2 AND x=1` | 1.0000 | same meaning ✓ |
| `LIMIT 1` vs `LIMIT 2` | **0.9333** | **different answers** ✗ |

Any threshold loose enough to catch the paraphrases also merges `LIMIT 1` with
`LIMIT 2`. Execution across a suite is *sound*; AST similarity is neither sound nor
complete. **They are therefore gated, not concatenated** — equivalently
$S = S_{\text{exec}}\cdot S_{\text{ast}}$ with $S_{\text{exec}}\in\{0,1\}$, so the AST
can refine within an execution class but never merge across one. Concatenating the two
embeddings would let the unsound signal undo the only guarantee in the construction.

### One pass, three gradings

`item_evidence` executes each query once against the whole suite and keeps everything;
`exact_classes`, `semantic_classes` and `correct_flags` all derive from it, so the two
kernels are guaranteed to be compared on identical evidence and nothing is executed
twice.

`semantic_classes(..., merge_errors=False)` restores the repo convention exactly (a
query that failed everywhere keeps its own class, the AST gets no vote) — the only
configuration in which the semantic kernel is a **pure refinement** with no merging at
all.

**Correctness is not decided by the variants.** Whether a model is right is still
decided on the original database against gold, so row sampling changing a gold result
on a variant is harmless by construction. `correct_flags` additionally reports
test-suite correctness as a separate, stricter grading.

## B5. Measured (Spider, 150 target items, 10 models, K=3 variants, 60 instances, 317 MB)

**The exact path reproduces the stored `true_class` on 150/150 items**, so the two
kernels are being compared on a faithful reimplementation, not an approximation.

### Kernel diagnostics

| | exact | semantic (`merge_errors=False`) | semantic (`=True`) |
|---|---|---|---|
| informative variants / item | — | 2.96 | 2.96 |
| clusters / item | 1.58 | **1.72** | 1.70 |
| items split / merged | — | **16 / 0** | 15 / 2 |
| pairwise precision vs exact | — | **1.0000** | 0.9984 |
| mean semantic entropy | 0.2193 | 0.2739 | 0.2711 |

`merge_errors=False` measures **precision 1.0000 with zero merges** — the soundness
guarantee of §B4, confirmed rather than asserted. Turning the AST fallback on merges 2
of 150 items (both all-erroring queries), which is the only merge path in the design.

### The statistics move in the predicted direction

| | exact | semantic |
|---|---|---|
| $e$ off-diagonal mean | 0.1239 | **0.1278** |
| $\gamma^{\text{both}}$ mean | 0.1863 | 0.2209 |
| $P(C=0\mid Z=0)$ | 0.8466 | **0.8340** |
| $\beta$ | 0.8115 | 0.7869 |

**Why $e$ goes UP under a kernel that only ever splits.** The suite splits the
previously-single *correct* class: models that matched gold on the shipped instance by
coincidence are demoted into a shared **wrong** class, and shared wrong answers are
exactly what $e$ counts (class 0 is excluded from collisions by construction). So the
test suite **converts invisible shared error — hidden inside the "correct" class — into
visible shared error**, which is precisely the collusion signal the framework was blind
to. Meanwhile $P(C=0\mid Z=0)$ falls, as predicted: single-instance matching was
recording coincidental agreement as real.

Note the direction is the *opposite* of the free-text case. For SQL, canonicalised
result comparison already absorbs paraphrase-like variation (row order, types,
rounding), so the residual equality error is dominated by **over-merging** (coincidental
collisions), and the suite fixes it by splitting. For "Paris" vs "It's Paris" the
dominant error is **under-merging**. Both are equality errors; which one dominates is a
property of the output space, not of the method.

### Ground truth itself moves

| | mean | max |
|---|---|---|
| single-instance execution accuracy | 0.743 | 0.800 |
| test-suite accuracy | 0.709 | 0.760 |
| **overestimate** | **3.40 pts** | 4.7 pts |

51 of 1500 cells demoted, **and the pool ranking changes**. Both gradings are therefore
reported below, because scoring one estimate against one truth and another against a
different truth would be meaningless.

### MAE in accuracy points, lower is better

| method | exact kernel / exact truth | semantic kernel / exact truth | exact kernel / suite truth | semantic kernel / suite truth |
|---|---|---|---|---|
| prior only | 3.73 | 3.73 | 6.47 | 6.47 |
| PoolEval-SQL | 13.45 | **11.19** | 16.85 | **14.59** |
| collision EM (`.tex`) | 11.15 | **10.04** | 14.55 | **13.44** |
| validated EM (no judge) | 9.18 | **7.64** | 12.58 | **11.04** |
| validated EM + judge b=10 | 7.57 | **6.31** | 10.97 | **9.71** |
| validated EM + judge b=40 | **3.20** | 3.27 | **6.60** | 6.67 |

The semantic kernel improves **every** unsupervised estimator, by 1.1–2.3 points, and
the ordering of methods is unchanged — so the gain is a better observation, not a
re-ranking artefact. At budget 40 the judge already dominates and the two kernels tie,
which is the expected ceiling: once enough items are pinned to ground truth, how the
remaining ones are clustered matters less.

## B6. What Part B does **not** change

* `pooleval/kernel.py`, `zoo/execute.py`, `zoo/build.py` — untouched. `result_key` is reused as-is.
* No neural model, no download: `sqlglot` for the AST, SQLite for execution. A neural embedder or NLI backend is a pluggable `equivalent(i,j)` predicate for free-text domains, not a dependency.
* The vision and graph ports keep exact match, correctly: their label spaces are closed (10 and 5 classes) and models emit label indices, so exact match is already exact. Semantic clustering there could only wrongly merge two distinct labels.

---

# Part C — matching free-text answers from VLMs / GLMs

## C1. Scope

Text-to-SQL does not use any of this: the meaning of a query **is** its behaviour on
databases, so running it on more instances is the direct test (Part B). `zoo/exec_suite.py`
contains no clustering, no embedding and no threshold.

This part applies where the answer space is genuinely open -- a vision-language or
graph-language model replying "dog", "a dog", "It's a dog", "The answer is dog". The
CNN/GNN ports emit label indices and need nothing; the moment the pool is VLMs or GLMs,
they do.

The operationally cheapest fix remains **prompting for the required output format**. What
follows measures what is recoverable when you cannot, or when a model ignores the
instruction.

## C2. `pooleval/answer_matching.py`

| matcher | evidence | cost | needs |
|---|---|---|---|
| `ExactMatcher` | string equality | free | — |
| `NormalizeMatcher` | peel casing, punctuation, articles, meta-frames | free | — |
| `EmbeddingMatcher` | sentence-embedding cosine >= threshold | ~1 s | sentence-transformers |
| `EntailmentMatcher` | **bidirectional** entailment, Farquhar et al. (2024) | ~290 s | an NLI model |
| `LabelGroundingMatcher` | map onto a known closed label set, optionally with declared aliases | ~0.2 s | the label set |

Clustering is greedy first-match against representatives (`partitions.greedy_meaning_clusters`),
not connected components: bidirectional entailment is not transitive, and taking the
transitive closure CHAINS unrelated answers together. Merging is the expensive direction
here, so the closure that cannot chain is the one to use.

## C3. Measured -- 8 shifts x 2 surface conditions, COMPLETE

Predictions are rendered as free text with one house style per model plus per-item
jitter, holding accuracy and agreement structure exactly fixed. A controlled simulation
of surface variation, not real VLM output -- it bounds what a matcher can recover.
`easy` varies only the frame; `hard` also varies the label wording ("3"/"three",
"NN"/"Neural Networks").

**Per-shift MAE (PoolEval):**

| shift | exact | normalize | entailment | grounding+aliases | *oracle* |
|---|---|---|---|---|---|
| easy graph_AC | 45.70 | 12.31 | 20.84 | **12.31** | 12.31 |
| easy graph_CD | 51.84 | 12.36 | 24.68 | **12.36** | 12.36 |
| easy mnist→usps | 55.00 | 7.68 | 23.47 | **7.68** | 7.68 |
| easy mnist→svhn | **20.64** | 55.79 | 28.92 | 55.79 | 55.79 |
| hard graph_AC | 51.99 | 27.20 | 35.05 | **12.31** | 12.31 |
| hard graph_CD | 54.57 | 32.05 | 39.66 | **12.36** | 12.36 |
| hard mnist→usps | 46.79 | 46.61 | 30.19 | **7.68** | 7.68 |
| hard mnist→svhn | **24.03** | 24.24 | 27.92 | 55.79 | 55.79 |

**Worst-case summary, hard surface (8 shifts).** The minimum-precision column is what
decides deployability: a matcher that helps on average while merging distinct classes
somewhere is not one you can ship without knowing in advance which case you are in.

| matcher | min P | mean R | mean MAE | worst MAE |
|---|---|---|---|---|
| exact string | 0.915 | 0.060 | 46.22 | 54.57 |
| normalize | 1.000 | 0.244 | 31.00 | 46.61 |
| embedding | **0.653** | 0.202 | 32.38 | 47.90 |
| bidirectional entailment | 0.984 | 0.197 | 32.78 | 39.66 |
| normalize + entailment | 0.988 | 0.298 | 26.10 | 29.87 |
| label grounding | 0.990 | 0.374 | 23.90 | 33.22 |
| **label grounding + declared aliases** | **1.000** | **1.000** | 19.98 | 55.79 |

### What this settles

1. **`label grounding + declared aliases` achieves P = R = 1.000 on all 16 cases.** It
   does not approximate the oracle partition; it IS the oracle, in both conditions, at
   0.2 s. A closed label set arrives with the vocabulary its classes are named in;
   declaring it costs one dictionary and solves surface variation exactly.
2. **Matching helps on 7 of 8 shifts, by a lot** -- 51.99 → 12.31, 46.79 → 7.68. An
   earlier note in this file claimed matchers recover "about a third" of the damage. That
   was drawn from one shift in the easy condition and was wrong.
3. **Embedding similarity stays out.** Min precision 0.653: it merges distinct classes,
   which in a framework that reads agreement as evidence manufactures consensus. It is
   kept in the codebase only as the reproducible negative control.
4. **Bidirectional entailment is conservative and mis-specified for this task.** It never
   merges two different digits, and it misses much because it requires MUTUAL entailment,
   which specificity breaks: "3" ⇄ "three" scores 0.841 but "3" ⇄ "numeral 3" only 0.336
   (one-way: "numeral 3" ⊨ "3" at 0.914), and "three" ⇄ "a handwritten three" collapses to
   0.006. Farquhar et al. need *semantic equivalence of sentences*; pool evaluation needs
   *reference to the same class*. Different relations.

### The 8th shift is not a matcher failure

On MNIST→SVHN the **oracle partition is worse than exact-string match** (55.79 vs 20.64).
The cause is measurable:

| partition | classes/item | estimated acc | true acc |
|---|---|---|---|
| oracle | 4.38 | **0.614** | 0.134 |
| exact-string shattered | 10.80 | 0.179 | 0.134 |

The pool is 13% accurate and agrees on wrong answers. Recovering the true partition
restores that false consensus and the estimator -- whose premise is agreement ⇒
correctness -- credits it at 0.614 against a truth of 0.134. Exact-string shattering
breaks the consensus by accident (10.8 singleton classes per item) and lands near 0.179
for entirely the wrong reason. So the negative worst-case gain in the summary is **not**
evidence against matching: it is the gauge trap, and MAE there measures the estimator's
failure rather than the partition's quality. On partition quality -- what a matcher is
responsible for -- grounding+aliases is perfect on all 16.

## C4. What Part C does **not** change

* No estimator, no EM, no kernel. Matchers produce a partition; everything downstream
  reads the partition.
* Nothing is downloaded unless `EmbeddingMatcher` or `EntailmentMatcher` is constructed.
* `tests/test_answer_matching.py` drives both neural backends through **injected
  scorers**, so the suite needs no model and stays deterministic.

---

## How to run everything

```bash
python -m pytest tests/test_validated_em.py tests/test_semantic.py -q   # 44 tests

python experiments/run_validated_em.py --domain text2sql \
    --budgets 0 5 10 20 40 --ig-candidates 50 --ig-iters 5 \
    --out results/validated_em_text2sql.json
python experiments/run_validated_em.py --domain vision \
    --n-target 600 --n-labeled 3000 --ig-candidates 25 \
    --out results/validated_em_vision.json
python experiments/run_validated_em.py --domain graph \
    --n-target 600 --n-labeled 3000 --ig-candidates 25 \
    --out results/validated_em_graph.json
python experiments/report_validated_em.py

python experiments/run_exec_suite.py --datasets spider --variants 3
python experiments/run_answer_matching.py --domain vision graph --surface easy hard
```

Acquisition head-to-head (`A_mu` vs label-entropy IG vs random vs hybrid) and the
design-based estimators: see `docs/acquisition_results.md`.

## Open decisions

1. **Which `gamma` conditioning is canonical** — `both_wrong` (what the spec counts) or `model_wrong` (what the `.tex`'s $Z$ means). Both run; the ablation tables decide.
2. **Do two queries that both crash count as the same wrong answer?** The repo says no (unique negative ids); the AST fallback says maybe. It is the only merge path in the whole design and produced both merges above. `merge_errors` switches it.
