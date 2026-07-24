# Selecting Subsets Close to the Unlabeled Test Set

**Goal.** From a pool of examples drawn from `FusionDataset`, pick a subset
$A$ (under a budget $|A| \le k$) whose *distribution* is close to an
**unlabeled test set** $T$. We combine two complementary set-level objectives:

- **Facility location** — guarantees every test example is *covered* by some
  chosen pool example (no test region is missed).
- **Maximum Mean Discrepancy (MMD)** — matches the *density / proportions* of
  the subset to the test set (no region is over- or under-represented).

Both are set functions optimized by the same lazy-greedy (CELF) loop already in
`pooleval/active.py` (`greedy_submodular`).

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $P = \{x_1,\dots,x_N\}$ | pool of candidate examples (from `FusionDataset`) |
| $T = \{t_1,\dots,t_M\}$ | unlabeled test set (target distribution) |
| $A \subseteq P$ | selected subset, $|A|\le k$ |
| $S_{ij}$ | similarity between test example $t_i$ and pool example $x_j$, $S_{ij}\in[0,1]$ |
| $\phi(\cdot)$ | feature map of a characteristic kernel $k(x,y)=\langle\phi(x),\phi(y)\rangle$ |
| $\text{value}_j$ | acquisition weight of pool example $x_j$ (optional) |

---

## 2. Facility-location coverage (test-anchored)

Each **test** example is served by its single nearest **chosen** example:

$$
f_{\text{cov}}(A) \;=\; \sum_{i=1}^{M} w_i \, \max_{j \in A} S_{ij},
$$

where $w_i \ge 0$ weights test example $t_i$ (uniform $w_i = 1/M$ by default).

The inner $\max$ is what prevents *centroid collapse*: adding a pool example
only increases $f_{\text{cov}}$ if it covers a **test region no chosen example
covers yet**. $f_{\text{cov}}$ is monotone and submodular, so greedy achieves

$$
f_{\text{cov}}(A_{\text{greedy}}) \;\ge\; \left(1 - \tfrac{1}{e}\right) f_{\text{cov}}(A^\star),
$$

and the bound is tight (Nemhauser–Wolsey–Fisher, 1978).

> **Relation to the current code.** `greedy_submodular` maximizes
> $\sum_{j\in A}\text{value}_j + \lambda \sum_i \text{value}_i \max_{j\in A} S_{ij}$,
> where $i$ ranges over the **pool**. The change here is to let $i$ range over
> the **test set** $T$ so coverage is measured against the target distribution.

---

## 3. Maximum Mean Discrepancy (density match)

Facility location covers the *support* of $T$ but does not match its *density*
— it will happily over-sample a sparse test corner. MMD fixes the proportions
by comparing mean embeddings in an RKHS:

$$
\mathrm{MMD}^2(A, T)
= \left\lVert \frac{1}{|A|}\sum_{j \in A} \phi(x_j)
           - \frac{1}{|T|}\sum_{i=1}^{M} \phi(t_i) \right\rVert_{\mathcal H}^2 .
$$

### What is an RKHS? (Reproducing Kernel Hilbert Space)

For our purposes, an RKHS is just *the feature space a kernel secretly works
in.* Built up in steps:

**1. The problem.** Comparing raw average vectors is too weak: two distributions
can share the same mean yet differ in spread, skew, or clustering. We need to
compare *all* the structure, not just the average.

**2. The feature map $\phi$.** A kernel comes with a (usually implicit, possibly
infinite-dimensional) feature map that lifts each point into a richer space,
$x \mapsto \phi(x)$. For the RBF kernel, $\phi(x)$ is **infinite-dimensional**
and encodes information about *all moments* of the point. That lifted space is
the **Hilbert space** $\mathcal H$ (a vector space with a well-behaved inner
product, possibly infinite-dimensional).

**3. The "reproducing" property.** We never compute $\phi(x)$ directly — it may
be infinite-dimensional. The kernel gives inner products in $\mathcal H$ for
free:

$$
k(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal H}.
$$

This is why $\mathrm{MMD}^2$ can be written purely in terms of $k(\cdot,\cdot)$
(next equation): you compute kernels, and the RKHS geometry comes along for the
ride.

**4. Mean embedding of a distribution.** Instead of averaging raw points,
average their *lifted* versions:

$$
\mu_A = \frac{1}{|A|}\sum_{j\in A}\phi(x_j), \qquad
\mu_T = \frac{1}{|T|}\sum_i \phi(t_i).
$$

Each $\mu$ is a **single point in $\mathcal H$ that summarizes an entire
distribution** — and because $\phi$ carried all moments, it captures spread,
skew, and modes, not just the ordinary average.

**5. MMD is the distance between these summaries.**
$\mathrm{MMD}^2(A,T) = \lVert \mu_A - \mu_T \rVert_{\mathcal H}^2$, i.e. the
Euclidean distance between the two mean embeddings *inside the RKHS*. For a
**characteristic** kernel (like RBF), $\mu_A = \mu_T \iff$ the two distributions
are identical.

> **One-line intuition.** A plain mean can only catch a difference in *average*;
> a mean *in an RKHS* catches a difference in the *whole distribution*, because
> the kernel secretly averaged all the moments too. Driving MMD to zero forces
> the subset's full distribution — proportions and all — to match the test
> set's.

Expanded into kernel evaluations (what you actually compute):

$$
\mathrm{MMD}^2(A, T)
= \frac{1}{|A|^2}\sum_{j,j'\in A} k(x_j, x_{j'})
- \frac{2}{|A||T|}\sum_{j\in A}\sum_{i} k(x_j, t_i)
+ \underbrace{\frac{1}{|T|^2}\sum_{i,i'} k(t_i, t_{i'})}_{\text{const. in } A}.
$$

A common characteristic kernel is the RBF (Gaussian):

$$
k(x,y) = \exp\!\left(-\frac{\lVert x - y\rVert^2}{2\sigma^2}\right).
$$

Because the RBF kernel is **characteristic**, $\mathrm{MMD}(A,T)=0 \iff$ the two
distributions are identical — matching the mean in $\mathcal H$ matches **all
moments** in input space, not just the first.

---

## 4. Combined objective

Maximize coverage while penalizing distribution mismatch:

$$
\boxed{\;
A^\star \;=\; \arg\max_{|A|\le k}\;
\underbrace{f_{\text{cov}}(A)}_{\text{cover the support}}
\;-\; \beta\,\underbrace{\mathrm{MMD}^2(A, T)}_{\text{match the density}}
\;}
$$

- $\beta \ge 0$ trades coverage against density match ($\beta = 0$ recovers pure
  facility location).
- The last term of $\mathrm{MMD}^2$ ($\frac{1}{|T|^2}\sum k(t_i,t_{i'})$) is
  **constant in $A$** and can be dropped during optimization.
- Both terms are set functions, so the **same lazy-greedy loop** applies; only
  the marginal-gain computation changes.

### Marginal gain (what greedy evaluates per candidate)

For adding example $x_j$ to the current set $A$, with running coverage
$c_i = \max_{j'\in A} S_{ij'}$:

$$
\Delta_{\text{cov}}(j \mid A) = \sum_{i=1}^{M} w_i \,\max\!\big(0,\; S_{ij} - c_i\big),
$$

$$
\Delta_{\text{mmd}}(j \mid A) = \mathrm{MMD}^2(A\cup\{j\}, T) - \mathrm{MMD}^2(A, T),
$$

$$
\Delta(j \mid A) = \Delta_{\text{cov}}(j\mid A) \;-\; \beta\,\Delta_{\text{mmd}}(j\mid A).
$$

The coverage gain is exactly the `np.maximum(0.0, S[:, j] - cover)` term in the
current `gain(j)`. The MMD gain is a new additive term; $-\mathrm{MMD}^2$ is
concave and behaves well under greedy in practice (it is not modular, so
`(1-1/e)` applies strictly only to the facility-location part).

> **Note on the CELF bound.** Lazy greedy's optimality certificate assumes a
> monotone submodular $f$. With the $-\beta\,\mathrm{MMD}^2$ term the objective
> is submodular-*ish* but not monotone; treat the greedy result as a strong
> heuristic here, or fix $\beta$ small so coverage dominates the ordering.

---

## 5. The similarity $S_{ij}$: avoid pooling away information

A single pooled embedding + cosine can average away the few tokens/fields that
decide a text2SQL example (a column name, `GROUP BY`, a nested subquery).
Represent each example as a **set of component vectors** (per token, or per
field: question tokens, table/column names, SQL clause types) and use a
**MaxSim (late-interaction)** similarity — the ColBERT idea, applied
**example-to-example** (both sides are your own examples, no query/doc roles):

$$
S(a, b) \;=\; \sum_{f \in a}\; \max_{g \in b}\; \big\langle E_a[f],\, E_b[g] \big\rangle .
$$

For each component $f$ of example $a$, take its best-matching component $g$ in
example $b$, then sum. This preserves rare/decisive components that mean-pooling
destroys. Use $S(t_i, x_j)$ as the $S_{ij}$ feeding facility location, and/or as
the kernel feeding MMD.

---

## 6. Algorithm sketch

```text
Input : pool P, test set T, budget k, weight beta, kernel k(.,.)
1. Build component embeddings for every example in P and T.
2. Precompute S_ij = MaxSim(t_i, x_j)  for i in T, j in P.        (coverage)
3. Precompute kernel blocks K_PP, K_PT (K_TT const, drop it).     (MMD)
4. cover_i <- 0 for all i in T
   A <- {}
5. while |A| < k:                       # lazy-greedy / CELF
      for each candidate j (top of heap):
         d_cov  = sum_i w_i * max(0, S_ij - cover_i)
         d_mmd  = MMD^2(A ∪ {j}, T) - MMD^2(A, T)   # incremental update
         gain_j = d_cov - beta * d_mmd
      pick argmax gain_j, add to A, update cover_i = max(cover_i, S_ij)
6. return A
```

Steps 2–5 mirror `greedy_submodular` in `pooleval/active.py`; the additions are
the test-anchored $S_{ij}$, the MMD blocks, and the `- beta * d_mmd` term in the
gain.

---

## 7. Intuition: "support" vs "density" (why facility location alone is not enough)

- **Support** = *where* the test examples are — which regions contain at least
  one test point.
- **Density** = *how many* test examples are in each region — the proportions.

Facility location only cares about **support**. In

$$
f_{\text{cov}}(A) = \sum_{i} w_i \max_{j\in A} S_{ij},
$$

each test point $i$ contributes through its **single nearest** chosen point.
Once a test point has *one* close neighbor in $A$, the $\max$ is (nearly)
satisfied and covering it again adds almost nothing. The objective says "give
every test point at least one nearby pick" — it says nothing about matching how
many test points live where.

### Worked example

Test set $T$ has 100 examples:

- **95** simple single-table `SELECT` queries — one dense region.
- **5** five-way-join / nested-subquery queries — one sparse corner.

Pick a subset of $k = 10$.

Facility location covers the 95 similar simple queries with ~3 picks (one pick
covers many), then spends ~7 picks covering the 5 join queries, because they are
mutually dissimilar and each demands its own nearby pick:

| region | % of test set | picks facility-location gives it |
|--------|---------------|----------------------------------|
| simple (dense) | 95% | ~3 |
| joins (sparse) | 5% | ~7 |

That is **over-sampling the sparse corner**: 70% of the budget went to 5% of the
test distribution. Evaluating a model on this subset would be dominated by hard
join queries and would not reflect real test performance (95% simple).

### What each objective does about it

- **Facility location alone** → covers every corner, but wrong proportions.
- **MMD alone** → right proportions, but can skip a small region entirely if
  dropping it barely moves the mean (a rare cluster gets zero picks).
- **Both** (the $f_{\text{cov}}(A) - \beta\,\mathrm{MMD}^2(A,T)$ objective) →
  cover every region *and* in roughly test-matching proportions.

MMD compares **average embeddings** (proportions), so to shrink it the subset's
mixture must resemble the test mixture (~95% simple, ~5% joins), pulling the
budget back toward the true proportions.

---

## 8. Is this "better than ColBERT"? (they are different layers)

ColBERT/MaxSim and facility-location+MMD are **not rivals** — they operate at
different layers and compose:

| Layer | Job | Options |
|-------|-----|---------|
| **Similarity** | how close are two examples? | pooled cosine, **ColBERT/MaxSim**, … |
| **Selection objective** | given similarities, which subset? | top-$k$, **facility location + MMD**, k-center, … |

ColBERT/MaxSim *produces* the number $S_{ij}$. Facility-location+MMD *consumes*
$S_{ij}$ to choose a set. So MaxSim can be the similarity feeding the objective —
"X vs Y" is ill-posed.

### The real comparison: top-$k$ MaxSim retrieval vs submodular selection

If ColBERT is used the usual way — rank all pool examples by MaxSim to the test
set and take the top $k$ — *that* is a selection scheme, and the comparison
becomes meaningful. Theoretically the submodular objective wins on subset
quality:

1. **Top-$k$ is redundant — it ignores what is already picked.** It maximizes a
   **modular** objective $\sum_{j\in A}\text{score}_j$ with no diminishing
   returns, so the top picks are often near-duplicates of the single densest
   test mode. Facility location is **submodular** (the $\max_{j\in A}$ makes each
   pick's value depend on the current set), so it stops piling onto a covered
   region — and enjoys the $(1-1/e)$ guarantee that top-$k$ does not.

2. **Top-$k$ matches neither support nor density — only score.** It has no
   mechanism to correct proportions; MMD does. On the dense/sparse example
   above, top-$k$ MaxSim would put all 10 picks on the 95% simple region
   (highest scores) and **miss the joins entirely**.

### Verdict

- **As a similarity**, ColBERT/MaxSim is **better** than pooled cosine
  (preserves decisive tokens) — use it *inside* the objective as $S_{ij}$.
- **As a selection method** (top-$k$ retrieval), it is **worse** than
  facility-location+MMD: modular top-$k$ has no anti-redundancy and no
  density-matching, while the submodular objective has both, with an
  approximation bound top-$k$ lacks.
- **When top-$k$ is fine anyway:** it is far cheaper (one independent scan, no
  $O(k\cdot N)$ greedy, no $|A|^2$ MMD kernel), so if the test set is essentially
  **unimodal**, redundancy and proportion-matching do not matter and its
  simplicity is a fair trade. The submodular machinery earns its cost when the
  test distribution is **multi-modal / imbalanced** — which text2SQL difficulty
  usually is.

**Bottom line:** use MaxSim *as* $S_{ij}$ and facility-location+MMD *as* the
selector. They are complements; the only honest "better than" claim is that
submodular selection dominates top-$k$-retrieval-as-selection on subset quality.

---

## 9. Combining MMD with ColBERT (and the PSD obstruction)

Yes, MMD and ColBERT can — and should — be combined, but *how* is constrained by
one theoretical fact.

### The catch: MMD needs a valid kernel; MaxSim is not one

MMD is defined only for a **positive semi-definite (PSD)** kernel — that is what
guarantees an RKHS exists and that $\mathrm{MMD}^2 \ge 0$ with
$\mathrm{MMD}^2 = 0 \iff$ same distribution. ColBERT's **MaxSim is not PSD**:

$$
S(a,b) = \sum_{f\in a}\max_{g\in b}\langle E_a[f], E_b[g]\rangle,
$$

because the $\max$ (and the asymmetry $S(a,b)\neq S(b,a)$) breaks
positive-definiteness. If you naively set $k(a,b)=\text{MaxSim}(a,b)$, then

- $\mathrm{MMD}^2$ can go **negative** (no longer a squared norm), and
- the "$\mathrm{MMD}=0 \iff$ distributions match" guarantee **disappears**.

So the only thing you cannot do is drop MaxSim directly into MMD.

### The right way: keep the multi-vector representation, use a PSD set-kernel

The valuable part of ColBERT is the **multi-vector representation** (a *set* of
token/field vectors per example), not the $\max$ itself. Keep the representation
and build a PSD **kernel-mean / mean-map** kernel on top:

$$
k(a,b) = \frac{1}{|a|\,|b|}\sum_{f\in a}\sum_{g\in b} k_0\big(E_a[f],\, E_b[g]\big),
$$

with $k_0$ an ordinary PSD kernel (e.g. RBF on token vectors). This is PSD (it is
literally an inner product of the two examples' mean token-embeddings), so MMD is
valid, *and* it still uses per-token vectors instead of one pooled vector.

**Trade-off.** Going from $\max$ to $\sum$ (mean) reintroduces *some* averaging —
the very thing ColBERT's $\max$ avoids. It is milder here (averaging is over
token *pairs* inside the kernel, not a collapse to one vector first), but it is a
real tension.

### Cleanest design: use each aggregation where it is legal

The objective has two terms with different requirements, so use a different
token-aggregation for each — both computed from the **same ColBERT token
embeddings**:

| Term | Needs PSD? | Aggregation over ColBERT vectors |
|------|-----------|----------------------------------|
| Facility location $f_{\text{cov}}$ | **No** — any sensible similarity works | **MaxSim** (keep the $\max$; its discriminative power is the point) |
| MMD $\mathrm{MMD}^2$ | **Yes** — must be a valid kernel | **mean-map set-kernel** $k(a,b)$ above (PSD, still multi-vector) |

Facility location never asks for a kernel — it only needs "is $t_i$ close to
$x_j$," so MaxSim is perfect and legal there. MMD needs PSD, so use the mean-map
kernel there. Same multi-vector backbone, two aggregations, each placed where it
is theoretically sound:

$$
A^\star = \arg\max_{|A|\le k}\;
\underbrace{\sum_i w_i \max_{j\in A}\text{MaxSim}(t_i,x_j)}_{\text{coverage — MaxSim}}
\;-\; \beta\,
\underbrace{\mathrm{MMD}^2_{k_{\text{mean-map}}}(A,T)}_{\text{density — PSD set-kernel}} .
$$

---

## 10. Which is theoretically sounder for *subset* selection: facility+MMD or ColBERT+MMD?

For choosing a **subset** (not a single sample), **facility-location + MMD** is
the sounder pairing — and the "subset, not single sample" framing is exactly what
tips it.

### Align the comparison first

ColBERT is a *similarity*, not a selection objective, so "ColBERT + MMD" as a
selector means: use MaxSim as a **relevance / top-$k$** term + MMD. The real
question is which **coverage term** to pair with MMD:

| Option | Coverage term | Nature |
|--------|---------------|--------|
| Facility + MMD | $\sum_i w_i \max_{j\in A} S_{ij}$ | **submodular** (diminishing returns) |
| ColBERT + MMD | top-$k$ by $\text{MaxSim}(t_i, x_j)$ | **modular** (independent relevance) |

MaxSim can be the underlying $S_{ij}$ in *both*; the difference is submodular
coverage vs modular relevance.

### Why facility + MMD wins for subsets

1. **Subset selection is a set problem → you need submodularity.** Modular
   top-$k$ scores each example independently, so it picks near-duplicates of the
   densest test mode — high individual relevance, poor *set*. Facility location's
   $\max_{j\in A}$ gives diminishing returns: a redundant pick adds ~nothing.
   This concern exists only because you pick *many*; for a single sample ($k=1$)
   the distinction vanishes — which is why "not a single sample" matters.

2. **The two terms are complementary — they cover each other's blind spots.**
   - **MMD** matches *density* but can silently **drop a rare mode** (removing a
     tiny cluster barely moves the mean).
   - **Facility location** guarantees *support coverage* — every region,
     including rare ones, gets a representative.

   Together they attack **orthogonal** failure modes: density + support.

3. **ColBERT + MMD is partly redundant, not complementary.** Top-$k$ MaxSim
   relevance and MMD are *both* pulled toward high-density, "typical" regions, so
   the terms overlap instead of complementing — and **neither** rescues a rare
   mode: MMD can drop it, and top-$k$ relevance won't pick it (rare examples are
   not especially "relevant" to the bulk test set). Two terms fighting over the
   dense center, no coverage guarantee.

4. **The guarantee.** Facility location is monotone submodular → greedy gets
   $(1-1/e)\cdot\text{OPT}$. Modular top-$k$ has no such *set-level* optimality
   once coupled with the MMD penalty.

### Honest synthesis

This is not really "facility vs ColBERT." Use **ColBERT/MaxSim as the similarity**
$S_{ij}$ *underneath* facility location, and the mean-map kernel for MMD (the §9
design). Then you get ColBERT's discriminative similarity **and** submodular
coverage **and** density matching. "ColBERT + MMD" (top-$k$ + density) throws
away the submodular coverage that subset selection specifically needs.

> **Bottom line.** For subsets, **facility-location + MMD**, with ColBERT as the
> similarity feeding it — because coverage and density are *complementary*, while
> relevance and density are *redundant*.

---

## References

- Nemhauser, Wolsey, Fisher (1978). *An analysis of approximations for
  maximizing submodular set functions.*
- Minoux (1978); Leskovec et al. (2007, CELF). *Lazy greedy.*
- Gretton et al. (2012). *A kernel two-sample test* (MMD).
- Khattab & Zaharia (2020). *ColBERT: late interaction / MaxSim.*
