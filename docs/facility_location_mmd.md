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

## References

- Nemhauser, Wolsey, Fisher (1978). *An analysis of approximations for
  maximizing submodular set functions.*
- Minoux (1978); Leskovec et al. (2007, CELF). *Lazy greedy.*
- Gretton et al. (2012). *A kernel two-sample test* (MMD).
- Khattab & Zaharia (2020). *ColBERT: late interaction / MaxSim.*
