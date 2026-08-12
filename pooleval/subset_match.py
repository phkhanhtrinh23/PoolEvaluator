"""Subset selection to match an unlabeled test set: facility+MMD vs ColBERT+MMD.

This implements the objective documented in `docs/facility_location_mmd.md`.

Two things live at DIFFERENT layers:

  * SIMILARITY -- how close two examples are. We use ColBERT-style MaxSim on a
    per-token (multi-vector) representation instead of a single pooled vector,
    so a few decisive tokens are not averaged away.

  * SELECTION OBJECTIVE -- given similarities, which subset A (|A| <= k)? We
    compare two first terms, each paired with the SAME MMD density term:

        facility :  f_cov(A)  = sum_i w_i * max_{j in A} S_ij     (SUBMODULAR)
        relevance:  f_rel(A)  = sum_{j in A} ( sum_i w_i S_ij )   (MODULAR, = "ColBERT top-k")

        maximize   first_term(A)  -  beta * MMD^2(A, T)

MMD needs a POSITIVE SEMI-DEFINITE kernel; MaxSim is not PSD (the max breaks it),
so MMD uses a separate PSD mean-map (kernel-mean) kernel built from the SAME token
vectors. See section 9 of the doc. Both first terms feed the same greedy loop.
"""
import numpy as np


# --------------------------------------------------------------------------- #
#  Similarity (ColBERT / MaxSim) -- multi-vector, NOT pooled                   #
# --------------------------------------------------------------------------- #
def maxsim_matrix(E_query, E_key):
    """Late-interaction similarity S[i,j] between example i (query side) and j (key side).

    E_* are [n, L, d] arrays of L UNIT-NORM token vectors per example. For unit
    vectors the inner product is cosine, so

        S[i,j] = mean_f  max_g  <E_query[i,f], E_key[j,g]>

    (for each query token f take its best-matching key token g, then average).
    Not symmetric and not PSD -- fine for facility location, which only needs a
    similarity, but NOT usable as the MMD kernel (see `meanmap_blocks`)."""
    nq, L, d = E_query.shape
    nk = E_key.shape[0]
    S = np.empty((nq, nk))
    for i in range(nq):
        # cross[f, j, g] = <E_query[i,f], E_key[j,g]>
        cross = np.einsum("fd,jgd->fjg", E_query[i], E_key)
        S[i] = cross.max(axis=2).mean(axis=0)   # max over key tokens, mean over query tokens
    return S


# --------------------------------------------------------------------------- #
#  PSD mean-map kernel for MMD -- keeps multi-vector, restores PSD             #
# --------------------------------------------------------------------------- #
def median_gamma(E, sample=2000, seed=0):
    """RBF gamma = 1/(2 sigma^2) from the median pairwise token distance (median heuristic)."""
    tok = E.reshape(-1, E.shape[-1])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tok), size=min(sample, len(tok)), replace=False)
    T = tok[idx]
    d2 = np.maximum(0.0, (T * T).sum(1)[:, None] + (T * T).sum(1)[None, :] - 2 * T @ T.T)
    med = np.median(d2[d2 > 0])
    sigma2 = med / 2.0 if med > 0 else 1.0
    return 1.0 / (2.0 * sigma2)


def _rbf_meanmap(A, B, gamma):
    """Mean-map kernel K[i,j] = mean_{f,g} exp(-gamma ||A[i,f]-B[j,g]||^2).

    This is <mu(A_i), mu(B_j)> for the RBF-induced feature mean, hence PSD."""
    Na, L, d = A.shape
    B2 = (B * B).sum(-1)                     # [Nb, L]
    K = np.empty((Na, B.shape[0]))
    for i in range(Na):
        a = A[i]                             # [L, d]
        a2 = (a * a).sum(-1)                 # [L]
        cross = np.einsum("fd,jgd->fjg", a, B)          # [L, Nb, L]
        d2 = a2[:, None, None] + B2[None, :, :] - 2 * cross
        K[i] = np.exp(-gamma * d2).mean(axis=(0, 2))
    return K


def meanmap_blocks(E_pool, E_test, gamma):
    """Precompute the kernel blocks MMD needs.

    Returns (K_PP, kt, C):
      K_PP  [N,N]  pool-pool mean-map kernel
      kt    [N]    kt[j] = mean_i K(pool_j, test_i)   (pool-to-test average)
      C     scalar mean_i,i' K(test_i, test_i')       (constant in A; = ||mu_T||^2)
    """
    K_PP = _rbf_meanmap(E_pool, E_pool, gamma)
    K_PT = _rbf_meanmap(E_pool, E_test, gamma)
    kt = K_PT.mean(axis=1)
    K_TT = _rbf_meanmap(E_test, E_test, gamma)
    C = float(K_TT.mean())
    return K_PP, kt, C


def mmd2_of(indices, K_PP, kt, C):
    """MMD^2(A, T) for a chosen index set A, from precomputed blocks.

        MMD^2 = 1/|A|^2 sum_{a,b in A} K_PP - 2/|A| sum_{a in A} kt_a + C
    """
    A = list(indices)
    m = len(A)
    if m == 0:
        return C                                   # mu(empty)=0 -> ||mu_T||^2
    AA = K_PP[np.ix_(A, A)].sum()
    AT = kt[np.asarray(A)].sum()
    return float(AA / (m * m) - 2.0 * AT / m + C)


# --------------------------------------------------------------------------- #
#  Greedy selection: first_term(A) - beta * MMD^2(A, T)                        #
# --------------------------------------------------------------------------- #
def select(S, K_PP, kt, C, budget, beta=1.0, mode="facility", w=None):
    """Greedily maximize  first_term(A) - beta * MMD^2(A, T)  under |A| <= budget.

    mode="facility"  : first term = sum_i w_i max_{j in A} S_ij   (submodular coverage)
    mode="relevance" : first term = sum_{j in A} sum_i w_i S_ij   (modular; "ColBERT top-k")
    mode="mmd_only"  : first term = 0                             (pure density match)

    Incremental bookkeeping keeps each round O(N): AA_sum, AT_sum and the per-pool
    cross vector are updated after every pick, so MMD^2 gains are computed in O(1)
    per candidate. Plain (non-lazy) greedy -- with the -beta*MMD^2 term the objective
    is not monotone, so CELF's certificate would not apply anyway; this stays exact
    and transparent for the comparison."""
    M, N = S.shape
    w = np.full(M, 1.0 / M) if w is None else np.asarray(w, float)
    rel = (w[:, None] * S).sum(axis=0)             # rel[j] = sum_i w_i S_ij (modular term)

    chosen = []
    cover = np.zeros(M)                            # max_{j in A} S_ij
    AA_sum = 0.0                                   # sum_{a,b in A} K_PP[a,b]
    AT_sum = 0.0                                   # sum_{a in A} kt_a
    cross = np.zeros(N)                            # cross[j] = sum_{a in A} K_PP[a,j]
    mmd_cur = C                                    # MMD^2 of current set (empty -> C)
    avail = list(range(N))

    for _ in range(min(budget, N)):
        m = len(chosen)
        # MMD^2 of A + {j} for every candidate at once
        new_AA = AA_sum + 2.0 * cross + np.diag(K_PP)
        new_AT = AT_sum + kt
        mmd_new = new_AA / ((m + 1) ** 2) - 2.0 * new_AT / (m + 1) + C
        d_mmd = mmd_new - mmd_cur                  # >=0 penalty change

        if mode == "facility":
            d_first = (w[:, None] * np.maximum(0.0, S - cover[:, None])).sum(axis=0)
        elif mode == "relevance":
            d_first = rel                          # constant marginal (modular)
        elif mode == "mmd_only":
            d_first = np.zeros(N)
        else:
            raise ValueError(mode)

        gain = d_first - beta * d_mmd
        gain[chosen] = -np.inf                     # do not re-pick
        j = int(np.argmax(gain))

        chosen.append(j)
        cover = np.maximum(cover, S[:, j])
        AA_sum = float(new_AA[j])
        AT_sum = float(new_AT[j])
        cross = cross + K_PP[:, j]
        mmd_cur = float(mmd_new[j])
    return chosen


def select_topk_maxsim(S, budget, w=None):
    """Pure ColBERT-style retrieval: rank pool by aggregate MaxSim to the test set,
    take the top-k. Modular, no anti-redundancy, no density term (beta=0 relevance)."""
    M, N = S.shape
    w = np.full(M, 1.0 / M) if w is None else np.asarray(w, float)
    rel = (w[:, None] * S).sum(axis=0)
    return list(np.argsort(-rel)[:budget])
