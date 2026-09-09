"""Coverage -> accuracy bound, and the two effective sample sizes.

This module implements, as executable code, the two derivations in
``pool_text2sql_crowd_source_mixture/pdfs/``:

  * ``coverage_hoeffding_derivation.md`` -- dataset coverage is monotone submodular,
    greedy gets ``alpha_K >= 1 - 1/e`` of optimal coverage, and (under smoothness +
    conditional independence) coverage plus a weighted Hoeffding term bound the error
    of a *target-matched* calibration estimator.

  * ``MTM08_analysis.md`` -- the Morita-Thall-Mueller curvature-matched effective
    sample size of a Beta prior is the sum of its two parameters, so PoolEval's
    ``Beta(1 + s*pi, 1 + s*(1-pi))`` anchor carries ``ESS = s + 2``.

TWO DIFFERENT THINGS ARE BOTH CALLED "EFFECTIVE SAMPLE SIZE" AND THEY ARE NOT THE
SAME NUMBER. Keeping them apart is most of the point of this file:

    n_eff  = 1 / sum_z w_z^2       CONCENTRATION ESS. How many independent bounded
                                   observations a *weighted average* is worth. Lives
                                   in the Hoeffding term. Depends only on how the
                                   target items distribute over their nearest
                                   calibration items.

    s + 2                          PRIOR ESS (MTM08). How many observations a Beta
                                   prior is worth against a Bernoulli likelihood,
                                   measured by matching log-density curvature. Lives
                                   in the MAP update, where the prior's mixing weight
                                   is s / (N + s).

Nothing here estimates a Lipschitz constant for free: :func:`estimate_lipschitz` is a
kNN-smoothed empirical proxy, and :func:`required_lipschitz` inverts the bound to ask
"how non-smooth would expected correctness have to be for this bound to fail?".
"""
import numpy as np

__all__ = [
    "shifted_cosine", "best_match", "coverage", "match_weights", "n_eff",
    "tv_to_uniform", "match_distances", "uniform_estimate", "weighted_estimate",
    "transfer_term", "hoeffding_term", "bound_weighted", "bound_uniform",
    "bound_realized", "required_neff", "required_coverage", "required_lipschitz",
    "bound_in_range", "alpha_K", "greedy_cover", "optimal_cover",
    "beta_curvature", "ess_curvature_delta", "beta_ess", "power_prior_beta",
    "prior_ess", "map_weight", "map_update", "estimate_lipschitz",
]


# --------------------------------------------------------------------------- #
#  Sections 2-3: similarity and coverage                                       #
# --------------------------------------------------------------------------- #
def shifted_cosine(phi_x, phi_z):
    """S[x, z] = (1 + <phi(x), phi(z)>) / 2 in [0, 1] for UNIT-NORM rows.

    The shift is not cosmetic. It is what makes the distance identity of Section 9
    exact: ||phi(x) - phi(z)||^2 = 4 (1 - S(x, z)). Without it, coverage of the empty
    selection could not be defined as zero without the first addition being able to
    *reduce* coverage.

    Note for non-negative features (TF-IDF): the inner product is already in [0, 1],
    so S floors at 0.5 and normalized coverage can never drop below 0.5. That is a
    property of the embedding, not of the theory; report raw cosine alongside.
    """
    phi_x = np.asarray(phi_x, dtype=float)
    phi_z = np.asarray(phi_z, dtype=float)
    return (1.0 + phi_x @ phi_z.T) / 2.0


def best_match(S):
    """For each target row, the (index, value) of its most similar calibration item.

    Ties break on the lowest index, deterministically, which is what Section 7's
    "fixed input-based tie-breaking" requires: the assignment must be a function of
    the inputs alone, never of the correctness outcomes being analyzed."""
    S = np.asarray(S, dtype=float)
    assign = np.argmax(S, axis=1)                 # numpy argmax already takes first max
    return assign, S[np.arange(S.shape[0]), assign]


def coverage(S):
    """f_cov = sum_x max_z S(x, z), and its normalized form c = f_cov / N."""
    _, best = best_match(S)
    return float(best.sum()), float(best.mean())


def match_weights(assign, n_cal):
    """w_z = (fraction of target items whose nearest calibration item is z), length n_cal.

    Non-negative and sums to one by construction: every target item has exactly one
    match. Some selected calibration items get weight zero -- they were selected but
    never became anybody's nearest neighbour."""
    assign = np.asarray(assign, dtype=int)
    w = np.bincount(assign, minlength=n_cal).astype(float)
    return w / w.sum()


def n_eff(w):
    """Concentration ESS 1 / sum_z w_z^2 (Section 12.4).

    Bounded by 1 <= n_eff <= r <= min(n, N), where r is the number of positive
    weights. A thousand target items all matching one labeled item gives perfect
    coverage and n_eff = 1: still only one random correctness observation."""
    w = np.asarray(w, dtype=float)
    return float(1.0 / np.sum(w ** 2))


def tv_to_uniform(w):
    """Total-variation distance between uniform weights and matching weights,
    (1/2) sum_z |1/n - w_z| over the n distinct selected items (Section 16.1).

    This is the extra term the *current* uniform calibration average pays, and it is
    computable from the assignment alone -- no labels."""
    w = np.asarray(w, dtype=float)
    n = len(w)
    return float(0.5 * np.sum(np.abs(1.0 / n - w)))


def match_distances(S, assign=None):
    """Per-target embedding distance to its matched calibration item.

    Uses the Section 9 identity d_x^2 = 4 (1 - S(x, z_Q(x))) rather than recomputing
    from the raw vectors, so the returned distances are exactly the ones the coverage
    bound is about. Returns (d, mean_d, mean_d2)."""
    S = np.asarray(S, dtype=float)
    if assign is None:
        assign, _ = best_match(S)
    best = S[np.arange(S.shape[0]), np.asarray(assign, dtype=int)]
    d2 = 4.0 * (1.0 - best)
    d = np.sqrt(np.maximum(d2, 0.0))
    return d, float(d.mean()), float(d2.mean())


# --------------------------------------------------------------------------- #
#  Sections 11 / 16: the two calibration estimators                            #
# --------------------------------------------------------------------------- #
def uniform_estimate(Y):
    """The estimator the paper currently uses: pi_j = (1/n) sum_z Y_j(z).

    Y is [M, n] correctness of each model on each selected calibration item."""
    return np.asarray(Y, dtype=float).mean(axis=1)


def weighted_estimate(Y, w):
    """The estimator the theorem is about: theta_hat_j = sum_z w_z Y_j(z).

    Same labeled items, different weights. Reweighting costs nothing -- the labels
    are already paid for -- so the only question is whether the target-matched
    workload is closer to the truth than the uniform one."""
    return np.asarray(Y, dtype=float) @ np.asarray(w, dtype=float)


# --------------------------------------------------------------------------- #
#  Sections 10-13: the bound                                                   #
# --------------------------------------------------------------------------- #
def transfer_term(L, c=None, mean_d=None):
    """Deterministic transfer error 2 L sqrt(1 - c), or the sharper L * mean(d_x).

    Section 10 derives the sharp form first and then relaxes mean(d) <= sqrt(mean d^2)
    to reach the coverage form. Both are valid; the sharp one is never looser, and it
    is just as computable, so both are reported."""
    if mean_d is not None:
        return float(L * mean_d)
    return float(2.0 * L * np.sqrt(max(0.0, 1.0 - c)))


def hoeffding_term(m, delta=0.05):
    """sqrt(log(2/delta) / (2 m)) -- Section 12.3 with m effective observations."""
    return float(np.sqrt(np.log(2.0 / delta) / (2.0 * m)))


def bound_weighted(L, c, m, delta=0.05, mean_d=None):
    """B_j = 2 L sqrt(1-c) + sqrt(log(2/delta) / (2 n_eff))  (Section 13)."""
    return transfer_term(L, c, mean_d) + hoeffding_term(m, delta)


def bound_uniform(L, c, n, tv, delta=0.05, mean_d=None):
    """Three-term bound for the CURRENT uniform estimator (Section 16.2).

    The Hoeffding term uses n, the plain count of distinct selected items, because
    uniform weights are 1/n each; the weight-mismatch term tv is the price of using
    the wrong workload."""
    return transfer_term(L, c, mean_d) + float(tv) + hoeffding_term(n, delta)


def bound_realized(L, c, m, N, delta=0.05, mean_d=None):
    """Bound against REALIZED target accuracy rather than expected (Section 17).

    Adds one more Hoeffding term for the target's own outcome randomness and splits
    the failure budget in half by the union bound, so both logs become log(4/delta)."""
    return (transfer_term(L, c, mean_d)
            + hoeffding_term(m, delta / 2.0) + hoeffding_term(N, delta / 2.0))


# --------------------------------------------------------------------------- #
#  Section 19: when is the uncapped bound actually useful?                     #
# --------------------------------------------------------------------------- #
def required_neff(L, c, delta=0.05, eps=1.0, mean_d=None):
    """Smallest n_eff making B <= eps, or inf when the transfer term alone exceeds it.

    Section 19.13: B <= eps iff A < eps AND m >= log(2/delta) / (2 (eps - A)^2)."""
    A = transfer_term(L, c, mean_d)
    if A >= eps:
        return float("inf")
    return float(np.log(2.0 / delta) / (2.0 * (eps - A) ** 2))


def required_coverage(L, m, delta=0.05, eps=1.0):
    """Smallest normalized coverage making B <= eps at a given n_eff (Section 19.10)."""
    H = hoeffding_term(m, delta)
    if H > eps:
        return float("nan")               # no coverage can rescue this sample size
    if L <= 0:
        return 0.0
    return float(max(0.0, 1.0 - (eps - H) ** 2 / (4.0 * L ** 2)))


def required_lipschitz(err, c, m, delta=0.05, mean_d=None):
    """Invert the bound: the smallest L for which B >= a measured error `err`.

    This is the honest diagnostic when L is unknown. If the answer is far above any
    plausible smoothness of expected correctness, the bound held with room to spare;
    if it is small, the bound is only barely doing work."""
    slack = err - hoeffding_term(m, delta)
    if slack <= 0:
        return 0.0
    denom = mean_d if mean_d is not None else 2.0 * np.sqrt(max(1e-12, 1.0 - c))
    return float(slack / denom)


def bound_in_range(L, c, m, delta=0.05, eps=1.0, mean_d=None):
    """Section 19.8's exact necessary-and-sufficient test, as a dict."""
    A = transfer_term(L, c, mean_d)
    need = required_neff(L, c, delta, eps, mean_d)
    return dict(A=A, transfer_ok=bool(A < eps), n_eff_required=need,
                n_eff_ok=bool(m >= need), B=A + hoeffding_term(m, delta),
                ok=bool(A < eps and m >= need))


# --------------------------------------------------------------------------- #
#  Section 6: greedy dataset selection                                         #
# --------------------------------------------------------------------------- #
def alpha_K(K):
    """1 - (1 - 1/K)^K, the exact finite-budget greedy factor. alpha_1 = 1,
    alpha_2 = 0.75, decreasing to 1 - 1/e ~ 0.632."""
    K = int(K)
    return 1.0 if K <= 1 else float(1.0 - (1.0 - 1.0 / K) ** K)


def greedy_cover(B, K):
    """Greedily pick <= K datasets maximizing f_cov(Q) = sum_x max_{r in Q} B[r, x].

    B is [R, N]: B[r, x] is dataset r's best match for target item x. Ties break on
    the lowest dataset index. Returns (chosen, f_cov_trace)."""
    B = np.asarray(B, dtype=float)
    R, N = B.shape
    cover = np.zeros(N)
    chosen, trace = [], []
    for _ in range(min(int(K), R)):
        gain = np.maximum(0.0, B - cover[None, :]).sum(axis=1)
        gain[chosen] = -np.inf
        r = int(np.argmax(gain))
        chosen.append(r)
        cover = np.maximum(cover, B[r])
        trace.append(float(cover.sum()))
    return chosen, trace


def optimal_cover(B, K):
    """Brute-force optimum over all C(R, K) collections. Only for small R."""
    from itertools import combinations
    B = np.asarray(B, dtype=float)
    R = B.shape[0]
    best, arg = -np.inf, None
    for comb in combinations(range(R), int(K)):
        v = float(B[list(comb)].max(axis=0).sum())
        if v > best:
            best, arg = v, comb
    return list(arg), best


# --------------------------------------------------------------------------- #
#  MTM08: curvature-matched prior ESS                                          #
# --------------------------------------------------------------------------- #
def beta_curvature(a, b, theta):
    """D(theta) = -d^2/dtheta^2 log Beta(a, b) density = (a-1)/theta^2 + (b-1)/(1-theta)^2."""
    return float((a - 1.0) / theta ** 2 + (b - 1.0) / (1.0 - theta) ** 2)


def ess_curvature_delta(a, b, m):
    """MTM08's distance delta(m) between prior curvature and m-observation posterior
    curvature, in the c -> inf epsilon-prior limit.

    Computed the long way -- build the epsilon prior, update it with m expected
    observations, difference the curvatures -- so that the closed form
    |S (S - m) (1/a + 1/b)| is *verified* rather than assumed."""
    S = a + b
    tbar = a / S
    Dp = beta_curvature(a, b, tbar)
    # posterior of the epsilon prior after m observations, E[Y] = m * tbar
    Dq = ((m * tbar - 1.0) / tbar ** 2
          + (m * (1.0 - tbar) - 1.0) / (1.0 - tbar) ** 2)
    return abs(Dp - Dq)


def beta_ess(a, b):
    """ESS(Beta(a, b)) = a + b (Part A)."""
    return float(a + b)


def power_prior_beta(a0, n0, pi, a00=1.0, b00=1.0):
    """Beta parameters of the Ibrahim-Chen power prior (Part B, Step 4).

    Historical data: pi * n0 successes in n0 trials, discounted by exponent a0, on
    top of an initial Beta(a00, b00). Returns (alpha, beta, s) with s = a0 * n0 the
    *added* evidence strength -- the quantity PoolEval calls the prior strength."""
    s = float(a0) * float(n0)
    return 1.0 * a00 + s * pi, 1.0 * b00 + s * (1.0 - pi), s


def prior_ess(s, a00=1.0, b00=1.0):
    """Total MTM08 ESS of the anchor: s + (a00 + b00), i.e. s + 2 by default.

    The paper calls s the effective sample size. The MTM08 ESS is s + 2; the extra 2
    is the uniform Beta(1,1) baseline, which contributes curvature but contributes
    *zero* exponents to the MAP objective (which is why :func:`map_weight` divides by
    N + s and not N + s + 2)."""
    return float(s) + float(a00) + float(b00)


def map_weight(s, N):
    """r = s / (N + s): how much of the MAP accuracy comes from the prior."""
    return float(s) / (float(N) + float(s))


def map_update(mu_hat, pi, s, N):
    """alpha_new = (N mu_hat + s pi) / (N + s), the convex combination of Section 7."""
    r = map_weight(s, N)
    return (1.0 - r) * np.asarray(mu_hat, dtype=float) + r * np.asarray(pi, dtype=float)


# --------------------------------------------------------------------------- #
#  Lipschitz proxy                                                             #
# --------------------------------------------------------------------------- #
def estimate_lipschitz(gram, Y, k=8, quantile=0.95):
    """kNN-smoothed proxy for the smoothness constant L_j of expected correctness.

    The assumption in Section 8.1 is about the *expectation* p_j(v), not the binary
    outcome Y_j(v), so a raw finite-difference over 0/1 labels is meaningless (two
    identical embeddings with different outcomes give L = inf). We therefore smooth
    each model's correctness over its k nearest calibration neighbours to get a crude
    p_hat_j, then take a high quantile of |p_hat_j(u) - p_hat_j(v)| / ||phi(u)-phi(v)||
    over all calibration pairs.

    `gram` is the [n, n] Gram matrix of unit-norm calibration embeddings; only inner
    products are ever needed, so callers never have to materialize the (huge, sparse)
    feature vectors densely.

    This is a proxy on the calibration set only. It cannot certify the assumption
    across the target/calibration gap, which is exactly what Section 8.1 warns about.
    """
    G = np.asarray(gram, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = G.shape[0]
    D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * G))
    order = np.argsort(-G, axis=1)[:, :max(1, min(k, n))]
    P = np.stack([Y[:, idx].mean(axis=1) for idx in order], axis=1)   # [M, n]
    iu = np.triu_indices(n, k=1)
    d = D[iu]
    keep = d > 1e-6
    out = []
    for j in range(Y.shape[0]):
        r = np.abs(P[j][iu[0]] - P[j][iu[1]])[keep] / d[keep]
        out.append(float(np.quantile(r, quantile)) if r.size else 0.0)
    return np.array(out)
