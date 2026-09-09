"""Verify the coverage/ESS derivations against the numbers stated in their proofs.

Every assertion here re-derives a claim from
``pdfs/coverage_hoeffding_derivation.md`` or ``pdfs/MTM08_analysis.md`` numerically,
so that a change to `pooleval/theory.py` that breaks the mathematics fails loudly.
"""
import numpy as np
import pytest

from pooleval import theory as th


# --------------------------------------------------------------------------- #
#  MTM08: Beta ESS by curvature matching                                       #
# --------------------------------------------------------------------------- #
def test_beta_3_7_has_ess_10_and_reproduces_the_paper_table():
    """MTM08_analysis.md Part A Step 7: delta(m) = 4.762 |10 - m|, zero at m = 10."""
    a, b = 3.0, 7.0
    assert th.beta_curvature(a, b, 0.3) == pytest.approx(34.4671, abs=1e-3)
    closed = lambda m: abs((a + b) * (a + b - m) * (1 / a + 1 / b))   # noqa: E731
    for m, expect in [(0, 47.619), (5, 23.810), (10, 0.0), (15, 23.810)]:
        assert th.ess_curvature_delta(a, b, m) == pytest.approx(expect, abs=1e-3)
        assert th.ess_curvature_delta(a, b, m) == pytest.approx(closed(m), abs=1e-9)
    grid = np.linspace(0, 30, 3001)
    assert grid[np.argmin([th.ess_curvature_delta(a, b, m) for m in grid])] \
        == pytest.approx(10.0, abs=1e-2)
    assert th.beta_ess(a, b) == 10.0


def test_ess_is_the_parameter_sum_for_arbitrary_betas():
    for a, b in [(1.0, 1.0), (2.5, 0.5), (12.0, 30.0), (0.7, 0.3)]:
        m = a + b
        assert th.ess_curvature_delta(a, b, m) == pytest.approx(0.0, abs=1e-8)


def test_power_prior_ess_is_10_a0_plus_2():
    """Part B: Beta(1,1) + 3-of-10 historical, discounted by a0."""
    for a0 in [0.0, 0.25, 0.5, 0.75, 1.0]:
        alpha, beta, s = th.power_prior_beta(a0, n0=10, pi=0.3)
        assert alpha == pytest.approx(1 + 3 * a0)
        assert beta == pytest.approx(1 + 7 * a0)
        assert th.beta_ess(alpha, beta) == pytest.approx(10 * a0 + 2)
        assert th.prior_ess(s) == pytest.approx(10 * a0 + 2)


def test_our_prior_has_ess_s_plus_2_and_mode_at_pi():
    """Part C: Beta(1 + s pi, 1 + s (1-pi)) -> ESS s+2, mode pi, mean != pi."""
    s, pi = 120.0, 0.42
    alpha, beta, s_out = th.power_prior_beta(a0=1.0, n0=s, pi=pi)
    assert s_out == pytest.approx(s)
    assert th.beta_ess(alpha, beta) == pytest.approx(s + 2)
    mode = (alpha - 1) / (alpha + beta - 2)
    mean = alpha / (alpha + beta)
    assert mode == pytest.approx(pi)
    assert mean != pytest.approx(pi)          # section 4.3: mean is NOT the mode


def test_map_update_is_the_stated_convex_combination():
    """Section 7: alpha_new = (T + s pi)/(N + s) = (1-r) mu_hat + r pi, r = s/(N+s)."""
    rng = np.random.default_rng(0)
    N, M = 400, 12
    T = rng.integers(0, N, size=M).astype(float)
    pi = rng.uniform(0.2, 0.8, size=M)
    for s in [0.0, 30.0, 400.0, 4000.0]:
        direct = (T + s * pi) / (N + s)
        assert np.allclose(th.map_update(T / N, pi, s, N), direct)
        assert th.map_weight(s, N) == pytest.approx(s / (N + s))
    assert th.map_weight(400.0, 400) == pytest.approx(0.5)   # s = N -> equal weight


# --------------------------------------------------------------------------- #
#  Coverage: geometry, monotonicity, submodularity, greedy                     #
# --------------------------------------------------------------------------- #
def _random_unit(n, d, rng):
    X = rng.normal(size=(n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_shifted_cosine_gives_the_exact_distance_identity():
    """Section 9: ||phi(x) - phi(z)||^2 = 4 (1 - S(x, z))."""
    rng = np.random.default_rng(1)
    X, Z = _random_unit(40, 16, rng), _random_unit(25, 16, rng)
    S = th.shifted_cosine(X, Z)
    assert S.min() >= -1e-12 and S.max() <= 1 + 1e-12
    D2 = ((X[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    assert np.allclose(D2, 4.0 * (1.0 - S))
    _, mean_d, mean_d2 = th.match_distances(S)
    _, c = th.coverage(S)
    assert mean_d2 == pytest.approx(4.0 * (1.0 - c))
    assert mean_d <= np.sqrt(mean_d2) + 1e-12          # Jensen, section 9


def test_coverage_is_monotone_and_submodular():
    rng = np.random.default_rng(2)
    R, N = 9, 60
    B = rng.uniform(0, 1, size=(R, N))
    f = lambda Q: float(B[list(Q)].max(axis=0).sum()) if Q else 0.0   # noqa: E731
    for _ in range(200):
        A = set(np.flatnonzero(rng.random(R) < 0.4).tolist())
        Bset = A | set(np.flatnonzero(rng.random(R) < 0.4).tolist())
        assert f(A) <= f(Bset) + 1e-12                              # monotone
        rest = [r for r in range(R) if r not in Bset]
        if rest:
            r = int(rng.choice(rest))
            gain_a = f(A | {r}) - f(A)
            gain_b = f(Bset | {r}) - f(Bset)
            assert gain_a >= gain_b - 1e-12                         # submodular


def test_greedy_beats_alpha_K_times_the_brute_force_optimum():
    """Section 6: f(greedy) >= alpha_K f(optimum), alpha_2 = 0.75, alpha_K >= 1-1/e."""
    assert th.alpha_K(1) == pytest.approx(1.0)
    assert th.alpha_K(2) == pytest.approx(0.75)
    assert th.alpha_K(1000) > 1 - 1 / np.e
    rng = np.random.default_rng(3)
    for trial in range(25):
        B = rng.uniform(0, 1, size=(10, 40)) ** rng.uniform(0.5, 4)
        for K in (2, 3):
            g, trace = th.greedy_cover(B, K)
            _, opt = th.optimal_cover(B, K)
            assert trace[-1] >= th.alpha_K(K) * opt - 1e-9, (trial, K)
            assert trace[-1] <= opt + 1e-9


# --------------------------------------------------------------------------- #
#  Weights, the two ESS, and the Hoeffding bound                               #
# --------------------------------------------------------------------------- #
def test_weights_sum_to_one_and_neff_obeys_its_bounds():
    """Section 19.15: 1 <= n_eff <= r <= min(n, N), r = number of positive weights."""
    rng = np.random.default_rng(4)
    for _ in range(50):
        N, n = int(rng.integers(5, 200)), int(rng.integers(2, 60))
        assign = rng.integers(0, n, size=N)
        w = th.match_weights(assign, n)
        assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
        r = int((w > 0).sum())
        m = th.n_eff(w)
        assert 1 - 1e-9 <= m <= r + 1e-9 <= min(n, N) + 1e-9
    assert th.n_eff(np.full(10, 0.1)) == pytest.approx(10.0)      # uniform -> n
    e = np.zeros(10); e[0] = 1.0
    assert th.n_eff(e) == pytest.approx(1.0)                      # degenerate -> 1


def test_weighted_hoeffding_actually_covers_at_the_stated_rate():
    """Section 12: P(|theta_hat - theta_tilde| >= eps) <= 2 exp(-2 eps^2 / sum w^2)."""
    rng = np.random.default_rng(5)
    n, trials, delta = 40, 20000, 0.05
    w = rng.dirichlet(np.full(n, 0.7))
    p = rng.uniform(0.1, 0.9, size=n)
    eps = th.hoeffding_term(th.n_eff(w), delta)
    Y = (rng.random((trials, n)) < p[None, :]).astype(float)
    err = np.abs(Y @ w - p @ w)
    assert (err >= eps).mean() <= delta       # the bound holds, with slack


def test_section_16_3_counterexample_reproduces_exactly():
    """Perfect coverage, valid smoothness, and uniform averaging still off by 0.32."""
    u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    X = np.vstack([np.tile(u, (90, 1)), np.tile(v, (10, 1))])
    Z = np.vstack([u, v])
    S = th.shifted_cosine(X, Z)
    f_cov, c = th.coverage(S)
    assert f_cov == pytest.approx(100.0) and c == pytest.approx(1.0)  # perfect

    assign, _ = th.best_match(S)
    w = th.match_weights(assign, 2)
    assert np.allclose(w, [0.9, 0.1])

    p = np.array([[0.9, 0.1]])                        # expected correctness at u, v
    theta = 0.9 * 0.9 + 0.1 * 0.1
    assert theta == pytest.approx(0.82)
    assert th.weighted_estimate(p, w)[0] == pytest.approx(0.82)   # matched: exact
    assert th.uniform_estimate(p)[0] == pytest.approx(0.50)       # uniform: 0.32 off
    assert abs(0.50 - theta) == pytest.approx(0.32)
    assert th.tv_to_uniform(w) == pytest.approx(0.4)              # and 0.4 >= 0.32
    assert th.tv_to_uniform(w) >= abs(0.50 - theta)


# --------------------------------------------------------------------------- #
#  Section 19: the feasibility algebra                                         #
# --------------------------------------------------------------------------- #
def test_section_15_and_19_numerical_examples():
    # Section 15: L = 0.2, c = 0.96, n_eff = 100, delta = 0.05
    assert th.transfer_term(0.2, 0.96) == pytest.approx(0.08)
    assert th.hoeffding_term(100, 0.05) == pytest.approx(0.1358, abs=1e-4)
    assert th.bound_weighted(0.2, 0.96, 100, 0.05) == pytest.approx(0.2158, abs=1e-4)
    # the 1-1/e substitution is LOOSER, not tighter
    worse = th.transfer_term(0.2, (1 - 1 / np.e) * 0.99)
    assert worse == pytest.approx(0.2447, abs=1e-4)
    assert worse + th.hoeffding_term(100, 0.05) == pytest.approx(0.3805, abs=1e-3)

    # Section 19.12: L = 1, c = 0.96 -> A = 0.4, threshold between m = 5 and m = 6
    assert th.transfer_term(1.0, 0.96) == pytest.approx(0.4)
    assert th.required_neff(1.0, 0.96, 0.05, eps=1.0) == pytest.approx(5.1234, abs=1e-3)
    assert th.bound_weighted(1.0, 0.96, 6, 0.05) == pytest.approx(0.9544, abs=1e-3)
    assert th.bound_weighted(1.0, 0.96, 5, 0.05) == pytest.approx(1.0074, abs=1e-3)
    assert th.bound_in_range(1.0, 0.96, 6)["ok"] and not th.bound_in_range(1.0, 0.96, 5)["ok"]

    # Section 19.13: L = 0.2, c = 0.96, target eps = 0.2 -> m >= 128.09
    assert th.required_neff(0.2, 0.96, 0.05, eps=0.2) == pytest.approx(128.0861, abs=1e-3)
    assert th.bound_weighted(0.2, 0.96, 129, 0.05) == pytest.approx(0.1996, abs=1e-3)

    # Section 19.2: perfect coverage cannot save n_eff = 1
    assert th.bound_weighted(0.2, 1.0, 1, 0.05) == pytest.approx(1.3581, abs=1e-3)


def test_required_coverage_inverts_required_neff():
    for L, m, eps in [(1.0, 20.0, 1.0), (0.5, 200.0, 0.5), (2.0, 50.0, 0.9)]:
        c = th.required_coverage(L, m, 0.05, eps)
        if not np.isnan(c):
            assert th.bound_weighted(L, c, m, 0.05) == pytest.approx(eps, abs=1e-9)
    assert np.isnan(th.required_coverage(1.0, 0.1, 0.05, eps=0.5))   # H > eps


def test_required_lipschitz_is_the_exact_inverse_of_the_bound():
    # only invertible when the measured error exceeds the sampling term alone
    for err, c, m in [(0.30, 0.9, 50.0), (0.30, 0.7, 200.0)]:
        L = th.required_lipschitz(err, c, m)
        assert th.bound_weighted(L, c, m) == pytest.approx(err, abs=1e-9)
    assert th.required_lipschitz(0.001, 0.9, 3.0) == 0.0   # sampling term alone exceeds
