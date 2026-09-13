"""Design-based accuracy estimators, and the acquisition criterion that targets the mean.

Unbiasedness is checked by Monte Carlo over the SAMPLING DESIGN with the finite
population held fixed -- which is what a design-based claim actually asserts.
"""
import numpy as np
import pytest

from pooleval.estimators import (horvitz_thompson, ht_variance,
                                 inverse_variance_fuse, model_assisted,
                                 model_assisted_variance)
from pooleval.validated_em import sampling_probabilities


# --------------------------------------------------------------------------- #
#  Horvitz-Thompson                                                            #
# --------------------------------------------------------------------------- #
def test_uniform_inclusion_reduces_ht_to_the_plain_sample_mean():
    """With pi_i = m/N, HT is (1/N) sum Z/(m/N) = (1/m) sum Z. Simple random sampling
    is a special case of the HT framework, not an alternative to it."""
    N, m = 50, 10
    values = np.array([[1.0, 0, 1, 1, 0, 1, 0, 0, 1, 1]])
    pi = np.full(m, m / N)
    assert horvitz_thompson(values, pi, N)[0] == pytest.approx(values.mean())


def test_ht_is_unbiased_over_the_sampling_design():
    rng = np.random.default_rng(0)
    N = 40
    Z = (rng.random((2, N)) < 0.6).astype(float)     # the fixed finite population
    truth = Z.mean(axis=1)
    m = 12
    pi = np.full(N, m / N)
    draws = []
    for _ in range(4000):
        idx = rng.choice(N, size=m, replace=False)
        draws.append(horvitz_thompson(Z[:, idx], pi[idx], N))
    assert np.allclose(np.mean(draws, axis=0), truth, atol=0.01)


def test_ht_is_unbiased_under_UNEQUAL_inclusion_probabilities():
    """The case that matters: an acquisition rule makes some items far likelier than
    others, and HT still recovers the population mean because it knows by how much."""
    rng = np.random.default_rng(1)
    N = 30
    Z = (rng.random((1, N)) < 0.5).astype(float)
    truth = Z.mean(axis=1)
    pi = rng.uniform(0.15, 0.9, size=N)              # Poisson design, independent draws
    draws = []
    for _ in range(20000):
        take = rng.random(N) < pi
        if not take.any():
            continue
        idx = np.flatnonzero(take)
        draws.append(horvitz_thompson(Z[:, idx], pi[idx], N))
    assert np.mean(draws, axis=0) == pytest.approx(truth, abs=0.01)


def test_ht_rejects_a_zero_inclusion_probability():
    """1/0 is not a weight. An item that could never be drawn is unrecoverable, which is
    exactly why a deterministic top-k selection cannot be HT-corrected afterwards."""
    with pytest.raises(ValueError):
        horvitz_thompson(np.ones((1, 2)), np.array([0.5, 0.0]), 10)


def test_ht_variance_is_positive_and_shrinks_as_inclusion_rises():
    values = np.ones((1, 6))
    high = ht_variance(values, np.full(6, 0.9), 20)[0]
    low = ht_variance(values, np.full(6, 0.2), 20)[0]
    assert 0 < high < low


# --------------------------------------------------------------------------- #
#  Model-assisted difference estimator                                         #
# --------------------------------------------------------------------------- #
def test_model_assisted_is_unbiased_even_when_the_model_is_badly_wrong():
    """The whole point: a bad model costs variance, never bias."""
    rng = np.random.default_rng(2)
    N, m = 40, 12
    Z = (rng.random((1, N)) < 0.7).astype(float)
    truth = Z.mean(axis=1)
    nonsense = np.full((1, N), 0.05)                 # model says 5%, truth is ~70%
    pi = np.full(N, m / N)
    draws = []
    for _ in range(4000):
        idx = np.sort(rng.choice(N, size=m, replace=False))
        draws.append(model_assisted(nonsense, Z[:, idx], idx, pi[idx], N))
    assert np.allclose(np.mean(draws, axis=0), truth, atol=0.01)


def test_a_perfect_model_makes_the_correction_vanish():
    rng = np.random.default_rng(3)
    N, m = 30, 8
    Z = (rng.random((1, N)) < 0.5).astype(float)
    idx = np.sort(rng.choice(N, size=m, replace=False))
    pi = np.full(m, m / N)
    assert model_assisted(Z, Z[:, idx], idx, pi, N)[0] == pytest.approx(Z.mean())
    assert model_assisted_variance(Z, Z[:, idx], idx, pi, N)[0] == pytest.approx(0.0)


def test_a_good_model_beats_plain_ht_on_variance():
    rng = np.random.default_rng(4)
    N, m = 60, 15
    p = rng.uniform(0.1, 0.9, size=(1, N))
    Z = (rng.random((1, N)) < p).astype(float)
    idx = np.sort(rng.choice(N, size=m, replace=False))
    pi = np.full(m, m / N)
    good = model_assisted_variance(p, Z[:, idx], idx, pi, N)[0]
    plain = ht_variance(Z[:, idx], pi, N)[0]
    assert good < plain


# --------------------------------------------------------------------------- #
#  Inverse-variance fusion                                                     #
# --------------------------------------------------------------------------- #
def test_inverse_variance_fusion_favours_the_sharper_estimate():
    est, var = inverse_variance_fuse([[0.8], [0.2]], [[0.001], [0.100]])
    assert est[0] > 0.75                              # pulled toward the sharp one
    assert var[0] < 0.001                             # and sharper than either


def test_fusing_two_equal_estimates_halves_the_variance():
    est, var = inverse_variance_fuse([[0.5], [0.5]], [[0.02], [0.02]])
    assert est[0] == pytest.approx(0.5)
    assert var[0] == pytest.approx(0.01)


def test_fusion_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        inverse_variance_fuse([[0.5]], [[0.1], [0.2]])


# --------------------------------------------------------------------------- #
#  Turning an acquisition score into a valid design                            #
# --------------------------------------------------------------------------- #
def test_sampling_probabilities_keep_every_item_reachable():
    """Positivity: the condition HT needs, and the one argmax destroys."""
    scores = {0: 10.0, 1: 0.0, 2: 0.0, 3: 0.0}
    p = sampling_probabilities(scores, [0, 1, 2, 3], epsilon=0.2)
    assert p.sum() == pytest.approx(1.0)
    assert p.min() > 0.0
    assert p[0] == max(p)                             # still prefers the best item


def test_epsilon_controls_how_far_the_design_is_from_uniform():
    scores = {0: 10.0, 1: 0.0, 2: 0.0, 3: 0.0}
    greedy = sampling_probabilities(scores, list(scores), epsilon=0.0)
    mixed = sampling_probabilities(scores, list(scores), epsilon=1.0)
    assert greedy[0] > mixed[0]
    assert np.allclose(mixed, 0.25)


def test_flat_scores_give_a_uniform_design():
    p = sampling_probabilities({i: 1.0 for i in range(5)}, list(range(5)))
    assert np.allclose(p, 0.2)
