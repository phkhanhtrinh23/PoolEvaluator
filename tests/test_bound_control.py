import numpy as np
import pytest

from experiments.run_bound_control import smooth_probabilities, violation_summary, mae_summary


def test_known_smoothness():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(20, 8))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    p = smooth_probabilities(x @ x[0], .5)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.abs(p[:, None] - p) <= .5 * np.linalg.norm(x[:, None] - x, axis=2) + 1e-12)
    np.testing.assert_allclose(smooth_probabilities(x @ x[0], 0), .5)
    with pytest.raises(ValueError):
        smooth_probabilities(x @ x[0], .6)


def test_zero_violations_not_zero_risk():
    summary = violation_summary(np.zeros(5000, dtype=bool))
    assert summary["count"] == 0
    assert summary["upper_95"] == pytest.approx(1 - .05 ** (1 / 5000))
    assert violation_summary([True, True])["upper_95"] == 1


def test_mae_uses_repetitions_as_units():
    summary = mae_summary(np.array([[.1, .3], [.3, .5]]))
    assert summary["mae"] == pytest.approx(30)
    assert summary["mc_se"] == pytest.approx(10)
