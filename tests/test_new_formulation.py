"""Tests for the closed-form pseudo-label agreement formulation."""
import numpy as np

from pooleval import (Config, NewFormulationPoolEval, PoolEval, agreement_em,
                      simulate)


def test_one_iteration_matches_closed_form_equations():
    C = np.array([[1, 1, 0, 1], [0, 1, 0, 0]], dtype=float)
    alpha0 = np.array([0.7, 0.4])
    beta0 = 0.8

    logit_a = np.log(alpha0 / (1.0 - alpha0))[:, None]
    logit_b = np.log(beta0 / (1.0 - beta0))
    tau = 1.0 / (1.0 + np.exp(-(logit_a + (2.0 * C - 1.0) * logit_b)))
    expected_alpha = tau.mean(axis=1)
    expected_beta = np.mean(tau * C + (1.0 - tau) * (1.0 - C))

    out = agreement_em(C, alpha0, beta_init=beta0, max_iters=1)
    np.testing.assert_allclose(out["alpha"], expected_alpha)
    np.testing.assert_allclose(out["beta"], expected_beta)


def test_new_estimator_uses_old_argmax_as_pseudo_label():
    cfg = Config(seed=3, N=80)
    run = simulate(cfg)
    old = PoolEval(cfg).evaluate(run)
    new = NewFormulationPoolEval(cfg).evaluate(run, obs=old["obs"], pseudo_out=old)

    expected = np.asarray([max(p, key=p.get) for p in old["latent_post"]])
    np.testing.assert_array_equal(new["pseudo_label"], expected)
    np.testing.assert_array_equal(new["C"], old["obs"] == expected[None, :])
    assert new["acc"].shape == (cfg.M,)
    assert 0.0 < new["beta"] < 1.0


def test_em_observed_likelihood_is_monotone():
    rng = np.random.default_rng(9)
    C = rng.binomial(1, 0.65, size=(6, 100))
    out = agreement_em(C, np.linspace(0.4, 0.8, 6), beta_init=0.7)
    assert np.all(np.diff(out["log_likelihood"]) >= -1e-8)
