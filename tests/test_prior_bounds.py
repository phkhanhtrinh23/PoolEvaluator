"""Tests for coverage, accuracy-bound, and prior-ESS diagnostics."""
import itertools

import numpy as np

from pooleval.prior_bounds import (
    beta_effective_sample_size,
    beta_power_prior_diagnostics,
    bound_feasibility,
    greedy_dataset_coverage,
    matching_diagnostics,
    optimal_dataset_coverage,
    prior_accuracy_diagnostics,
    shifted_cosine_similarity,
)


def test_greedy_coverage_satisfies_finite_budget_guarantee():
    similarity = np.array([
        [1.0, 0.2, 0.8, 0.1, 0.7, 0.3],
        [0.1, 0.9, 0.4, 0.8, 0.2, 0.7],
        [0.6, 0.5, 1.0, 0.2, 0.1, 0.9],
        [0.2, 0.3, 0.4, 1.0, 0.8, 0.1],
    ])
    dataset_ids = np.repeat(np.arange(3), 2)
    greedy = greedy_dataset_coverage(similarity, dataset_ids, budget=2)
    optimum = optimal_dataset_coverage(similarity, dataset_ids, budget=2)
    alpha_two = 1.0 - (1.0 - 1.0 / 2.0) ** 2

    assert greedy["coverage"] >= alpha_two * optimum["coverage"]
    assert len(greedy["selected_datasets"]) == 2
    assert len(greedy["selected_items"]) == 4


def test_optimal_coverage_matches_manual_enumeration():
    rng = np.random.default_rng(7)
    similarity = rng.uniform(size=(8, 12))
    dataset_ids = np.repeat(np.arange(4), 3)
    result = optimal_dataset_coverage(similarity, dataset_ids, budget=2)
    manual = max(
        similarity[:, np.isin(dataset_ids, pair)].max(axis=1).sum()
        for pair in itertools.combinations(range(4), 2)
    )
    np.testing.assert_allclose(result["coverage"], manual)


def test_perfect_coverage_counterexample_needs_matching_weights():
    target = np.vstack([
        np.repeat([[1.0, 0.0]], 90, axis=0),
        np.repeat([[0.0, 1.0]], 10, axis=0),
    ])
    calibration = np.array([[1.0, 0.0], [0.0, 1.0]])
    similarity = shifted_cosine_similarity(target, calibration)
    outcomes = np.array([0.9, 0.1])
    diag = prior_accuracy_diagnostics(
        similarity, np.arange(2), outcomes,
        lipschitz=0.8 / np.sqrt(2.0), delta=0.05,
    )

    np.testing.assert_allclose(diag["weights"], [0.9, 0.1])
    np.testing.assert_allclose(diag["matched_accuracy"], [0.82])
    np.testing.assert_allclose(diag["uniform_accuracy"], [0.5])
    np.testing.assert_allclose(diag["weight_mismatch"], 0.4)
    np.testing.assert_allclose(diag["normalized_coverage"], 1.0)
    np.testing.assert_allclose(diag["effective_sample_size"], 1.0 / 0.82)


def test_matching_ess_is_distinct_from_beta_ess():
    similarity = np.eye(3)
    match = matching_diagnostics(similarity, np.arange(3))
    prior = beta_power_prior_diagnostics(
        prior_accuracy=0.3, calibration_count=10, discount=0.5,
        target_count=20,
    )

    np.testing.assert_allclose(match["effective_sample_size"], 3.0)
    np.testing.assert_allclose(prior["alpha"], 2.5)
    np.testing.assert_allclose(prior["beta"], 4.5)
    np.testing.assert_allclose(prior["added_strength"], 5.0)
    np.testing.assert_allclose(prior["beta_effective_sample_size"], 7.0)
    np.testing.assert_allclose(prior["map_prior_weight"], 0.2)


def test_beta_three_seven_has_effective_sample_size_ten():
    np.testing.assert_allclose(beta_effective_sample_size(3.0, 7.0), 10.0)


def test_bound_feasibility_reproduces_six_versus_five_example():
    six = bound_feasibility(0.96, 6.0, lipschitz=1.0, delta=0.05)
    five = bound_feasibility(0.96, 5.0, lipschitz=1.0, delta=0.05)

    assert six["feasible"]
    assert six["bound"] < 1.0
    assert not five["feasible"]
    assert five["bound"] > 1.0
    np.testing.assert_allclose(six["required_effective_sample_size"],
                               5.123443, rtol=1e-5)


def test_zero_embedding_is_rejected():
    with np.testing.assert_raises_regex(ValueError, "zero vector"):
        shifted_cosine_similarity(np.zeros((1, 2)), np.ones((1, 2)))
