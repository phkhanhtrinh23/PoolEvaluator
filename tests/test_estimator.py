import numpy as np

from pooleval.estimator import (
    PoolEvaluator,
    calibration_parameters,
    leave_one_out_agreement,
    random_initialization,
    truncated_normal,
)


def test_leave_one_out_does_not_vote_for_itself():
    responses = [["a", "a", "b"], ["x", "y", "y"]]
    agreement, consensus = leave_one_out_agreement(responses, [0.9, 0.8, 0.7])
    assert consensus[0].tolist() == ["a", "a", "a"]
    assert agreement[0].tolist() == [1.0, 1.0, 0.0]
    assert consensus[1].tolist() == ["y", "x", "x"]


def test_calibrated_em_returns_probabilities_and_ranking():
    source = [[0, 0, 1], [1, 1, 0], [0, 2, 0], [2, 2, 2], [0, 0, 1]]
    initial = calibration_parameters(source, [0, 1, 0, 2, 0])
    target = [[0, 0, 1], [1, 1, 0], [0, 2, 0], [2, 2, 2]]
    result = PoolEvaluator(max_iterations=200).fit(target, initial)
    assert result.alpha.shape == (3,)
    assert np.all((result.alpha > 0) & (result.alpha < 1))
    assert sorted(result.ranking.tolist()) == [0, 1, 2]
    assert result.iterations <= 200


def test_exact_validation_pins_all_models_on_item():
    responses = [[0, 0, 1], [1, 2, 2], [0, 0, 0]]
    initial = ([0.7, 0.6, 0.5], [0.8, 0.8, 0.8], [0.2, 0.2, 0.2])
    result = PoolEvaluator().fit(responses, initial, validated={1: 2})
    assert result.posterior[1].tolist() == [0.0, 1.0, 1.0]


def test_information_gain_scores_only_unvalidated_items():
    responses = [[0, 0, 1], [1, 2, 2], [0, 0, 0]]
    initial = ([0.7, 0.6, 0.5], [0.8, 0.8, 0.8], [0.2, 0.2, 0.2])
    evaluator = PoolEvaluator(max_iterations=30)
    estimate = evaluator.fit(responses, initial, validated={2: 0})
    gains = evaluator.information_gain(responses, estimate, {2: 0})
    assert np.isfinite(gains[:2]).all()
    assert gains[2] == -np.inf


def test_calibration_alpha_averages_subsets_equally():
    responses = [[0, 1], [0, 1], [0, 1], [1, 1]]
    gold = [0, 0, 0, 0]
    alpha, _, _ = calibration_parameters(responses, gold, subset_ids=["large", "large", "large", "small"])
    assert np.allclose(alpha, [0.5, 1e-6])


def test_random_initialization_is_truncated_normal_and_seeded():
    draws = truncated_normal(20000, 0.5, 0.25, np.random.default_rng(0))
    assert np.all((draws > 0.0) & (draws < 1.0))
    assert abs(draws.mean() - 0.5) < 0.01
    # Truncating N(0.5, 0.25^2) to (0, 1) removes ~4.6% of mass and shrinks the std to ~0.221.
    assert abs(draws.std() - 0.221) < 0.01
    first = random_initialization(5, 0.5, 0.25, seed=3)
    second = random_initialization(5, 0.5, 0.25, seed=3)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_fit_without_prior_uses_configured_truncated_normal():
    responses = [[0, 0, 1], [1, 1, 0], [0, 2, 0], [2, 2, 2]]
    evaluator = PoolEvaluator(max_iterations=1, seed=7, init_mean=0.3, init_std=0.1)
    direct = PoolEvaluator(max_iterations=1).fit(responses, random_initialization(3, 0.3, 0.1, 7))
    assert np.allclose(evaluator.fit(responses).alpha, direct.alpha)


def test_invalid_outputs_are_not_judge_candidates():
    # Item 0: models 2 and 3 failed (distinct failure ids); item 1: every output failed.
    responses = [[0, 0, -1, -2], [-3, -4, -5, -6], [1, 1, 0, 1], [2, 0, 2, 2]]
    valid = np.array(responses) >= 0
    evaluator = PoolEvaluator(max_iterations=30)
    initial = ([0.7, 0.6, 0.5, 0.4], [0.8] * 4, [0.2] * 4)
    estimate = evaluator.fit(responses, initial)
    gains = evaluator.information_gain(responses, estimate, {}, valid=valid)
    assert gains[1] == -np.inf
    seen = []

    def judge(item, candidates, support):
        seen.append((item, candidates, support))
        return candidates[0]

    evaluator.refine(responses, initial, judge, rounds=4, valid=valid)
    assert seen and all(item != 1 for item, _, _ in seen)
    for item, candidates, support in seen:
        assert all(candidate >= 0 for candidate in candidates)
        assert len(support) == len(candidates) and all(value > 0 for value in support)


def test_refine_passes_accuracy_weighted_support():
    responses = [[0, 0, 1], [1, 2, 2], [0, 0, 0], [1, 1, 0]]
    evaluator = PoolEvaluator(max_iterations=30)
    initial = ([0.7, 0.6, 0.5], [0.8, 0.8, 0.8], [0.2, 0.2, 0.2])
    calls = []

    def judge(item, candidates, support):
        calls.append((item, candidates, support))
        return candidates[0]

    evaluator.refine(responses, initial, judge, rounds=1)
    item, candidates, support = calls[0]
    before = evaluator.fit(responses, initial)
    row = np.asarray(responses[item])
    assert support == [float(before.alpha[row == c].sum()) for c in candidates]
