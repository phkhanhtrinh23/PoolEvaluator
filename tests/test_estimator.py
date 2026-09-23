import numpy as np

from pooleval.estimator import PoolEvaluator, calibration_parameters, leave_one_out_agreement


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
