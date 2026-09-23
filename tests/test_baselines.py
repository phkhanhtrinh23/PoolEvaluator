import numpy as np

from baselines import (
    AgreementLine,
    AverageThresholdedConfidence,
    DawidSkene,
    DifferenceOfConfidence,
    GeneralizedDisagreementEquality,
    Independent,
    LLMJudge,
    Majority,
    baseline_registry,
)


SOURCE = np.array(
    [
        ["a", "a", "b"],
        ["b", "b", "b"],
        ["c", "x", "c"],
        ["d", "d", "y"],
    ],
    dtype=object,
)
GOLD = np.array(["a", "b", "c", "d"], dtype=object)


def test_response_baselines_return_model_accuracies():
    independent = Independent().evaluate(SOURCE, GOLD)
    majority = Majority().evaluate(SOURCE)
    dawid_skene = DawidSkene().evaluate(SOURCE)
    agreement_line = AgreementLine().evaluate(SOURCE, SOURCE, GOLD)
    judged = LLMJudge().evaluate(SOURCE, lambda item, candidates: GOLD[item])

    assert np.allclose(independent, [1.0, 0.75, 0.5])
    assert np.allclose(judged, independent)
    for estimate in (majority, dawid_skene, agreement_line):
        assert estimate.shape == (3,)
        assert np.all((0.0 <= estimate) & (estimate <= 1.0))
    assert dawid_skene[0] > dawid_skene[2]


def test_confidence_and_replica_baselines():
    source = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    target = np.array([[0.7, 0.3], [0.45, 0.55], [0.8, 0.2]])
    gold = np.array([0, 1, 0])

    estimates = [
        DifferenceOfConfidence().evaluate(source, target, gold),
        AverageThresholdedConfidence().evaluate(source, target, gold),
        AverageThresholdedConfidence("negative_entropy").evaluate(source, target, gold),
        GeneralizedDisagreementEquality().evaluate([0, 1, 1], [0, 0, 1]),
    ]
    assert all(0.0 <= estimate <= 1.0 for estimate in estimates)
    assert np.isclose(estimates[-1], 2 / 3)
    assert len(baseline_registry()) == 8
