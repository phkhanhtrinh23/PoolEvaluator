import numpy as np

from zoo.new_formulation_real import evaluate, load_run, subset_run


def test_real_artifact_evaluation_runs_without_gold_leakage():
    run, _ = load_run("spider")
    original_prior = run.prior.copy()
    out = evaluate(subset_run(run, np.arange(12)))
    np.testing.assert_array_equal(run.prior, original_prior)
    assert out["new_acc"].shape == (run.M,)
    assert 0.0 <= out["pseudo_accuracy"] <= 1.0
    assert 0.0 < out["beta"] < 1.0
