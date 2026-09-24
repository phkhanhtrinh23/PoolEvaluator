from pooleval.pipeline import run_smoke


def test_pipeline_smoke(capsys):
    report = run_smoke()
    assert len(report["ranking"]) == 4
    assert len(report["estimated_accuracy"]) == 4
    capsys.readouterr()


def test_zero_subset_budget_uses_random_prior():
    import numpy as np

    from pooleval.config import paper_config
    from pooleval.core import estimate_pool, subset_budget, unique_failures

    config = paper_config()
    assert subset_budget(0, config["tasks"]["text2sql"]) == 0
    assert subset_budget(None, config["tasks"]["image"]) == 10
    target = np.array([[0, 0, 1], [1, 1, 0], [0, 2, 0], [2, 2, 2]], dtype=object)
    estimate, initialization = estimate_pool(target, config, seed=1)
    assert initialization == "random-truncated-normal"
    assert estimate.alpha.shape == (3,)
    source = np.array([[0, 0, 1], [1, 1, 0]], dtype=object)
    _, initialization = estimate_pool(target, config, seed=1, source=source, source_gold=[0, 1], subset_ids=["a", "b"])
    assert initialization == "retrieved-subsets"
    labels = np.array([[0, -1, -1], [1, 1, -1]])
    remapped = unique_failures(labels, labels < 0)
    failures = remapped[labels < 0].tolist()
    assert len(set(failures)) == len(failures) and all(value < 0 for value in failures)
