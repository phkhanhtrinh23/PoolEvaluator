from pooleval.pipeline import run_smoke


def test_pipeline_smoke(capsys):
    report = run_smoke()
    assert len(report["ranking"]) == 4
    assert len(report["estimated_accuracy"]) == 4
    capsys.readouterr()


def test_zero_subset_budget_uses_random_prior():
    import numpy as np

    from pooleval.config import paper_config
    from pooleval.core import estimate_pool, judge_rounds, subset_budget, unique_failures

    config = paper_config()
    assert subset_budget(0, config["tasks"]["text2sql"]) == 0
    assert subset_budget(None, config["tasks"]["image"]) == 10
    assert judge_rounds(None, config["tasks"]["node"]) == 5 and judge_rounds(3, config["tasks"]["node"]) == 3
    target = np.array([[0, 0, 1], [1, 1, 0], [0, 2, 0], [2, 2, 2]], dtype=object)
    run = estimate_pool(target, config, seed=1)
    assert run.initialization == "random-truncated-normal"
    assert run.estimate.alpha.shape == (3,)
    source = np.array([[0, 0, 1], [1, 1, 0]], dtype=object)
    run = estimate_pool(target, config, seed=1, source=source, source_gold=[0, 1], subset_ids=["a", "b"])
    assert run.initialization == "retrieved-subsets"
    labels = np.array([[0, -1, -1], [1, 1, -1]])
    remapped = unique_failures(labels, labels < 0)
    failures = remapped[labels < 0].tolist()
    assert len(set(failures)) == len(failures) and all(value < 0 for value in failures)


def test_judge_rounds_are_logged_with_warm_and_cold_iterations():
    import numpy as np

    from pooleval.config import paper_config
    from pooleval.core import estimate_pool

    rng = np.random.default_rng(0)
    gold = rng.integers(0, 3, 40)
    correct = rng.random((40, 4)) < np.array([0.8, 0.6, 0.5, 0.3])
    target = np.where(correct, gold[:, None], (gold[:, None] + 1) % 3).astype(object)
    oracle = lambda item, candidates, support: gold[item] if gold[item] in candidates else f"__none__{item}"  # noqa: E731
    run = estimate_pool(target, paper_config(), seed=3, judge=oracle, rounds=2)
    table = run.round_table(correct.mean(axis=0))
    assert table[0]["round"] == 0 and "mae" in table[0]
    assert [row["round"] for row in table[1:]] == list(range(1, len(table)))
    for row in table[1:]:
        assert {"iterations_warm", "iterations_cold", "iteration_reduction", "mae", "judge_seconds"} <= set(row)
    assert {"stage1_prior", "stage2_em", "judge"} <= set(run.seconds)


def test_repeated_runs_sample_pools_and_report_confidence_intervals():
    import numpy as np

    from pooleval.config import paper_config
    from pooleval.repeats import run_repeated, summarize

    rng = np.random.default_rng(1)
    gold = rng.integers(0, 3, 30)
    correct = rng.random((30, 6)) < np.linspace(0.3, 0.9, 6)
    target = np.where(correct, gold[:, None], (gold[:, None] + 1) % 3).astype(object)
    out = run_repeated(
        target, paper_config(), 7, slot_names=[f"m{j}" for j in range(6)], correct=correct,
        sizes=(3, 5, 9), repeats=3, column_seconds=[1.0] * 6, fixed_seconds=2.0,
    )
    assert out["protocol"]["pool_sizes"] == [3, 5] and out["protocol"]["skipped_pool_sizes"] == [9]
    assert out["overall"]["runs"] == 6 and out["by_pool_size"]["3"]["runs"] == 3
    mae = out["overall"]["metrics"]["mae"]
    assert mae["ci95_low"] <= mae["mean"] <= mae["ci95_high"]
    first = out["runs"][0]
    assert len(first["models"]) == 3 and first["latency_seconds"] >= 2.0 + 3.0
    again = run_repeated(
        target, paper_config(), 7, slot_names=[f"m{j}" for j in range(6)], correct=correct, sizes=(3,), repeats=1,
    )
    assert again["runs"][0]["models"] == first["models"]
    assert summarize([1.0])["n"] == 1 and summarize([])["n"] == 0


def test_repeated_runs_draw_one_variant_per_slot():
    import numpy as np

    from pooleval.config import paper_config
    from pooleval.repeats import run_repeated

    target = np.array([[0, 0, 1, 1, 0, 2], [1, 1, 1, 0, 1, 1], [2, 2, 0, 2, 2, 2]] * 4, dtype=object)
    out = run_repeated(
        target, paper_config(), 0, slot_names=["a", "b", "c"], slots=[[0, 1], [2, 3], [4, 5]],
        column_names=["a0", "a1", "b0", "b1", "c0", "c1"], sizes=(3,), repeats=4,
    )
    for run in out["runs"]:
        assert sorted(name[0] for name in run["models"]) == ["a", "b", "c"]
