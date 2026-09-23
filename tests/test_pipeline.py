from pooleval.pipeline import run_smoke


def test_pipeline_smoke(capsys):
    report = run_smoke()
    assert len(report["ranking"]) == 4
    assert len(report["estimated_accuracy"]) == 4
    capsys.readouterr()
