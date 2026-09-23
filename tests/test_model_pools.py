import pytest

from pooleval.config import load_model_pool


def test_appendix_model_counts_and_unique_ids():
    expected = {"text2sql": 35, "image": 20, "node": 20}
    for task, count in expected.items():
        specs = load_model_pool(task)
        assert len(specs) == count
        assert len({spec.name for spec in specs}) == count


def test_every_model_has_supported_backend():
    supported = {
        "transformers",
        "image_classification",
        "zero_shot_image",
        "timm",
        "pyg",
        "snapshot",
        "adapter",
        "external",
    }
    for task in ("text2sql", "image", "node"):
        assert {spec.backend for spec in load_model_pool(task)} <= supported


def test_all_pyg_architectures_forward_when_optional_runtime_is_present():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from pooleval.adapters import predict_nodes
    from pooleval.models import load_model

    graph = Data(
        x=torch.randn(6, 16),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2], [1, 2, 3, 4, 5, 0, 2, 0]]),
    )
    for spec in load_model_pool("node"):
        if spec.backend == "pyg":
            loaded = load_model(spec, device_map=None, in_channels=16, hidden_channels=8, out_channels=3)
            assert len(predict_nodes(loaded, graph)) == 6


def test_external_graph_checkpoints_use_pipeline_runner_hook():
    from types import SimpleNamespace

    from pooleval.adapters import predict_nodes
    from pooleval.models import load_model

    spec = next(spec for spec in load_model_pool("node") if spec.backend == "external")
    loaded = load_model(spec)
    graph = SimpleNamespace(external_runner=lambda checkpoint, graph, names: [1, 0])
    assert predict_nodes(loaded, graph, ["a", "b"]) == [1, 0]
