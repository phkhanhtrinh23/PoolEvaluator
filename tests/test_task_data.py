import numpy as np
import pytest

from pooleval.encoders import HashedBagOfWords, propagated_features, rank_by_similarity


def test_rank_by_similarity_prefers_matching_subset():
    encoder = HashedBagOfWords()
    target = encoder.encode(["list all singers", "count the singers"])
    subsets = {"music": encoder.encode(["show every singer", "how many singers"]), "cars": encoder.encode(["car makers"])}
    assert rank_by_similarity(target, subsets)[0][0] == "music"


def test_propagated_features_are_normalized_and_parameter_free():
    x = np.eye(4)
    edges = np.array([[0, 1, 2], [1, 2, 3]])
    features = propagated_features(x, edges, hops=2)
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)
    assert np.allclose(features, propagated_features(x, edges, hops=2))


def test_image_meta_subsets_are_deterministic():
    pytest.importorskip("PIL")
    from PIL import Image

    from pooleval.image_data import ImageSet, SHIFTS, make_meta_subsets, subset_images

    images = [Image.fromarray((np.ones((8, 8, 3)) * i).astype(np.uint8)) for i in range(30)]
    source = ImageSet("toy", images.__getitem__, list(range(30)))
    subsets = make_meta_subsets(source, count=len(SHIFTS), size=5, seed=1)
    assert [s.shift for s in subsets] == list(SHIFTS)
    again = make_meta_subsets(source, count=len(SHIFTS), size=5, seed=1)
    assert subsets == again
    for subset in subsets:
        first = [np.asarray(img) for img in subset_images(source, subset)]
        second = [np.asarray(img) for img in subset_images(source, subset)]
        assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_meta_graphs_follow_gnnevaluator_augmentations():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from pooleval.graph_data import NodeTask, make_meta_graphs

    n = 60
    edges = torch.randint(0, n, (2, 200))
    graph = Data(x=torch.randn(n, 8), edge_index=edges, y=torch.randint(0, 3, (n,)))
    task = NodeTask("toy", graph, np.arange(40), np.arange(40, 60), graph, np.arange(n), ["a", "b", "c"])
    metas = make_meta_graphs(task, count=6, seed=0)
    assert [m.method for m in metas] == ["edge_drop", "node_fmask", "g_sample"] * 2
    for meta in metas:
        assert meta.graph.num_nodes == len(meta.source_nodes)
        assert set(meta.source_nodes.tolist()) <= set(range(40, 60))
        assert torch.equal(meta.graph.y, graph.y[torch.as_tensor(meta.source_nodes)])


def test_external_runner_protocol_hides_labels(tmp_path):
    import json
    import sys

    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from pooleval.config import load_model_pool
    from pooleval.node_pipeline import run_external

    runner = tmp_path / "runner.py"
    runner.write_text(
        "import argparse, json, torch\n"
        "p = argparse.ArgumentParser(); p.add_argument('--job'); p.add_argument('--output')\n"
        "a = p.parse_args(); job = json.load(open(a.job))\n"
        "out = {}\n"
        "for g in job['graphs']:\n"
        "    saved = torch.load(g['graph'])\n"
        "    assert set(saved) == {'x', 'edge_index'}\n"
        "    out[g['key']] = [len(job['class_names']) - 1] * len(g['nodes'])\n"
        "json.dump(out, open(a.output, 'w'))\n"
    )
    spec = next(s for s in load_model_pool("node") if s.backend == "external")
    graph = Data(x=torch.randn(5, 3), edge_index=torch.tensor([[0, 1], [1, 2]]), y=torch.tensor([0, 1, 1, 0, 1]))
    output = run_external(
        spec,
        {"python": sys.executable, "script": str(runner)},
        {"target": (graph, np.array([0, 3]), ["t0", "t1", "t2", "t3", "t4"])},
        ["a", "b"],
        tmp_path / "checkpoints",
        tmp_path / "job",
    )
    assert output == {"target": [1, 1]}
    job = json.loads((tmp_path / "job" / "job.json").read_text())
    assert job["model"] == spec.name and job["graphs"][0]["nodes"] == [0, 3]


def test_node_metadata_parsers():
    from pooleval.node_metadata import _cora_fields, _page_text

    extraction = (
        "URL: http://x/p.ps\nTitle: Learning to Plan  \nAuthor: A. B.\n"
        "Abstract: First line of the abstract\ncontinues here.\nReference: [1] ...\n"
    )
    fields = _cora_fields(extraction)
    assert fields["Title"] == "Learning to Plan"
    assert fields["Abstract"] == "First line of the abstract continues here."
    page = "MIME-Version: 1.0\nServer: CERN\n\n<html><title>CS 302</title><script>x=1</script><body>Home &amp; syllabus</body></html>"
    assert _page_text(page) == ("CS 302", "Home & syllabus")


def test_node_files_check_fingerprint_and_domain_names(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from pooleval.graph_data import _data, _node_files
    from pooleval.node_metadata import _write

    y = [0, 1, 0, 1]
    graph = _data(np.eye(4), np.array([[0, 1], [1, 2]]), y)
    names = {"by_domain": {"a": ["x", "y"], "b": ["y", "x"]}, "domain_of_node": ["a", "a", "b", "b"]}
    _write(tmp_path, y, "test", ["built for a test"], ["t0", "t1", "t2", "t3"], names)
    text, resolved, notes = _node_files(tmp_path, graph, np.array([2, 3]))
    assert text[2] == "t2" and resolved == ["y", "x"] and "built for a test" in notes
    with pytest.raises(ValueError, match="different classes"):
        _node_files(tmp_path, graph, np.array([0, 3]))
    shuffled = _data(np.eye(4), np.array([[0, 1], [1, 2]]), [1, 0, 0, 1])
    with pytest.raises(ValueError, match="different node order"):
        _node_files(tmp_path, shuffled)
