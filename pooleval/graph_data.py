"""Node-classification sources, targets, and meta-graphs (paper Appendix D).

Targets and the labeled nodes used to train the pool:

* ACMv9, Citationv1, DBLPv7 (GNNEvaluator): the pool trains on a different source
  graph (``tasks.node.sources`` in the config) and predicts every node of the target
  graph.  The GNNEvaluator raw files (``<name>_docs.txt``, ``<name>_edgelist.txt``,
  ``<name>_labels.txt``) are read when present, otherwise the ``.mat`` release of the
  same graphs is downloaded.
* ogbn-arxiv: train on the OGB train split (papers up to 2017) and evaluate on the test
  split (2019 onward).  Node text is the paper title and abstract.
* GOOD-Cora, GOOD-Twitch, GOOD-WebKB: train on the GOOD train split and evaluate on the
  out-of-distribution test split of the configured domain/shift.

Meta-graphs follow GNNEvaluator: the subgraph induced by held-out labeled source nodes
is augmented by EdgeDrop, node-feature masking, or subgraph sampling at a random rate.
Label names and node text come from ``class_names.json`` and ``node_text.json`` (a
list aligned with node indices) in the dataset directory; ``python -m
pooleval.node_metadata`` builds them from public raw releases.  When a
``node_meta.json`` fingerprint is present, the files are used only if the loaded graph
has the same node count and label sequence.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


GNNEVALUATOR = ("acmv9", "citationv1", "dblpv7")
GOOD_TARGETS = {"good-cora": "GOODCora", "good-twitch": "GOODTwitch", "good-webkb": "GOODWebKB"}
NODE_TARGETS = (*GNNEVALUATOR, "ogbn-arxiv", *GOOD_TARGETS)
ADAGCN_URL = "https://raw.githubusercontent.com/daiquanyu/AdaGCN_TKDE/main/input/{}.mat"
ARXIV_TEXT_URL = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
DOMAINS = {
    "acmv9": "citation network", "citationv1": "citation network", "dblpv7": "citation network",
    "ogbn-arxiv": "citation network of arXiv papers", "good-cora": "citation network",
    "good-twitch": "social network of Twitch users", "good-webkb": "network of university web pages",
}


@dataclass
class NodeTask:
    name: str
    source: Any  # torch_geometric Data with x, edge_index, y
    train_idx: np.ndarray
    meta_idx: np.ndarray
    target: Any
    target_idx: np.ndarray
    class_names: list[str]
    source_text: list[str] | None = None
    target_text: list[str] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _data(x: Any, edge_index: Any, y: Any) -> Any:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected

    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    return Data(
        x=torch.as_tensor(x, dtype=torch.float),
        edge_index=to_undirected(edge_index),
        y=torch.as_tensor(y, dtype=torch.long).view(-1),
    )


def _optional_json(directory: Path, name: str) -> Any:
    path = directory / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _node_files(
    directory: Path, graph: Any, target_idx: np.ndarray | None = None
) -> tuple[list[str] | None, list[str] | None, list[str]]:
    """Read ``node_text.json`` and ``class_names.json``, verified against ``node_meta.json``.

    ``class_names.json`` is a list, or per-domain lists ``{"by_domain": {domain: list},
    "domain_of_node": [...]}``, in which case the names of the domain holding
    ``target_idx`` are returned.
    """
    notes: list[str] = []
    meta = _optional_json(directory, "node_meta.json")
    if meta is not None:
        from .node_metadata import labels_sha1

        if meta["num_nodes"] != graph.num_nodes or meta["labels_sha1"] != labels_sha1(graph.y.cpu().numpy()):
            raise ValueError(
                f"{directory}: node_text.json/class_names.json were built for a different node order "
                "or label sequence; rebuild them with `python -m pooleval.node_metadata` or remove them"
            )
        notes += list(meta.get("notes", []))
    text = _optional_json(directory, "node_text.json")
    if text is not None and len(text) != graph.num_nodes:
        raise ValueError(f"{directory}/node_text.json has {len(text)} entries for {graph.num_nodes} nodes")
    names = _optional_json(directory, "class_names.json")
    if isinstance(names, dict):
        by_domain, domain_of_node = names["by_domain"], names["domain_of_node"]
        nodes = np.arange(graph.num_nodes) if target_idx is None else np.asarray(target_idx)
        domains = sorted({domain_of_node[int(i)] for i in nodes})
        lists = {tuple(by_domain[d]) for d in domains}
        if len(lists) != 1:
            raise ValueError(f"{directory}: the target nodes span domains {domains} with different class names")
        names = list(lists.pop())
    return text, names, notes


def _load_citation_graph(name: str, root: Path) -> tuple[Any, list[str] | None, list[str], list[str]]:
    directory = root / "gnnevaluator" / name
    notes: list[str] = []
    docs = next(iter(sorted(directory.rglob(f"{name}_docs.txt"))), None) if directory.exists() else None
    if docs is not None:
        raw = docs.parent
        x = np.loadtxt(raw / f"{name}_docs.txt", delimiter=",", dtype=np.float32)
        edges = np.loadtxt(raw / f"{name}_edgelist.txt", delimiter=",", dtype=np.int64).T
        y = np.loadtxt(raw / f"{name}_labels.txt", dtype=np.int64)
    else:
        import scipy.io as sio
        import scipy.sparse as sp

        path = directory / f"{name}.mat"
        if not path.exists():
            from urllib.request import urlretrieve

            directory.mkdir(parents=True, exist_ok=True)
            urlretrieve(ADAGCN_URL.format(name), path)
        mat = sio.loadmat(path)
        attributes = mat["attrb"]
        x = np.asarray(attributes.todense() if sp.issparse(attributes) else attributes, dtype=np.float32)
        groups = mat["group"]
        groups = np.asarray(groups.todense() if sp.issparse(groups) else groups)
        y = groups.argmax(1)
        edges = np.vstack(sp.coo_matrix(mat["network"]).nonzero())
    graph = _data(x, edges, y)
    text, names, file_notes = _node_files(directory, graph)
    names = names or [f"category {c}" for c in range(int(y.max()) + 1)]
    return graph, text, list(names), notes + file_notes


def _gnnevaluator_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """GNNEvaluator's 70/10/20 source split: train on 70%, meta-graphs from the rest."""
    order = np.random.default_rng([seed, 5]).permutation(n)
    cut = int(0.7 * n)
    return np.sort(order[:cut]), np.sort(order[cut:])


def _arxiv(root: Path) -> NodeTask:
    import torch

    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:
        raise RuntimeError("ogbn-arxiv requires `pip install ogb`") from exc
    original = torch.load
    torch.load = lambda *a, **k: original(*a, **{**k, "weights_only": False})  # OGB caches pickled Data
    try:
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root / "ogb"))
    finally:
        torch.load = original
    split = dataset.get_idx_split()
    data = dataset[0]
    graph = _data(data.x, data.edge_index, data.y)
    mapping = Path(dataset.root) / "mapping"
    with gzip.open(mapping / "labelidx2arxivcategeory.csv.gz", "rt") as handle:
        rows = list(csv.reader(handle))[1:]
    # "arxiv cs na" -> "cs.NA"
    names = [f"{parts[1]}.{parts[2].upper()}" for parts in (category.split() for _, category in rows)]
    text = _optional_json(root / "ogb", "node_text.json") or _arxiv_text(root / "ogb", mapping, graph.num_nodes)
    return NodeTask(
        "ogbn-arxiv", graph, split["train"].numpy(), split["valid"].numpy(),
        graph, split["test"].numpy(), names, text, text,
    )


def _arxiv_text(directory: Path, mapping: Path, n: int) -> list[str] | None:
    path = directory / "titleabs.tsv.gz"
    if not path.exists():
        try:
            from urllib.request import urlretrieve

            urlretrieve(ARXIV_TEXT_URL, path)
        except OSError:
            return None
    with gzip.open(mapping / "nodeidx2paperid.csv.gz", "rt") as handle:
        paper_ids = [row[1] for row in list(csv.reader(handle))[1:]]
    text: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                text[parts[0]] = f"Title: {parts[1]}. Abstract: {parts[2]}"
    return [text.get(paper_id, "") for paper_id in paper_ids[:n]]


def _good(name: str, root: Path, settings: Mapping[str, Any]) -> NodeTask:
    try:
        import GOOD.data.good_datasets as good_datasets
    except ImportError as exc:
        raise RuntimeError("GOOD targets require the GOOD package (github.com/divelab/GOOD)") from exc
    options = settings.get("good", {}).get(name, {})
    cls = getattr(good_datasets, GOOD_TARGETS[name])
    dataset, _ = cls.load(
        str(root / "good"), domain=options.get("domain"), shift=options.get("shift", "covariate"), generate=False
    )
    data = dataset[0] if hasattr(dataset, "__getitem__") else dataset.data
    graph = _data(data.x, data.edge_index, data.y)
    mask = lambda key: np.flatnonzero(getattr(data, key).cpu().numpy())  # noqa: E731
    directory = root / "good" / GOOD_TARGETS[name]
    text, names, notes = _node_files(directory, graph, mask("test_mask"))
    names = names or [f"category {c}" for c in range(int(graph.y.max()) + 1)]
    return NodeTask(
        name, graph, mask("train_mask"), mask("id_val_mask"), graph, mask("test_mask"),
        list(names), text, text, notes,
    )


def load_node_task(name: str, root: str | Path, settings: Mapping[str, Any], seed: int, source: str | None = None) -> NodeTask:
    root = Path(root)
    if name in GNNEVALUATOR:
        source_name = source or settings.get("sources", {}).get(name)
        if source_name not in GNNEVALUATOR or source_name == name:
            raise ValueError(f"{name} needs a different GNNEvaluator source graph, got {source_name!r}")
        source_graph, source_text, names, notes = _load_citation_graph(source_name, root)
        target_graph, target_text, target_names, target_notes = _load_citation_graph(name, root)
        if len(names) != len(target_names):
            raise ValueError(f"{source_name} and {name} do not share a label space")
        if source_graph.num_features != target_graph.num_features:
            raise ValueError(f"{source_name} and {name} do not share a feature space")
        train_idx, meta_idx = _gnnevaluator_split(source_graph.num_nodes, seed)
        return NodeTask(
            f"{source_name}->{name}", source_graph, train_idx, meta_idx, target_graph,
            np.arange(target_graph.num_nodes), names, source_text, target_text, notes + target_notes,
        )
    if name == "ogbn-arxiv":
        return _arxiv(root)
    if name in GOOD_TARGETS:
        return _good(name, root, settings)
    raise KeyError(f"unknown node target {name!r}")


@dataclass
class MetaGraph:
    """One labeled meta-graph P_r; every node is a calibration item."""

    index: int
    graph: Any
    source_nodes: np.ndarray  # source-graph index of each meta-graph node
    method: str
    rate: float


def make_meta_graphs(task: NodeTask, count: int, seed: int) -> list[MetaGraph]:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import subgraph

    held_out = torch.as_tensor(task.meta_idx, dtype=torch.long)
    base_edges, _ = subgraph(held_out, task.source.edge_index, relabel_nodes=True, num_nodes=task.source.num_nodes)
    base_x = task.source.x[held_out]
    base_y = task.source.y[held_out]
    methods = ("edge_drop", "node_fmask", "g_sample")
    graphs: list[MetaGraph] = []
    for r in range(int(count)):
        rng = np.random.default_rng([seed, r, 11])
        generator = torch.Generator().manual_seed(int(rng.integers(2**31)))
        method = methods[r % len(methods)]
        x, edges, keep = base_x, base_edges, np.arange(held_out.numel())
        if method == "edge_drop":
            rate = float(rng.uniform())
            edge_mask = torch.rand(edges.shape[1], generator=generator) >= rate
            edges = edges[:, edge_mask]
        elif method == "node_fmask":
            rate = float(rng.uniform())
            feature_mask = torch.rand(x.shape[1], generator=generator) < rate
            x = x.clone()
            x[:, feature_mask] = 0.0
        else:
            rate = float(rng.uniform(1e-5, 0.5))
            size = max(32, int(rate * held_out.numel()))
            keep = np.sort(rng.choice(held_out.numel(), size=min(size, held_out.numel()), replace=False))
            keep_t = torch.as_tensor(keep, dtype=torch.long)
            edges, _ = subgraph(keep_t, base_edges, relabel_nodes=True, num_nodes=held_out.numel())
            x = base_x[keep_t]
        graph = Data(x=x, edge_index=edges, y=base_y[torch.as_tensor(keep)])
        graphs.append(MetaGraph(r, graph, task.meta_idx[keep], method, rate))
    return graphs
