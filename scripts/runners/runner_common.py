"""Helpers shared by the external graph-model runners.

The runners execute inside each model's own environment, so this module depends only on
torch and numpy.  The job protocol is documented in ``pooleval.node_pipeline``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def parse_args(description: str, extra: Sequence[tuple[str, dict[str, Any]]] = ()) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    for flag, options in extra:
        parser.add_argument(flag, **options)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args


def load_job(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_graph(entry: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    stored = torch.load(entry["graph"], map_location="cpu")
    return stored["x"].float(), stored["edge_index"].long()


def load_text(entry: dict[str, Any]) -> list[str] | None:
    path = entry.get("node_text")
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def neighbor_lists(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Undirected neighbor lists without self loops or duplicates, in ascending order."""
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        if u != v:
            neighbors[u].add(v)
            neighbors[v].add(u)
    return [sorted(n) for n in neighbors]


def sample_subgraph(
    neighbors: list[list[int]], center: int, hops: int, per_hop: int, rng: random.Random
) -> tuple[list[int], torch.Tensor]:
    """Center-first k-hop sample (at most ``per_hop`` new neighbors per expanded node).

    Returns the node list (center at index 0) and the edges among those nodes,
    relabeled to list positions.
    """
    order = [center]
    seen = {center}
    frontier = [center]
    for _ in range(hops):
        following: list[int] = []
        for node in frontier:
            fresh = [n for n in neighbors[node] if n not in seen]
            if len(fresh) > per_hop:
                fresh = rng.sample(fresh, per_hop)
            for n in fresh:
                seen.add(n)
                order.append(n)
                following.append(n)
        frontier = following
    position = {node: i for i, node in enumerate(order)}
    edges = [
        (position[u], position[v]) for u in order for v in neighbors[u] if v in position
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)
    return order, edge_index


def parse_class(text: str, class_names: Sequence[str]) -> int:
    """The class whose name the answer mentions (longest name first), else -1."""
    lowered = (text or "").casefold()
    order = sorted(range(len(class_names)), key=lambda k: -len(class_names[k]))
    for k in order:
        if class_names[k].casefold() in lowered:
            return k
    return -1


def write_output(path: str | Path, predictions: dict[str, list[int]]) -> None:
    Path(path).write_text(json.dumps(predictions), encoding="utf-8")
