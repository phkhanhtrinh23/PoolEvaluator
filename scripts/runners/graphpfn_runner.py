"""GraphPFN-1.3 runner: in-context node classification with the ``graphpfn`` package.

Run in GraphPFN's environment (``pip install graphpfn``; PyTorch < 2.5 with DGL).  The
labeled source training nodes are the in-context examples.  When a requested graph is
the source graph itself, its training nodes are the context; otherwise the requested
graph is appended to the source graph as a disjoint component so that its nodes are
predicted with the same context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner_common import load_graph, load_job, parse_args, write_output  # noqa: E402


def checkpoint_uri(checkpoint: str | None) -> str:
    if checkpoint and Path(checkpoint).is_dir():
        files = sorted(Path(checkpoint).glob("*.pt"))
        if files:
            return str(files[0])
    if checkpoint and Path(checkpoint).is_file():
        return checkpoint
    return "hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt"


def main() -> None:
    args = parse_args(__doc__, [("--d-pca", {"type": int, "default": 64})])
    import dgl
    from graphpfn import GraphDataset, predict_icl

    dgl.random.seed(args.seed)
    job = load_job(args.job)
    classes = len(job["class_names"])
    task_type = "multiclass" if classes > 2 else "binclass"
    source_x, source_edges = load_graph(job["train"])
    train_nodes = torch.as_tensor(job["train"]["nodes"], dtype=torch.long)
    train_labels = np.asarray(job["train"]["labels"], dtype=np.float32)
    checkpoint = checkpoint_uri(job.get("checkpoint"))
    output: dict[str, list[int]] = {}
    for entry in job["graphs"]:
        if entry.get("same_as_train"):
            x, edges = source_x, source_edges
            offset = 0
        else:
            graph_x, graph_edges = load_graph(entry)
            x = torch.cat([source_x, graph_x])
            edges = torch.cat([source_edges, graph_edges + source_x.shape[0]], dim=1)
            offset = source_x.shape[0]
        n = x.shape[0]
        targets = np.full(n, np.nan, dtype=np.float32)
        targets[train_nodes.numpy()] = train_labels
        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_nodes.numpy()] = True
        test_mask = np.zeros(n, dtype=bool)
        requested = np.asarray(entry["nodes"], dtype=np.int64) + offset
        test_mask[requested] = True
        dataset = GraphDataset(
            name=f"pooleval-{entry['key']}",
            graph=dgl.graph((edges[0], edges[1]), num_nodes=n),
            features={"other": x.numpy().astype(np.float32)},
            targets=targets,
            masks={"train": train_mask, "val": train_mask.copy(), "test": test_mask},
            task_type=task_type,
        )
        preprocessing = {"d_pca": args.d_pca} if x.shape[1] > args.d_pca else {}
        predictions = predict_icl(
            dataset=dataset,
            model_kwargs={"checkpoint": checkpoint},
            preprocessing_kwargs=preprocessing,
            device=args.device,
        )
        scores = np.asarray(predictions)[requested]
        labels = scores.argmax(-1) if scores.ndim == 2 else (scores > 0.5).astype(int)
        output[entry["key"]] = [int(v) for v in labels]
        print(f"[graphpfn] {entry['key']}: {len(requested)} nodes", file=sys.stderr)
    write_output(args.output, output)


if __name__ == "__main__":
    main()
