"""End-to-end PoolEvaluator run on a node-classification target (paper Appendix D).

The sixteen PyG architectures are trained on the labeled source nodes with the run
seed.  GraphGPT, GraphPFN, and the two LLaGA checkpoints only run inside their
official runtimes, so each is executed as a subprocess configured under
``external_runners`` in the config (README section 6):

    external_runners:
      GraphGPT-7B: {python: /envs/graphgpt/bin/python, script: /path/to/runner.py}

The runner is called as ``python script --job job.json --output predictions.json``.
``job.json`` holds the model's checkpoint paths, the class names, and one entry per
graph: ``{"key", "graph" (torch file with x and edge_index only -- never labels),
"nodes", "node_text" (JSON list or null)}``.  The runner writes
``{key: [class index or -1 per node]}``.  An answer of -1 (unmappable text) is an
invalid output: it agrees with nothing and is never a judge candidate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import load_model_pool, paper_config
from .core import (
    estimate_pool, judge_history, release, safe_name, subset_budget, task_settings,
    unique_failures, write_report,
)
from .encoders import propagated_features, rank_by_similarity
from .graph_data import DOMAINS, MetaGraph, NodeTask, load_node_task, make_meta_graphs
from .judges import JudgeEnsemble
from .metrics import evaluate
from .models import load_model
from .reproducibility import seed_everything


def _forward(model: Any, x: Any, edge_index: Any) -> Any:
    try:
        return model(x, edge_index)
    except TypeError:
        return model(x)


def train_member(spec: Any, task: NodeTask, train_cfg: dict[str, Any], seed: int, device: str) -> Any:
    """Fit one PyG architecture on the labeled source training nodes."""
    import torch

    torch.manual_seed(seed)
    loaded = load_model(
        spec,
        device_map=device,
        dtype="float32",
        in_channels=task.source.num_features,
        hidden_channels=int(train_cfg.get("hidden", 128)),
        out_channels=task.num_classes,
    )
    model = loaded.model
    x = task.source.x.to(device)
    edge_index = task.source.edge_index.to(device)
    y = task.source.y.to(device)
    train = torch.as_tensor(task.train_idx, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(train_cfg.get("lr", 0.01)), weight_decay=float(train_cfg.get("weight_decay", 5e-4))
    )
    model.train()
    for _ in range(int(train_cfg.get("epochs", 200))):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(_forward(model, x, edge_index)[train], y[train])
        loss.backward()
        optimizer.step()
    model.eval()
    return loaded


def predict_graph(loaded: Any, graph: Any, nodes: Sequence[int], device: str) -> list[int]:
    import torch

    with torch.inference_mode():
        logits = _forward(loaded.model, graph.x.to(device), graph.edge_index.to(device))
    return logits.argmax(-1)[torch.as_tensor(np.asarray(nodes), device=device)].cpu().tolist()


def run_external(
    spec: Any,
    runner: dict[str, Any],
    requests: dict[str, tuple[Any, np.ndarray, list[str] | None]],
    class_names: Sequence[str],
    checkpoint_dir: str | Path,
    job_dir: Path,
) -> dict[str, list[int]]:
    """Run an official-runtime pool member through the JSON job protocol."""
    import torch

    loaded = load_model(spec, cache_dir=checkpoint_dir)
    external = loaded.model
    job_dir.mkdir(parents=True, exist_ok=True)
    graphs = []
    for key, (graph, nodes, text) in requests.items():
        graph_path = job_dir / f"{key}.pt"
        torch.save({"x": graph.x.cpu(), "edge_index": graph.edge_index.cpu()}, graph_path)
        text_path = None
        if text is not None:
            text_path = job_dir / f"{key}_text.json"
            text_path.write_text(json.dumps(text), encoding="utf-8")
        graphs.append(
            {"key": key, "graph": str(graph_path), "nodes": [int(n) for n in nodes],
             "node_text": None if text_path is None else str(text_path)}
        )
    runtime = None
    if spec.runtime_repo:
        name = spec.runtime_repo.removesuffix(".git").rsplit("/", 1)[-1]
        runtime = str(Path(checkpoint_dir).expanduser().resolve() / "runtimes" / name)
    job = {
        "model": spec.name,
        "checkpoint": external.checkpoint,
        "base": external.base,
        "auxiliary": external.auxiliary,
        "runtime": runtime,
        "class_names": list(class_names),
        "graphs": graphs,
    }
    job_path = job_dir / "job.json"
    output_path = job_dir / "predictions.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    subprocess.run(
        [runner["python"], runner["script"], "--job", str(job_path), "--output", str(output_path)],
        check=True,
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))
    for key, (_, nodes, _) in requests.items():
        if len(output.get(key, [])) != len(nodes):
            raise ValueError(f"{spec.name} runner returned {len(output.get(key, []))} labels for {key}")
    return {key: [int(v) for v in output[key]] for key in requests}


def _neighbors(graph: Any) -> Any:
    import scipy.sparse as sp

    edges = graph.edge_index.cpu().numpy()
    n = graph.num_nodes
    return sp.csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n, n))


def run_node(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    config = paper_config(args.config)
    runtime_cfg = config.get("runtime", {})
    seed = int(args.seed if args.seed is not None else config["seed"])
    deterministic = bool(runtime_cfg.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)
    settings = task_settings(config, "node")
    k = subset_budget(args.subset_budget, settings)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.node_root or config["data"]["node_root"])
    task = load_node_task(args.dataset, root, settings, seed, source=args.source)

    n = min(int(args.target_items or settings.get("target_items", 1000)), len(task.target_idx))
    target_nodes = np.sort(np.random.default_rng([seed, 1]).choice(task.target_idx, n, replace=False))
    target_labels = task.target.y.numpy()[target_nodes]
    print(f"[target] {task.name}: {n} of {len(task.target_idx)} target nodes, {task.num_classes} classes")

    hops = int(settings.get("encoder_hops", 2))
    target_features = propagated_features(task.target.x, task.target.edge_index, hops)
    requests: dict[str, tuple[Any, np.ndarray, list[str] | None]] = {
        "target": (task.target, target_nodes, task.target_text)
    }
    chosen: list[MetaGraph] = []
    retrieval: list[tuple[Any, float]] = []
    if k > 0:
        metas = make_meta_graphs(task, settings.get("meta_subsets", 50), seed)
        embeddings = {
            m.index: propagated_features(m.graph.x, m.graph.edge_index, hops) for m in metas
        }
        retrieval = rank_by_similarity(target_features[target_nodes], embeddings)
        chosen = [metas[index] for index, _ in retrieval[:k]]
        for meta in chosen:
            text = None
            if task.source_text is not None:
                text = [task.source_text[i] for i in meta.source_nodes]
            requests[f"meta-{meta.index}"] = (meta.graph, np.arange(meta.graph.num_nodes), text)
        print(f"[retrieval] {k} of {len(metas)} meta-graphs: "
              + ", ".join(f"{m.index}:{m.method}@{m.rate:.2f}" for m in chosen))
    else:
        print("[retrieval] K=0: random truncated-normal initialization, no calibration subsets")

    artifacts = Path(args.artifact_root or config["data"]["output_root"]).resolve()
    cache_dir = artifacts / "predictions" / "node" / safe_name(task.name) / f"seed{seed}_n{n}"
    runners = config.get("external_runners") or {}
    specs = load_model_pool("node")[: args.max_models]
    used: list[Any] = []
    excluded: list[str] = []
    predictions: dict[str, dict[str, list[int]]] = {}
    for index, spec in enumerate(specs, start=1):
        cache = cache_dir / f"{safe_name(spec.name)}.json"
        cached: dict[str, list[int]] = json.loads(cache.read_text()) if cache.exists() else {}
        missing = {key: value for key, value in requests.items() if key not in cached}
        if spec.backend == "external" and missing and spec.name not in runners:
            if args.skip_external:
                excluded.append(spec.name)
                print(f"[predict] {index}/{len(specs)} {spec.name}: skipped (no external runner)")
                continue
            raise RuntimeError(
                f"{spec.name} needs its official runtime: set external_runners.{spec.name} in "
                "the config (README section 6) or pass --skip-external"
            )
        print(f"[predict] {index}/{len(specs)} {spec.name}")
        if missing:
            if spec.backend == "external":
                cached.update(run_external(
                    spec, runners[spec.name], missing, task.class_names, args.checkpoint_dir,
                    cache_dir / "jobs" / safe_name(spec.name),
                ))
            else:
                loaded = train_member(spec, task, settings.get("train", {}), seed, device)
                for key, (graph, nodes, _) in missing.items():
                    cached[key] = predict_graph(loaded, graph, nodes, device)
                release(loaded)
                del loaded
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(cached), encoding="utf-8")
        predictions[spec.name] = cached
        used.append(spec)

    def matrix(key: str) -> np.ndarray:
        return np.array([predictions[s.name][key] for s in used], dtype=int).T

    raw_target = matrix("target")
    target_valid = raw_target >= 0
    target_matrix = unique_failures(raw_target, ~target_valid)
    source_matrix = source_gold = None
    subset_ids: list[int] = []
    if chosen:
        raw_source = np.concatenate([matrix(f"meta-{m.index}") for m in chosen])
        source_matrix = unique_failures(raw_source, raw_source < 0)
        source_gold = np.concatenate([m.graph.y.numpy() for m in chosen]).tolist()
        subset_ids = [m.index for m in chosen for _ in range(m.graph.num_nodes)]

    judge = None
    gofa = None
    rounds = int(settings.get("judge_rounds", 0))
    if args.judge and rounds > 0:
        from .local_judges import NodeJudgeItem, load_node_judge

        if task.target_text is None:
            raise RuntimeError(
                f"GOFA judges text-attributed graphs; add node_text.json for {args.dataset} "
                "(README section 2) or run without --judge"
            )
        gofa = load_node_judge(config, args.checkpoint_dir)
        ensemble = JudgeEnsemble([gofa])
        adjacency = _neighbors(task.target)
        max_neighbors = int(config["judges"]["node"].get("max_neighbors", 10))

        def judge(item: int, candidates: list[Any], support: list[float]) -> Any:
            node = int(target_nodes[item])
            neighbors = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]][:max_neighbors]
            choice = ensemble.choose(
                NodeJudgeItem(
                    task.target_text[node],
                    tuple(task.target_text[int(v)] for v in neighbors),
                    tuple(task.class_names[c] for c in candidates),
                    DOMAINS.get(args.dataset, "graph"),
                ),
                support,
            )
            return candidates[choice] if choice >= 0 else f"__none__{item}"

    try:
        estimate, initialization = estimate_pool(
            target_matrix, config, seed, valid=target_valid,
            source=source_matrix, source_gold=source_gold, subset_ids=subset_ids,
            judge=judge, rounds=rounds,
        )
    finally:
        if gofa is not None:
            gofa.close()
    truth = (raw_target == target_labels[:, None]).mean(axis=0).astype(float)
    report = {
        "task": "node",
        "dataset": args.dataset,
        "graphs": task.name,
        "seed": seed,
        "deterministic": deterministic,
        "subset_budget": k,
        "initialization": initialization,
        "models": [spec.name for spec in used],
        "excluded_models": excluded,
        "target_items": n,
        "calibration_items": 0 if source_matrix is None else int(source_matrix.shape[0]),
        "retrieved_subsets": [
            {"subset": m.index, "method": m.method, "rate": m.rate, "similarity": score}
            for m, (_, score) in zip(chosen, retrieval)
        ],
        "estimated_accuracy": estimate.alpha.tolist(),
        "true_accuracy": truth.tolist(),
        "ranking": [used[i].name for i in estimate.ranking],
        "metrics": evaluate(estimate.alpha, truth),
        "iterations": estimate.iterations,
        "converged": estimate.converged,
        "validated_items": sorted(estimate.validated),
        "judge_history": judge_history(estimate),
        "notes": task.notes,
    }
    suffix = "" if k == settings.get("subset_budget", k) else f"_k{k}"
    write_report(artifacts / f"node_{safe_name(args.dataset)}{suffix}_report.json", report)
    return report
