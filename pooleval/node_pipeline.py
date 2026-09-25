"""End-to-end PoolEvaluator run on a node-classification target (paper Appendix D).

The sixteen PyG architectures are trained on the labeled source nodes, each with three
seeds.  GraphGPT, GraphPFN, and the two LLaGA checkpoints run inside their official
runtimes through ``scripts/runners/``, each executed as a subprocess configured under
``external_runners`` in the config (README section 6):

    external_runners:
      GraphGPT-7B: {python: /envs/graphgpt/bin/python, script: scripts/runners/graphgpt_runner.py}

The runner is called as ``python script --job job.json --output predictions.json``.
``job.json`` holds the model's checkpoint paths, the class names, a domain description,
the labeled source training graph (``train``: graph file, node ids, their labels), and
one entry per requested graph: ``{"key", "graph" (torch file with x and edge_index
only), "nodes", "node_text" (JSON list or null), "same_as_train"}``.  Labels of
requested nodes are never written.  The runner writes ``{key: [class index or -1 per
node]}``; -1 (unmappable text) is an invalid output that agrees with nothing and is
never a judge candidate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import load_model_pool, paper_config
from .core import (
    estimate_pool, judge_rounds, release, repeat_settings, safe_name, subset_budget, task_settings,
    unique_failures, write_report,
)
from .encoders import propagated_features, rank_by_similarity
from .graph_data import DOMAINS, MetaGraph, NodeTask, load_node_task, make_meta_graphs
from .judges import CachedJudge, JudgeEnsemble
from .metrics import evaluate
from .models import load_model
from .repeats import run_repeated
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
    task: NodeTask,
    checkpoint_dir: str | Path,
    job_dir: Path,
    domain: str = "graph",
) -> dict[str, list[int]]:
    """Run an official-runtime pool member through the JSON job protocol."""
    import torch

    from .config import ROOT

    loaded = load_model(spec, cache_dir=checkpoint_dir)
    external = loaded.model
    job_dir.mkdir(parents=True, exist_ok=True)
    train_path = job_dir / "train.pt"
    torch.save({"x": task.source.x.cpu(), "edge_index": task.source.edge_index.cpu()}, train_path)
    train_text_path = None
    if task.source_text is not None:
        train_text_path = job_dir / "train_text.json"
        train_text_path.write_text(json.dumps(task.source_text), encoding="utf-8")
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
             "node_text": None if text_path is None else str(text_path),
             "same_as_train": graph is task.source}
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
        "class_names": list(task.class_names),
        "domain": domain,
        "train": {
            "graph": str(train_path),
            "nodes": [int(n) for n in task.train_idx],
            "labels": [int(v) for v in task.source.y[torch.as_tensor(task.train_idx)].tolist()],
            "node_text": None if train_text_path is None else str(train_text_path),
        },
        "graphs": graphs,
    }
    job_path = job_dir / "job.json"
    output_path = job_dir / "predictions.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    script = Path(runner["script"])
    if not script.is_absolute():
        script = ROOT / script
    subprocess.run(
        [runner["python"], str(script), "--job", str(job_path), "--output", str(output_path),
         *[str(a) for a in runner.get("args", [])]],
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
    rounds = judge_rounds(args.judge_rounds, settings)
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
    retrieval_start = time.perf_counter()
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
    retrieval_seconds = time.perf_counter() - retrieval_start

    artifacts = Path(args.artifact_root or config["data"]["output_root"]).resolve()
    cache_dir = artifacts / "predictions" / "node" / safe_name(task.name) / f"seed{seed}_n{n}"
    runners = config.get("external_runners") or {}
    specs = load_model_pool("node")[: args.max_models]
    # Each architecture is trained with ``train_seeds`` seeds; the first is the run seed,
    # which makes up the main pool.  Checkpoint-based members have a single variant.
    train_seeds = [seed + offset for offset in range(max(1, int(settings.get("train_seeds", 3))))]
    used: list[Any] = []
    excluded: list[str] = []
    columns: list[dict[str, Any]] = []  # one entry per (member, seed) variant
    slots: list[list[int]] = []
    for index, spec in enumerate(specs, start=1):
        variants = [None] if spec.backend == "external" else train_seeds
        slot: list[int] = []
        for variant, train_seed in enumerate(variants):
            suffix = "" if variant == 0 else f"__seed{variant}"
            cache = cache_dir / f"{safe_name(spec.name)}{suffix}.json"
            cached: dict[str, Any] = json.loads(cache.read_text()) if cache.exists() else {}
            timing: dict[str, float] = cached.setdefault("_seconds", {})
            missing = {key: value for key, value in requests.items() if key not in cached}
            if spec.backend == "external" and missing and spec.name not in runners:
                if args.skip_external:
                    excluded.append(spec.name)
                    print(f"[predict] {index}/{len(specs)} {spec.name}: skipped (no external runner)")
                    break
                raise RuntimeError(
                    f"{spec.name} needs its official runtime: set external_runners.{spec.name} in "
                    "the config (README section 6) or pass --skip-external"
                )
            label = spec.name if train_seed is None else f"{spec.name} (seed {train_seed})"
            print(f"[predict] {index}/{len(specs)} {label}")
            if missing:
                start = time.perf_counter()
                if spec.backend == "external":
                    cached.update(run_external(
                        spec, runners[spec.name], missing, task, args.checkpoint_dir,
                        cache_dir / "jobs" / safe_name(spec.name), DOMAINS.get(args.dataset, "graph"),
                    ))
                    elapsed = time.perf_counter() - start
                    for key in missing:
                        timing[key] = elapsed * len(missing[key][1]) / sum(len(v[1]) for v in missing.values())
                else:
                    loaded = train_member(spec, task, settings.get("train", {}), train_seed, device)
                    timing.setdefault("train", time.perf_counter() - start)
                    for key, (graph, nodes, _) in missing.items():
                        begin = time.perf_counter()
                        cached[key] = predict_graph(loaded, graph, nodes, device)
                        timing[key] = time.perf_counter() - begin
                    release(loaded)
                    del loaded
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(cached), encoding="utf-8")
            slot.append(len(columns))
            columns.append({"name": spec.name, "seed": train_seed, "predictions": cached})
        if slot:
            slots.append(slot)
            used.append(spec)

    def matrix(key: str, cols: list[int]) -> np.ndarray:
        return np.array([columns[c]["predictions"][key] for c in cols], dtype=int).T

    wide = list(range(len(columns)))
    main_cols = [slot[0] for slot in slots]
    raw_wide = matrix("target", wide)
    valid_wide = raw_wide >= 0
    target_wide = unique_failures(raw_wide, ~valid_wide)
    correct_wide = raw_wide == target_labels[:, None]
    source_wide = source_gold = None
    subset_ids: list[int] = []
    if chosen:
        raw_source = np.concatenate([matrix(f"meta-{m.index}", wide) for m in chosen])
        source_wide = unique_failures(raw_source, raw_source < 0)
        source_gold = np.concatenate([m.graph.y.numpy() for m in chosen]).tolist()
        subset_ids = [m.index for m in chosen for _ in range(m.graph.num_nodes)]

    judge = None
    gofa = None
    if args.judge and rounds > 0:
        from .local_judges import NodeJudgeItem, load_node_judge

        if task.target_text is None:
            raise RuntimeError(f"GOFA judges text-attributed graphs; add node_text.json for {args.dataset}")
        gofa = load_node_judge(config, args.checkpoint_dir)
        adjacency = _neighbors(task.target)
        max_neighbors = int(config["judges"]["node"].get("max_neighbors", 10))

        def make_item(item: int, candidates: list[Any]) -> Any:
            node = int(target_nodes[item])
            neighbors = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]][:max_neighbors]
            return NodeJudgeItem(
                task.target_text[node],
                tuple(task.target_text[int(v)] for v in neighbors),
                tuple(task.class_names[c] for c in candidates),
                DOMAINS.get(args.dataset, "graph"),
            )

        judge = CachedJudge(
            JudgeEnsemble([gofa]), make_item, f"node:{task.name}:{cache_dir.name}",
            artifacts / "judge_cache" / f"node_{safe_name(args.dataset)}.json",
        )

    requested = list(requests)
    column_cost: list[float | None] = []
    for column in columns:
        timing = column["predictions"].get("_seconds", {})
        keys = requested if column["seed"] is None else ["train", *requested]
        column_cost.append(float(sum(timing[key] for key in keys)) if all(key in timing for key in keys) else None)
    try:
        run = estimate_pool(
            target_wide[:, main_cols], config, seed, valid=valid_wide[:, main_cols],
            source=None if source_wide is None else source_wide[:, main_cols],
            source_gold=source_gold, subset_ids=subset_ids, judge=judge, rounds=rounds,
        )
        repeated = None
        if args.repeats:
            sizes, repeats = repeat_settings(args, config)
            repeated = run_repeated(
                target_wide, config, seed,
                slot_names=[spec.name for spec in used], slots=slots,
                column_names=[c["name"] if c["seed"] is None else f"{c['name']}@seed{c['seed']}" for c in columns],
                correct=correct_wide, valid=valid_wide,
                source=source_wide, source_gold=source_gold, subset_ids=subset_ids,
                judge=judge, rounds=rounds, sizes=sizes, repeats=repeats,
                column_seconds=column_cost, fixed_seconds=retrieval_seconds,
            )
    finally:
        if gofa is not None:
            gofa.close()
    estimate = run.estimate
    truth = correct_wide[:, main_cols].mean(axis=0).astype(float)
    main_cost = [column_cost[c] for c in main_cols]
    method = sum(run.seconds[key] for key in ("stage1_prior", "stage2_em", "stage3_select_em", "judge"))
    latency = {
        "retrieval": retrieval_seconds,
        "model_training_inference": None if None in main_cost else float(sum(main_cost)),
        **{key: float(value) for key, value in run.seconds.items()},
    }
    latency["total"] = None if None in main_cost else retrieval_seconds + float(sum(main_cost)) + method
    report = {
        "task": "node",
        "dataset": args.dataset,
        "graphs": task.name,
        "seed": seed,
        "train_seeds": train_seeds,
        "deterministic": deterministic,
        "subset_budget": k,
        "judge_rounds_budget": rounds,
        "initialization": run.initialization,
        "models": [spec.name for spec in used],
        "excluded_models": excluded,
        "target_items": n,
        "calibration_items": 0 if source_wide is None else int(source_wide.shape[0]),
        "retrieved_subsets": [
            {"subset": m.index, "method": m.method, "rate": m.rate, "similarity": score}
            for m, (_, score) in zip(chosen, retrieval)
        ],
        "estimated_accuracy": estimate.alpha.tolist(),
        "true_accuracy": truth.tolist(),
        "true_accuracy_by_seed": {
            spec.name: [float(correct_wide[:, c].mean()) for c in slot] for spec, slot in zip(used, slots)
        },
        "ranking": [used[i].name for i in estimate.ranking],
        "metrics": evaluate(estimate.alpha, truth),
        "iterations": estimate.iterations,
        "converged": estimate.converged,
        "validated_items": sorted(estimate.validated),
        "judge_rounds": run.round_table(truth),
        "latency_seconds": latency,
    }
    if repeated is not None:
        report["repeated"] = repeated
    suffix = "" if k == settings.get("subset_budget", k) else f"_k{k}"
    if rounds != settings.get("judge_rounds", rounds):
        suffix += f"_v{rounds}"
    write_report(artifacts / f"node_{safe_name(args.dataset)}{suffix}_report.json", report)
    return report
