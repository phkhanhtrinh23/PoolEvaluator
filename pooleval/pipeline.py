"""Runnable paper pipeline: retrieve, generate, EX-Extended, EM, and judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adapters import predict_text2sql, text2sql_prompt
from .config import load_model_pool, paper_config
from .core import estimate_pool, judge_history, release, subset_budget, task_settings, write_report
from .data import Text2SQLItem, load_text2sql, schema_prompt
from .databases import build_instances
from .encoders import text_encoder
from .estimator import PoolEvaluator, calibration_parameters
from .execution import execute, response_classes, signature
from .judges import JudgeItem, paper_judges
from .metrics import evaluate
from .models import load_model
from .reproducibility import seed_everything
from .retrieval import select_subsets


def _cache_file(root: Path, dataset: str, split: str, model_name: str) -> Path:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in model_name)
    return root / "predictions" / dataset / split / f"{safe}.json"


def _generate(
    items: Sequence[Text2SQLItem],
    specs: Sequence[Any],
    split: str,
    artifact_root: Path,
    cache_dir: str | None,
    dtype: str,
) -> dict[str, dict[str, str]]:
    predictions: dict[str, dict[str, str]] = {}
    if not items:
        return predictions
    prompts = [text2sql_prompt(item.question, schema_prompt(item.schema), item.evidence) for item in items]
    for index, spec in enumerate(specs, start=1):
        path = _cache_file(artifact_root, items[0].dataset, split, spec.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        cached: dict[str, str] = {}
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
        missing = [i for i, item in enumerate(items) if item.id not in cached]
        print(f"[generate:{split}] {index}/{len(specs)} {spec.name}, missing={len(missing)}")
        if missing:
            loaded = load_model(spec, cache_dir=cache_dir, dtype=dtype)
            generated = predict_text2sql(loaded, [prompts[i] for i in missing])
            cached.update({items[i].id: sql for i, sql in zip(missing, generated)})
            path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
            release(loaded)
            del loaded
        predictions[spec.name] = cached
    return predictions


def _suites(items: Sequence[Text2SQLItem], root: Path) -> dict[Path, list[Path]]:
    suites: dict[Path, list[Path]] = {}
    for item in items:
        if item.db_path not in suites:
            records = build_instances(item.db_path, root / item.dataset)
            suites[item.db_path] = [item.db_path] + [Path(record.instance) for record in records]
    return suites


def _execute_pool(
    items: Sequence[Text2SQLItem],
    specs: Sequence[Any],
    predictions: dict[str, dict[str, str]],
    suites: dict[Path, list[Path]],
    timeout: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    """Return response classes, EX-Ext correctness, candidate validity, and gold classes.

    An output is a valid judge candidate only if it executes on the original database
    and all five modified instances: EX-Ext can never accept an output that fails on any
    of them, so such outputs are dropped from the judge's candidate set.
    """
    matrix = np.zeros((len(items), len(specs)), dtype=int)
    correctness = np.zeros((len(items), len(specs)), dtype=bool)
    valid = np.zeros((len(items), len(specs)), dtype=bool)
    gold_answers: list[int] = []
    for i, item in enumerate(items):
        paths = suites[item.db_path]
        gold = signature(item.gold_sql, paths, timeout)
        predicted = [signature(predictions[spec.name][item.id], paths, timeout) for spec in specs]
        classes = response_classes(predicted)
        matrix[i, :] = classes
        for j, answer in enumerate(predicted):
            correctness[i, j] = all(a is not None and a == b for a, b in zip(answer, gold))
            valid[i, j] = bool(answer) and all(part is not None for part in answer)
        matching = [classes[j] for j in range(len(specs)) if correctness[i, j]]
        gold_answers.append(matching[0] if matching else max(classes, default=-1) + 1)
        print(f"[execute] {i + 1}/{len(items)} {item.dataset}/{item.db_id}")
    return matrix, correctness, valid, gold_answers


def run_text2sql(args: argparse.Namespace) -> dict[str, Any]:
    config = paper_config(args.config)
    runtime_cfg = config.get("runtime", {})
    requested_seed = getattr(args, "seed", None)
    requested_dtype = getattr(args, "dtype", None)
    seed = int(requested_seed if requested_seed is not None else config["seed"])
    deterministic = bool(runtime_cfg.get("deterministic", True))
    dtype = requested_dtype or runtime_cfg.get("inference_dtype", "bfloat16")
    seed_everything(seed, deterministic=deterministic)
    settings = task_settings(config, "text2sql")
    k = subset_budget(getattr(args, "subset_budget", None), settings)
    data_cfg = config["data"]
    layout = {
        "text2sql_root": args.text2sql_root or data_cfg["text2sql_root"],
        "bird_metadata_root": args.bird_metadata_root or data_cfg["bird_metadata_root"],
        "bird_database_root": args.bird_database_root or data_cfg["bird_database_root"],
    }
    target = load_text2sql(args.dataset, "target", args.target_items, **layout)
    source: list[Text2SQLItem] = []
    retrieval: list[tuple[str, float]] = []
    if k > 0:
        source_pool = load_text2sql(args.dataset, "train", args.source_candidates, **layout)
        encoder = text_encoder(settings.get("encoder", "colbert-ir/colbertv2.0"), args.checkpoint_dir)
        source, retrieval = select_subsets(target, source_pool, k, encoder)
        release(encoder)
        del encoder
        if args.source_items:
            source = source[: args.source_items]
        print(f"[retrieval] selected {len(source)} items from {min(k, len(retrieval))} database subsets")
    else:
        print("[retrieval] K=0: random truncated-normal initialization, no calibration subsets")

    specs = load_model_pool("text2sql")[: args.max_models]
    artifacts = Path(args.artifact_root or data_cfg["output_root"]).resolve()
    predictions_target = _generate(
        target, specs, "target", artifacts, args.checkpoint_dir, dtype
    )
    predictions_source = _generate(
        source, specs, "source", artifacts, args.checkpoint_dir, dtype
    )
    db_root = artifacts / "db_instances"
    target_suites = _suites(target, db_root)
    timeout = float(config["execution"]["timeout_seconds"])
    target_matrix, target_correct, target_valid, _ = _execute_pool(
        target, specs, predictions_target, target_suites, timeout
    )
    source_matrix = source_gold = None
    if source:
        source_suites = _suites(source, db_root)
        source_matrix, _, _, source_gold = _execute_pool(
            source, specs, predictions_source, source_suites, timeout
        )

    judge = None
    if args.judge:
        judge_cfg = config["judges"]["text2sql"]
        ensemble = paper_judges(
            allow_partial=not args.require_all_judges and judge_cfg.get("allow_partial", True),
            members=judge_cfg.get("members"),
        )

        def judge(item_index: int, candidates: list[Any], support: list[float]) -> Any:
            item = target[item_index]
            rendered: list[str] = []
            for candidate in candidates:
                model_index = int(np.flatnonzero(target_matrix[item_index] == candidate)[0])
                sql = predictions_target[specs[model_index].name][item.id]
                result = execute(item.db_path, sql, timeout)
                rendered.append(f"SQL: {sql}\nOriginal-database result: {repr(result)[:1200]}")
            choice = ensemble.choose(
                JudgeItem(item.question, schema_prompt(item.schema), tuple(rendered), item.evidence),
                support,
            )
            return candidates[choice] if choice >= 0 else f"__none__{item_index}"

    estimate, initialization = estimate_pool(
        target_matrix,
        config,
        seed,
        valid=target_valid,
        source=source_matrix,
        source_gold=source_gold,
        subset_ids=[item.db_id for item in source],
        judge=judge,
        rounds=settings.get("judge_rounds", 0),
    )

    truth = target_correct.mean(axis=0)
    report = {
        "task": "text2sql",
        "dataset": args.dataset,
        "seed": seed,
        "deterministic": deterministic,
        "requested_inference_dtype": dtype,
        "subset_budget": k,
        "initialization": initialization,
        "models": [spec.name for spec in specs],
        "target_items": len(target),
        "calibration_items": len(source),
        "retrieved_subsets": retrieval[:k],
        "estimated_accuracy": estimate.alpha.tolist(),
        "true_ex_extended": truth.tolist(),
        "ranking": [specs[i].name for i in estimate.ranking],
        "metrics": evaluate(estimate.alpha, truth),
        "iterations": estimate.iterations,
        "converged": estimate.converged,
        "validated_items": sorted(estimate.validated),
        "judge_history": judge_history(estimate),
    }
    suffix = "" if k == settings.get("subset_budget", k) else f"_k{k}"
    write_report(artifacts / f"{args.dataset}{suffix}_report.json", report)
    return report


def run_smoke() -> dict[str, Any]:
    seed_everything(42)
    source = np.asarray(
        [[0, 0, 1, 2], [0, 1, 0, 2], [1, 1, 1, 0], [2, 0, 2, 2], [0, 0, 0, 1]],
        dtype=object,
    )
    gold = [0, 0, 1, 2, 0]
    target = np.asarray(
        [[0, 0, 1, 2], [1, 1, 1, 0], [2, 0, 2, 2], [0, 0, 0, 1]], dtype=object
    )
    initial = calibration_parameters(source, gold)
    estimate = PoolEvaluator().fit(target, initial)
    report = {
        "estimated_accuracy": estimate.alpha.tolist(),
        "ranking": estimate.ranking.tolist(),
        "iterations": estimate.iterations,
        "converged": estimate.converged,
    }
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    from .graph_data import NODE_TARGETS
    from .image_data import IMAGE_TARGETS

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")

    def common(run: argparse.ArgumentParser, target_items: int | None) -> None:
        run.add_argument("--config", default="configs/paper.yaml")
        run.add_argument("--target-items", type=int, default=target_items)
        run.add_argument("--subset-budget", type=int, help="override K; 0 = random initialization")
        run.add_argument("--max-models", type=int)
        run.add_argument("--checkpoint-dir", default="checkpoints")
        run.add_argument("--dtype", choices=["auto", "bfloat16", "float32"])
        run.add_argument("--seed", type=int)
        run.add_argument("--artifact-root")
        run.add_argument("--judge", action="store_true")

    run = sub.add_parser("text2sql")
    run.add_argument("--dataset", choices=["spider", "bird"], required=True)
    common(run, 2)
    run.add_argument("--source-candidates", type=int, default=100)
    run.add_argument("--source-items", type=int)
    run.add_argument("--require-all-judges", action="store_true")
    run.add_argument("--text2sql-root")
    run.add_argument("--bird-metadata-root")
    run.add_argument("--bird-database-root")

    image = sub.add_parser("image")
    image.add_argument("--dataset", choices=sorted(IMAGE_TARGETS), required=True)
    common(image, None)
    image.add_argument("--image-root")

    node = sub.add_parser("node")
    node.add_argument("--dataset", choices=list(NODE_TARGETS), required=True)
    common(node, None)
    node.add_argument("--node-root")
    node.add_argument("--source", help="GNNEvaluator source graph (default from config)")
    node.add_argument(
        "--skip-external", action="store_true",
        help="leave out GraphGPT/GraphPFN/LLaGA when no external runner is configured",
    )

    args = parser.parse_args(argv)
    if args.command == "smoke":
        run_smoke()
    elif args.command == "text2sql":
        args.max_models = args.max_models or 35
        run_text2sql(args)
    elif args.command == "image":
        from .image_pipeline import run_image

        run_image(args)
    else:
        from .node_pipeline import run_node

        run_node(args)


if __name__ == "__main__":
    main()
