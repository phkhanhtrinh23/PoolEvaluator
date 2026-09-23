"""Runnable paper pipeline: retrieve, generate, EX-Extended, EM, and judge."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adapters import predict_text2sql, text2sql_prompt
from .config import load_model_pool, paper_config
from .data import Text2SQLItem, load_text2sql, schema_prompt
from .databases import build_instances
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
            del loaded
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
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
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    matrix = np.zeros((len(items), len(specs)), dtype=int)
    correctness = np.zeros((len(items), len(specs)), dtype=bool)
    gold_answers: list[int] = []
    for i, item in enumerate(items):
        paths = suites[item.db_path]
        gold = signature(item.gold_sql, paths, timeout)
        predicted = [signature(predictions[spec.name][item.id], paths, timeout) for spec in specs]
        classes = response_classes(predicted)
        matrix[i, :] = classes
        for j, answer in enumerate(predicted):
            correctness[i, j] = all(a is not None and a == b for a, b in zip(answer, gold))
        matching = [classes[j] for j in range(len(specs)) if correctness[i, j]]
        gold_answers.append(matching[0] if matching else max(classes, default=-1) + 1)
        print(f"[execute] {i + 1}/{len(items)} {item.dataset}/{item.db_id}")
    return matrix, correctness, gold_answers


def run_text2sql(args: argparse.Namespace) -> dict[str, Any]:
    config = paper_config(args.config)
    runtime_cfg = config.get("runtime", {})
    requested_seed = getattr(args, "seed", None)
    requested_dtype = getattr(args, "dtype", None)
    seed = int(requested_seed if requested_seed is not None else config["seed"])
    deterministic = bool(runtime_cfg.get("deterministic", True))
    dtype = requested_dtype or runtime_cfg.get("inference_dtype", "bfloat16")
    seed_everything(seed, deterministic=deterministic)
    data_cfg = config["data"]
    layout = {
        "fusion_root": args.fusion_sql_root or data_cfg["fusion_sql_root"],
        "bird_metadata_root": args.bird_metadata_root or data_cfg["bird_metadata_root"],
        "bird_database_root": args.bird_database_root or data_cfg["bird_database_root"],
    }
    target = load_text2sql(args.dataset, "target", args.target_items, **layout)
    source_pool = load_text2sql(args.dataset, "train", args.source_candidates, **layout)
    source, retrieval = select_subsets(target, source_pool, config["method"]["subset_budget"])
    if args.source_items:
        source = source[: args.source_items]
    print(f"[retrieval] selected {len(source)} items from {min(15, len(retrieval))} database subsets")

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
    source_suites = _suites(source, db_root)
    timeout = float(config["execution"]["timeout_seconds"])
    target_matrix, target_correct, _ = _execute_pool(
        target, specs, predictions_target, target_suites, timeout
    )
    source_matrix, _, source_gold = _execute_pool(source, specs, predictions_source, source_suites, timeout)
    initial = calibration_parameters(
        source_matrix,
        source_gold,
        config["method"]["smoothing"],
        subset_ids=[item.db_id for item in source],
    )
    evaluator = PoolEvaluator(
        config["method"]["max_em_iterations"],
        config["method"]["tolerance"],
        config["method"]["smoothing"],
        seed=seed,
    )
    estimate = evaluator.fit(target_matrix, initial)

    if args.judge:
        ensemble = paper_judges(allow_partial=not args.require_all_judges)

        def judge(item_index: int, candidates: list[Any]) -> Any:
            item = target[item_index]
            rendered: list[str] = []
            for candidate in candidates:
                model_index = int(np.flatnonzero(target_matrix[item_index] == candidate)[0])
                sql = predictions_target[specs[model_index].name][item.id]
                result = execute(item.db_path, sql, timeout)
                rendered.append(f"SQL: {sql}\nOriginal-database result: {repr(result)[:1200]}")
            choice = ensemble.choose(
                JudgeItem(item.question, schema_prompt(item.schema), tuple(rendered), item.evidence)
            )
            return candidates[choice] if choice >= 0 else f"__none__{item_index}"

        estimate = evaluator.refine(
            target_matrix,
            (estimate.alpha, estimate.beta, estimate.gamma),
            judge,
            rounds=config["method"]["judge_rounds"],
            batch_size=config["method"]["judge_batch_size"],
        )

    truth = target_correct.mean(axis=0)
    report = {
        "dataset": args.dataset,
        "seed": seed,
        "deterministic": deterministic,
        "requested_inference_dtype": dtype,
        "models": [spec.name for spec in specs],
        "target_items": len(target),
        "calibration_items": len(source),
        "retrieved_subsets": retrieval[: config["method"]["subset_budget"]],
        "estimated_accuracy": estimate.alpha.tolist(),
        "true_ex_extended": truth.tolist(),
        "ranking": [specs[i].name for i in estimate.ranking],
        "metrics": evaluate(estimate.alpha, truth),
        "iterations": estimate.iterations,
        "converged": estimate.converged,
        "validated_items": sorted(estimate.validated),
    }
    output = artifacts / f"{args.dataset}_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[saved] {output}")
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
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    run = sub.add_parser("text2sql")
    run.add_argument("--dataset", choices=["spider", "bird"], required=True)
    run.add_argument("--config", default="configs/paper.yaml")
    run.add_argument("--target-items", type=int, default=2)
    run.add_argument("--source-candidates", type=int, default=100)
    run.add_argument("--source-items", type=int)
    run.add_argument("--max-models", type=int, default=35)
    run.add_argument("--checkpoint-dir", default="checkpoints")
    run.add_argument("--dtype", choices=["auto", "bfloat16", "float32"])
    run.add_argument("--seed", type=int)
    run.add_argument("--artifact-root")
    run.add_argument("--judge", action="store_true")
    run.add_argument("--require-all-judges", action="store_true")
    run.add_argument("--fusion-sql-root")
    run.add_argument("--bird-metadata-root")
    run.add_argument("--bird-database-root")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        run_smoke()
    else:
        run_text2sql(args)


if __name__ == "__main__":
    main()
