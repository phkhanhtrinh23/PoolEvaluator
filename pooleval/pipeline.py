"""Runnable paper pipeline: retrieve, generate, EX-Extended, EM, and judge."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adapters import predict_text2sql, text2sql_prompt
from .config import load_model_pool, paper_config
from .core import (
    estimate_pool, judge_rounds, release, repeat_settings, subset_budget, task_settings, write_report,
)
from .data import HAS_TRAIN_SPLIT, TEXT2SQL_DATASETS, Text2SQLItem, load_text2sql, schema_prompt
from .databases import build_instances
from .encoders import text_encoder
from .estimator import PoolEvaluator, calibration_parameters
from .execution import execute, execute_raw, response_classes, signature
from .judges import CachedJudge, JudgeItem, paper_judges
from .metrics import evaluate
from .models import load_model
from .repeats import run_repeated
from .reproducibility import seed_everything
from .result_match import entsql_match, spider2_match
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
) -> tuple[dict[str, dict[str, str]], dict[str, float | None]]:
    """Generate (or reuse) every model's SQL; also return each model's generation seconds.

    Generation time is accumulated in ``<model>.timing.json`` next to the prediction
    cache, so reused predictions still carry their cost (seconds per item x items).
    """
    predictions: dict[str, dict[str, str]] = {}
    seconds: dict[str, float | None] = {}
    if not items:
        return predictions, {spec.name: 0.0 for spec in specs}
    prompts = [
        text2sql_prompt(item.question, schema_prompt(item.schema), item.evidence, item.dialect) for item in items
    ]
    for index, spec in enumerate(specs, start=1):
        path = _cache_file(artifact_root, items[0].dataset, split, spec.name)
        timing_path = path.with_name(path.stem + ".timing.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        cached: dict[str, str] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        timing = (
            json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {"seconds": 0.0, "items": 0}
        )
        missing = [i for i, item in enumerate(items) if item.id not in cached]
        print(f"[generate:{split}] {index}/{len(specs)} {spec.name}, missing={len(missing)}")
        if missing:
            start = time.perf_counter()
            loaded = load_model(spec, cache_dir=cache_dir, dtype=dtype)
            generated = predict_text2sql(loaded, [prompts[i] for i in missing])
            timing["seconds"] += time.perf_counter() - start
            timing["items"] += len(missing)
            cached.update({items[i].id: sql for i, sql in zip(missing, generated)})
            path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
            timing_path.write_text(json.dumps(timing), encoding="utf-8")
            release(loaded)
            del loaded
        predictions[spec.name] = cached
        seconds[spec.name] = timing["seconds"] / timing["items"] * len(items) if timing["items"] else None
    return predictions, seconds


def _suites(items: Sequence[Text2SQLItem], root: Path) -> dict[Any, list[Any]]:
    suites: dict[Any, list[Any]] = {}
    for item in items:
        if item.db_path not in suites:
            records = build_instances(item.db_path, root / item.dataset)
            instances = [
                Path(record.instance) if item.dialect == "sqlite" else record.instance for record in records
            ]
            suites[item.db_path] = [item.db_path] + instances
    return suites


@dataclass
class Execution:
    """Executed pool outputs for a set of items."""

    classes: np.ndarray  # response classes (EX-Extended signatures)
    correct: np.ndarray  # per item and model, against the gold (False where no gold)
    valid: np.ndarray  # runs on the original database and all five instances
    has_gold: np.ndarray
    gold_classes: list[Any]
    model_seconds: np.ndarray  # per-model execution time of its own outputs


def _gold_correct(item: Text2SQLItem, sql: str, gold: tuple[Any, ...] | None, answer: tuple[Any, ...], timeout: float) -> bool:
    if item.gold_sql:
        return all(a is not None and a == b for a, b in zip(answer, gold or ()))
    if not item.gold_results:
        return False
    result = execute_raw(item.db_path, sql, timeout)
    if result is None:
        return False
    columns, rows = result
    if item.gold_matcher == "entsql":
        return entsql_match(rows, item.gold_results[0])
    spec = json.loads(item.gold_spec or "{}")
    return spider2_match(columns, rows, item.gold_results, spec.get("condition_cols"), spec.get("ignore_order", False))


def _execute_pool(
    items: Sequence[Text2SQLItem],
    specs: Sequence[Any],
    predictions: dict[str, dict[str, str]],
    suites: dict[Any, list[Any]],
    timeout: float,
) -> Execution:
    """Execute every output on the original database and its five EX-Extended instances.

    Correctness is EX-Extended against gold SQL or, for benchmarks that release gold
    result tables, the benchmark's official table matcher.  An output is a valid judge
    candidate only if it executes on the original database and all five instances.
    """
    n, m = len(items), len(specs)
    classes = np.zeros((n, m), dtype=int)
    correctness = np.zeros((n, m), dtype=bool)
    valid = np.zeros((n, m), dtype=bool)
    has_gold = np.array([item.has_gold for item in items], dtype=bool)
    model_seconds = np.zeros(m, dtype=float)
    gold_classes: list[Any] = []
    for i, item in enumerate(items):
        paths = suites[item.db_path]
        gold = signature(item.gold_sql, paths, timeout) if item.gold_sql else None
        predicted = []
        for j, spec in enumerate(specs):
            start = time.perf_counter()
            predicted.append(signature(predictions[spec.name][item.id], paths, timeout))
            model_seconds[j] += time.perf_counter() - start
        row_classes = response_classes(predicted)
        classes[i, :] = row_classes
        for j, answer in enumerate(predicted):
            valid[i, j] = bool(answer) and all(part is not None for part in answer)
            if has_gold[i]:
                correctness[i, j] = _gold_correct(item, predictions[specs[j].name][item.id], gold, answer, timeout)
        matching = [row_classes[j] for j in range(m) if correctness[i, j]]
        gold_classes.append(matching[0] if matching else max(row_classes, default=-1) + 1)
        print(f"[execute] {i + 1}/{n} {item.dataset}/{item.db_id}")
    return Execution(classes, correctness, valid, has_gold, gold_classes, model_seconds)


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
    rounds = judge_rounds(getattr(args, "judge_rounds", None), settings)
    data_cfg = config["data"]
    layout = {
        "text2sql_root": args.text2sql_root or data_cfg["text2sql_root"],
        "bird_metadata_root": args.bird_metadata_root or data_cfg["bird_metadata_root"],
        "bird_database_root": args.bird_database_root or data_cfg["bird_database_root"],
        "language": getattr(args, "language", None) or settings.get("language", "en"),
        "database_names": (settings.get("database_names") or {}).get(args.dataset, {}),
        "max_evidence_chars": settings.get("max_evidence_chars"),
    }
    seconds: dict[str, float] = {}
    target = load_text2sql(args.dataset, "target", args.target_items, **layout)
    source: list[Text2SQLItem] = []
    retrieval: list[tuple[str, float]] = []
    if k > 0:
        start = time.perf_counter()
        names = [args.dataset] if args.dataset in HAS_TRAIN_SPLIT else list(
            settings.get("calibration_sources", ["spider", "bird"])
        )
        source_pool: list[Text2SQLItem] = []
        for name in names:
            source_pool += load_text2sql(name, "train", args.source_candidates, **layout)
        source_pool = [item for item in source_pool if item.gold_sql]
        encoder = text_encoder(settings.get("encoder", "colbert-ir/colbertv2.0"), args.checkpoint_dir)
        source, retrieval = select_subsets(target, source_pool, k, encoder)
        release(encoder)
        del encoder
        if args.source_items:
            source = source[: args.source_items]
        seconds["retrieval"] = time.perf_counter() - start
        print(f"[retrieval] selected {len(source)} items from {min(k, len(retrieval))} database subsets")
    else:
        seconds["retrieval"] = 0.0
        print("[retrieval] K=0: random truncated-normal initialization, no calibration subsets")

    specs = load_model_pool("text2sql")[: args.max_models]
    artifacts = Path(args.artifact_root or data_cfg["output_root"]).resolve()
    predictions_target, target_generation = _generate(target, specs, "target", artifacts, args.checkpoint_dir, dtype)
    predictions_source, source_generation = _generate(source, specs, "source", artifacts, args.checkpoint_dir, dtype)
    db_root = artifacts / "db_instances"
    timeout = float(config["execution"]["timeout_seconds"])
    start = time.perf_counter()
    target_suites = _suites(target, db_root)
    source_suites = _suites(source, db_root) if source else {}
    seconds["database_instances"] = time.perf_counter() - start
    target_exec = _execute_pool(target, specs, predictions_target, target_suites, timeout)
    source_matrix = source_gold = None
    source_exec_seconds = np.zeros(len(specs))
    if source:
        source_exec = _execute_pool(source, specs, predictions_source, source_suites, timeout)
        source_matrix, source_gold, source_exec_seconds = source_exec.classes, source_exec.gold_classes, source_exec.model_seconds
    target_matrix = target_exec.classes

    judge = None
    if args.judge and rounds > 0:
        judge_cfg = config["judges"]["text2sql"]
        ensemble = paper_judges(
            allow_partial=not args.require_all_judges and judge_cfg.get("allow_partial", True),
            members=judge_cfg.get("members"),
        )

        def make_item(item_index: int, candidates: list[Any]) -> JudgeItem:
            item = target[item_index]
            rendered: list[str] = []
            for candidate in candidates:
                model_index = int(np.flatnonzero(target_matrix[item_index] == candidate)[0])
                sql = predictions_target[specs[model_index].name][item.id]
                result = execute(item.db_path, sql, timeout)
                rendered.append(f"SQL: {sql}\nOriginal-database result: {repr(result)[:1200]}")
            return JudgeItem(item.question, schema_prompt(item.schema), tuple(rendered), item.evidence)

        judge = CachedJudge(
            ensemble, make_item, f"text2sql:{args.dataset}", artifacts / "judge_cache" / f"text2sql_{args.dataset}.json"
        )

    subset_ids = [item.db_id for item in source]
    run = estimate_pool(
        target_matrix, config, seed, valid=target_exec.valid,
        source=source_matrix, source_gold=source_gold, subset_ids=subset_ids,
        judge=judge, rounds=rounds,
    )
    estimate = run.estimate
    labeled = bool(target_exec.has_gold.all())
    truth = target_exec.correct.mean(axis=0) if labeled else None

    generation = [
        None if target_generation[s.name] is None or source_generation[s.name] is None
        else target_generation[s.name] + source_generation[s.name]
        for s in specs
    ]
    model_cost = [
        None if g is None else g + float(target_exec.model_seconds[j] + source_exec_seconds[j])
        for j, g in enumerate(generation)
    ]
    shared = seconds["retrieval"] + seconds["database_instances"]
    method = sum(run.seconds[key] for key in ("stage1_prior", "stage2_em", "stage3_select_em", "judge"))
    latency = {
        **seconds,
        "model_generation": None if None in generation else float(sum(generation)),
        "execution": float(target_exec.model_seconds.sum() + source_exec_seconds.sum()),
        **{key: float(value) for key, value in run.seconds.items()},
    }
    latency["total"] = None if None in model_cost else shared + float(sum(model_cost)) + method

    report = {
        "task": "text2sql",
        "dataset": args.dataset,
        "seed": seed,
        "deterministic": deterministic,
        "requested_inference_dtype": dtype,
        "subset_budget": k,
        "judge_rounds_budget": rounds,
        "initialization": run.initialization,
        "models": [spec.name for spec in specs],
        "target_items": len(target),
        "calibration_items": len(source),
        "retrieved_subsets": retrieval[:k],
        "estimated_accuracy": estimate.alpha.tolist(),
        "ranking": [specs[i].name for i in estimate.ranking],
        "iterations": estimate.iterations,
        "converged": estimate.converged,
        "validated_items": sorted(estimate.validated),
        "judge_rounds": run.round_table(truth),
        "latency_seconds": latency,
    }
    if truth is not None:
        report["true_ex_extended"] = truth.tolist()
        report["metrics"] = evaluate(estimate.alpha, truth)
    if getattr(args, "repeats", 0):
        sizes, repeats = repeat_settings(args, config)
        report["repeated"] = run_repeated(
            target_matrix, config, seed,
            slot_names=[spec.name for spec in specs],
            correct=target_exec.correct if labeled else None,
            valid=target_exec.valid,
            source=source_matrix, source_gold=source_gold, subset_ids=subset_ids,
            judge=judge, rounds=rounds, sizes=sizes, repeats=repeats,
            column_seconds=model_cost, fixed_seconds=shared,
        )
    suffix = "" if k == settings.get("subset_budget", k) else f"_k{k}"
    if rounds != settings.get("judge_rounds", rounds):
        suffix += f"_v{rounds}"
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
        run.add_argument("--judge-rounds", type=int, help="override V, the number of single-item judge rounds")
        run.add_argument(
            "--repeats", type=int, default=0,
            help="also run this many repeats per pool size over sampled pools (the paper uses 10)",
        )
        run.add_argument("--pool-sizes", type=int, nargs="+", help="sampled pool sizes (default from config)")
        run.add_argument("--max-models", type=int)
        run.add_argument("--checkpoint-dir", default="checkpoints")
        run.add_argument("--dtype", choices=["auto", "bfloat16", "float32"])
        run.add_argument("--seed", type=int)
        run.add_argument("--artifact-root")
        run.add_argument("--judge", action="store_true")

    run = sub.add_parser("text2sql")
    run.add_argument("--dataset", choices=list(TEXT2SQL_DATASETS), required=True)
    common(run, 2)
    run.add_argument("--source-candidates", type=int, default=100)
    run.add_argument("--source-items", type=int)
    run.add_argument("--require-all-judges", action="store_true")
    run.add_argument("--language", choices=["en", "zh"], help="EntSQL question language")
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
