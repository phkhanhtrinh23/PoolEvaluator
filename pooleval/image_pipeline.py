"""End-to-end PoolEvaluator run on an image-classification target (paper Appendix C)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .adapters import image_features, image_logits
from .config import load_model_pool, paper_config
from .core import (
    estimate_pool, judge_rounds, release, repeat_settings, safe_name, subset_budget, task_settings,
    write_report,
)
from .encoders import image_encoder, rank_by_similarity
from .image_data import (
    DOMAINS, IMAGE_TARGETS, TEMPLATES, ImageSet, class_names, load_images, make_meta_subsets,
    subset_images, subset_labels,
)
from .judges import CachedJudge, JudgeEnsemble
from .metrics import evaluate
from .models import LoadedModel, load_model
from .repeats import run_repeated
from .reproducibility import seed_everything


def fit_linear_head(
    features: Any, labels: Sequence[int], num_classes: int, seed: int, epochs: int = 100
) -> Callable[[Any], Any]:
    """Train a standardized linear probe on frozen features (full-batch Adam)."""
    import torch

    x = features.float()
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
    generator = torch.Generator().manual_seed(int(seed))
    head = torch.nn.Linear(x.shape[1], num_classes)
    with torch.no_grad():
        head.weight.normal_(0.0, 0.01, generator=generator)
        head.bias.zero_()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    z = (x - mean) / std
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(head(z), y).backward()
        optimizer.step()
    head.eval()
    return lambda feats: head((feats.float() - mean) / std)


def predict_member(
    loaded: LoadedModel,
    images: Sequence[Any],
    names: Sequence[str],
    allowed: Sequence[int],
    template: str,
    head: Callable[[Any], Any] | None,
    batch_size: int,
) -> list[int]:
    """Predict into the target vocabulary, restricted to the target's ``allowed`` classes."""
    import torch

    if loaded.spec.backend == "zero_shot_image":
        logits = image_logits(loaded, images, [names[c] for c in allowed], template, batch_size)
        return [int(allowed[i]) for i in logits.argmax(-1).tolist()]
    if head is not None:
        with torch.no_grad():
            return head(image_features(loaded, images, batch_size)).argmax(-1).tolist()
    logits = image_logits(loaded, images, batch_size=batch_size)
    if logits.shape[1] != len(names):
        raise ValueError(
            f"{loaded.spec.name} has {logits.shape[1]} outputs; expected the {len(names)}-class vocabulary"
        )
    columns = torch.as_tensor(list(allowed))
    return [int(allowed[i]) for i in logits[:, columns].argmax(-1).tolist()]


def _member_predictions(
    spec: Any,
    cache: Path,
    requests: dict[str, list[Any]],
    names: Sequence[str],
    allowed: Sequence[int],
    template: str,
    needs_head: bool,
    source_train: Callable[[], ImageSet],
    settings: dict[str, Any],
    args: argparse.Namespace,
    dtype: str,
    seed: int,
) -> dict[str, list[int]]:
    cached: dict[str, Any] = json.loads(cache.read_text()) if cache.exists() else {}
    timing: dict[str, float] = cached.setdefault("_seconds", {})
    missing = [key for key in requests if key not in cached]
    if not missing:
        return cached
    start = time.perf_counter()
    loaded = load_model(spec, cache_dir=args.checkpoint_dir, dtype=dtype)
    batch = int(settings.get("batch_size", 64))
    head = None
    if needs_head and spec.backend != "zero_shot_image":
        train = source_train()
        rng = np.random.default_rng([seed, 17])
        count = min(int(settings.get("head_train_items", 5000)), len(train))
        chosen = np.sort(rng.choice(len(train), count, replace=False))
        features = image_features(loaded, train.images(chosen), batch)
        head = fit_linear_head(
            features, train.labels[chosen], len(names), seed, settings.get("head_epochs", 100)
        )
    timing.setdefault("setup", time.perf_counter() - start)
    for key in missing:
        start = time.perf_counter()
        cached[key] = predict_member(loaded, requests[key], names, allowed, template, head, batch)
        timing[key] = time.perf_counter() - start
        print(f"  {spec.name}: {key} ({len(requests[key])} images)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cached), encoding="utf-8")
    release(loaded, head)
    del loaded, head
    return cached


def run_image(args: argparse.Namespace) -> dict[str, Any]:
    config = paper_config(args.config)
    runtime_cfg = config.get("runtime", {})
    seed = int(args.seed if args.seed is not None else config["seed"])
    deterministic = bool(runtime_cfg.get("deterministic", True))
    dtype = args.dtype or runtime_cfg.get("inference_dtype", "bfloat16")
    seed_everything(seed, deterministic=deterministic)
    settings = task_settings(config, "image")
    k = subset_budget(args.subset_budget, settings)
    rounds = judge_rounds(args.judge_rounds, settings)
    source_name, vocabulary = IMAGE_TARGETS[args.dataset]
    root = Path(args.image_root or config["data"]["image_root"])
    names = class_names(vocabulary)
    template = TEMPLATES[vocabulary]

    target_all = load_images(args.dataset, "test", root, settings.get("cifar10c"))
    allowed = sorted(set(target_all.labels.tolist())) if vocabulary == "imagenet" else list(range(len(names)))
    n = min(int(args.target_items or settings.get("target_items", 1000)), len(target_all))
    target_idx = np.sort(np.random.default_rng([seed, 1]).choice(len(target_all), n, replace=False))
    target_images = target_all.images(target_idx)
    target_labels = target_all.labels[target_idx]
    print(f"[target] {args.dataset}: {n} of {len(target_all)} images, {len(allowed)} classes")

    requests: dict[str, list[Any]] = {"target": target_images}
    chosen = []
    retrieval: list[tuple[Any, float]] = []
    source_test = None
    retrieval_start = time.perf_counter()
    if k > 0:
        source_test = load_images(source_name, "test", root)
        subsets = make_meta_subsets(
            source_test, settings.get("meta_subsets", 50), settings.get("meta_subset_size", 200), seed
        )
        encoder = image_encoder(settings.get("encoder", "facebook/dinov2-small"), args.checkpoint_dir)
        target_embedding = encoder.encode(target_images)
        subset_embedding = {s.index: encoder.encode(subset_images(source_test, s)) for s in subsets}
        release(encoder)
        del encoder
        retrieval = rank_by_similarity(target_embedding, subset_embedding)
        chosen = [subsets[index] for index, _ in retrieval[:k]]
        for subset in chosen:
            requests[f"meta-{subset.index}"] = subset_images(source_test, subset)
        print(f"[retrieval] {k} of {len(subsets)} {source_name} meta-subsets: "
              + ", ".join(f"{s.index}:{s.shift}@{s.severity:.2f}" for s in chosen))
    else:
        print("[retrieval] K=0: random truncated-normal initialization, no calibration subsets")
    retrieval_seconds = time.perf_counter() - retrieval_start

    specs = load_model_pool("image")[: args.max_models]
    artifacts = Path(args.artifact_root or config["data"]["output_root"]).resolve()
    cache_dir = artifacts / "predictions" / "image" / args.dataset / (
        f"seed{seed}_n{n}_m{settings.get('meta_subset_size', 200)}"
    )
    source_train_cache: list[ImageSet] = []

    def source_train() -> ImageSet:
        if not source_train_cache:
            source_train_cache.append(load_images(source_name, "train", root))
        return source_train_cache[0]

    predictions: dict[str, dict[str, list[int]]] = {}
    for index, spec in enumerate(specs, start=1):
        print(f"[predict] {index}/{len(specs)} {spec.name}")
        predictions[spec.name] = _member_predictions(
            spec, cache_dir / f"{safe_name(spec.name)}.json", requests, names, allowed, template,
            vocabulary != "imagenet", source_train, settings, args, dtype, seed,
        )

    target_matrix = np.array([predictions[s.name]["target"] for s in specs], dtype=object).T
    source_matrix = source_gold = None
    subset_ids: list[int] = []
    if chosen:
        source_matrix = np.concatenate(
            [np.array([predictions[s.name][f"meta-{c.index}"] for s in specs], dtype=object).T for c in chosen]
        )
        source_gold = np.concatenate([subset_labels(source_test, c) for c in chosen]).tolist()
        subset_ids = [c.index for c in chosen for _ in c.indices]

    judge = None
    if args.judge and rounds > 0:
        from .local_judges import ImageJudgeItem, load_image_judge

        ensemble = JudgeEnsemble([load_image_judge(config, args.checkpoint_dir, dtype)])
        judge = CachedJudge(
            ensemble,
            lambda item, candidates: ImageJudgeItem(
                target_images[item], tuple(names[c] for c in candidates), DOMAINS[vocabulary]
            ),
            f"image:{args.dataset}:{cache_dir.name}",
            artifacts / "judge_cache" / f"image_{args.dataset}.json",
        )

    run = estimate_pool(
        target_matrix, config, seed,
        source=source_matrix, source_gold=source_gold, subset_ids=subset_ids,
        judge=judge, rounds=rounds,
    )
    estimate = run.estimate
    correct = target_matrix == target_labels[:, None]
    truth = correct.mean(axis=0).astype(float)
    requested = list(requests)
    model_cost = [
        sum(predictions[s.name]["_seconds"].get(key, float("nan")) for key in ["setup", *requested])
        for s in specs
    ]
    model_cost = [None if np.isnan(cost) else float(cost) for cost in model_cost]
    method = sum(run.seconds[key] for key in ("stage1_prior", "stage2_em", "stage3_select_em", "judge"))
    latency = {
        "retrieval": retrieval_seconds,
        "model_inference": None if None in model_cost else float(sum(model_cost)),
        **{key: float(value) for key, value in run.seconds.items()},
    }
    latency["total"] = None if None in model_cost else retrieval_seconds + float(sum(model_cost)) + method
    report = {
        "task": "image",
        "dataset": args.dataset,
        "source": source_name,
        "seed": seed,
        "deterministic": deterministic,
        "requested_inference_dtype": dtype,
        "subset_budget": k,
        "judge_rounds_budget": rounds,
        "initialization": run.initialization,
        "models": [spec.name for spec in specs],
        "target_items": n,
        "calibration_items": 0 if source_matrix is None else int(source_matrix.shape[0]),
        "retrieved_subsets": [
            {"subset": c.index, "shift": c.shift, "severity": c.severity, "similarity": score}
            for c, (_, score) in zip(chosen, retrieval)
        ],
        "estimated_accuracy": estimate.alpha.tolist(),
        "true_accuracy": truth.tolist(),
        "ranking": [specs[i].name for i in estimate.ranking],
        "metrics": evaluate(estimate.alpha, truth),
        "iterations": estimate.iterations,
        "converged": estimate.converged,
        "validated_items": sorted(estimate.validated),
        "judge_rounds": run.round_table(truth),
        "latency_seconds": latency,
    }
    if args.repeats:
        sizes, repeats = repeat_settings(args, config)
        report["repeated"] = run_repeated(
            target_matrix, config, seed,
            slot_names=[spec.name for spec in specs], correct=correct,
            source=source_matrix, source_gold=source_gold, subset_ids=subset_ids,
            judge=judge, rounds=rounds, sizes=sizes, repeats=repeats,
            column_seconds=model_cost, fixed_seconds=retrieval_seconds,
        )
    suffix = "" if k == settings.get("subset_budget", k) else f"_k{k}"
    if rounds != settings.get("judge_rounds", rounds):
        suffix += f"_v{rounds}"
    write_report(artifacts / f"image_{args.dataset}{suffix}_report.json", report)
    return report
