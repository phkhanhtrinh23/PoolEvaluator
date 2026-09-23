"""Download and lazy-load every model listed in the paper appendices.

Nothing is downloaded on import.  ``download_pool`` materializes Hugging Face
repositories, while ``ModelPool`` loads one member at a time so a 35-model pool does
not need to fit in memory simultaneously.
"""

from __future__ import annotations

import argparse
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import ModelSpec, load_model_pool


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Any
    processor: Any = None
    local_path: Path | None = None


@dataclass(frozen=True)
class ExternalCheckpoint:
    """Weights consumed by an official project-specific graph runtime."""

    checkpoint: str
    base: str | None = None
    auxiliary: str | None = None
    runtime_repo: str | None = None


def _repo_specs(task: str) -> list[ModelSpec]:
    return [s for s in load_model_pool(task) if s.backend != "pyg"]


def _shared_path(cache_dir: str | Path, model_id: str) -> Path:
    return Path(cache_dir).expanduser().resolve() / "shared" / model_id.replace("/", "__")


def download_pool(task: str, cache_dir: str | Path = "checkpoints", dry_run: bool = False) -> list[Path]:
    """Download a task pool with ``snapshot_download``; PyG architectures need no files."""
    root = Path(cache_dir).expanduser().resolve()
    planned: list[Path] = []
    specs = sum((_repo_specs(t) for t in ("text2sql", "image", "node")), []) \
        if task == "all" else _repo_specs(task)
    downloaded: set[str] = set()
    for spec in specs:
        target = root / spec.task / spec.name
        entries = [(spec.id, target)]
        if spec.base:
            entries.append((spec.base, _shared_path(root, spec.base)))
        if spec.auxiliary:
            entries.append((spec.auxiliary, _shared_path(root, spec.auxiliary)))
        for model_id, destination in entries:
            if model_id in downloaded:
                continue
            downloaded.add(model_id)
            planned.append(destination)
            print(f"{('[dry-run] ' if dry_run else '')}{model_id} -> {destination}")
            if dry_run:
                continue
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError("install model dependencies: pip install -e '.[models]'") from exc
            snapshot_download(repo_id=model_id, local_dir=destination)
    return planned


def prepare_runtimes(
    task: str = "node", cache_dir: str | Path = "checkpoints", dry_run: bool = False
) -> list[Path]:
    """Clone official source trees required by non-Transformers graph checkpoints."""
    specs = load_model_pool(task)
    repos = list(dict.fromkeys(spec.runtime_repo for spec in specs if spec.runtime_repo))
    root = Path(cache_dir).expanduser().resolve() / "runtimes"
    paths: list[Path] = []
    for repo in repos:
        name = repo.removesuffix(".git").rsplit("/", 1)[-1]
        target = root / name
        paths.append(target)
        print(f"{('[dry-run] ' if dry_run else '')}{repo} -> {target}")
        if dry_run or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", repo, str(target)], check=True)
    return paths


def _source(spec: ModelSpec, cache_dir: str | Path | None) -> str:
    if cache_dir is None:
        return spec.id
    path = Path(cache_dir).expanduser().resolve() / spec.task / spec.name
    return str(path if path.exists() else spec.id)


def _local_device(device_map: str | None) -> Any:
    import torch

    if device_map == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_map and device_map.split(":", 1)[0] in {"cpu", "cuda", "mps", "xpu"}:
        return torch.device(device_map)
    return torch.device("cpu")


def resolve_inference_dtype(dtype: str, device_map: str | None = "auto") -> Any:
    """Resolve a requested inference dtype, safely falling back from bfloat16."""
    import torch

    normalized = str(dtype).casefold().replace("torch.", "")
    if normalized == "auto":
        return "auto"
    if normalized in {"float32", "fp32", "float"}:
        return torch.float32
    if normalized not in {"bfloat16", "bf16"}:
        raise ValueError("dtype must be one of: auto, bfloat16, float32")

    device = _local_device(device_map)
    supported = False
    if device.type == "cuda" and torch.cuda.is_available():
        indices = (
            range(torch.cuda.device_count())
            if device_map == "auto"
            else [device.index if device.index is not None else torch.cuda.current_device()]
        )

        def device_supports_bfloat16(index: int) -> bool:
            with torch.cuda.device(index):
                return bool(torch.cuda.is_bf16_supported())

        supported = all(device_supports_bfloat16(index) for index in indices)
    if supported:
        return torch.bfloat16
    warnings.warn(
        f"bfloat16 inference is not safely supported on {device.type}, using float32",
        RuntimeWarning,
        stacklevel=2,
    )
    return torch.float32


def _resolve_local_inference_dtype(dtype: str, device_map: str | None) -> Any:
    """Resolve dtype for timm and PyG, which do not accept Transformers' ``auto``."""
    import torch

    resolved = resolve_inference_dtype(dtype, device_map)
    return torch.float32 if resolved == "auto" else resolved


def _load_transformers(spec: ModelSpec, source: str, device_map: str | None, dtype: str) -> LoadedModel:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install model dependencies: pip install -e '.[models]'") from exc
    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    cls = AutoModelForSeq2SeqLM if getattr(config, "is_encoder_decoder", False) else AutoModelForCausalLM
    resolved_dtype = resolve_inference_dtype(dtype, device_map)
    kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolved_dtype}
    if device_map:
        kwargs["device_map"] = device_map
    model = cls.from_pretrained(source, **kwargs).eval()
    return LoadedModel(spec, model, tokenizer, Path(source) if Path(source).exists() else None)


def _load_image(spec: ModelSpec, source: str, device_map: str | None, dtype: str) -> LoadedModel:
    try:
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoModelForImageClassification,
            AutoProcessor,
        )
    except ImportError as exc:
        raise RuntimeError("install model dependencies: pip install -e '.[models]'") from exc
    resolved_dtype = resolve_inference_dtype(dtype, device_map)
    kwargs: dict[str, Any] = {"torch_dtype": resolved_dtype, "trust_remote_code": True}
    if device_map:
        kwargs["device_map"] = device_map
    if spec.backend == "zero_shot_image":
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        model = AutoModel.from_pretrained(source, **kwargs).eval()
    else:
        processor = AutoImageProcessor.from_pretrained(source, trust_remote_code=True)
        model = AutoModelForImageClassification.from_pretrained(source, **kwargs).eval()
    return LoadedModel(spec, model, processor, Path(source) if Path(source).exists() else None)


def _load_timm(spec: ModelSpec, source: str, device_map: str | None, dtype: str) -> LoadedModel:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("install model dependencies: pip install -e '.[models]'") from exc
    model_name = f"local-dir:{source}" if Path(source).exists() else f"hf-hub:{source}"
    model = timm.create_model(model_name, pretrained=True).eval()
    model = model.to(
        device=_local_device(device_map),
        dtype=_resolve_local_inference_dtype(dtype, device_map),
    )
    data_config = timm.data.resolve_model_data_config(model)
    processor = timm.data.create_transform(**data_config, is_training=False)
    return LoadedModel(spec, model, processor, Path(source) if Path(source).exists() else None)


def _load_pyg(
    spec: ModelSpec,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    device_map: str | None,
    dtype: str,
) -> LoadedModel:
    try:
        import torch
        from torch_geometric import nn as gnn
    except ImportError as exc:
        raise RuntimeError("install model dependencies: pip install -e '.[models]'") from exc

    if spec.id == "MLP":
        model = gnn.MLP([in_channels, hidden_channels, out_channels])
    elif spec.id == "APPNP":
        class APPNPModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(in_channels, out_channels)
                self.propagation = gnn.APPNP(K=10, alpha=0.1)

            def forward(self, x, edge_index):
                return self.propagation(self.linear(x), edge_index)

        model = APPNPModel()
    elif spec.id == "JumpingKnowledge":
        model = gnn.models.GraphSAGE(in_channels, hidden_channels, 2, out_channels=out_channels, jk="cat")
    elif spec.id == "GCN2Conv":
        model = gnn.models.GCN(in_channels, hidden_channels, 2, out_channels=out_channels)
    else:
        conv = getattr(gnn, spec.id)
        special = {
            "ARMAConv": lambda: conv(in_channels, out_channels),
            "ChebConv": lambda: conv(in_channels, out_channels, K=3),
            "GINConv": lambda: conv(torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels))),
            "MixHopConv": lambda: conv(in_channels, out_channels, powers=[0, 1, 2]),
            "SuperGATConv": lambda: conv(in_channels, out_channels),
        }
        model = special.get(spec.id, lambda: conv(in_channels, out_channels))()
    model = model.eval().to(
        device=_local_device(device_map),
        dtype=_resolve_local_inference_dtype(dtype, device_map),
    )
    return LoadedModel(spec, model)


def load_model(
    spec: ModelSpec,
    cache_dir: str | Path | None = None,
    device_map: str | None = "auto",
    dtype: str = "bfloat16",
    in_channels: int = 128,
    hidden_channels: int = 128,
    out_channels: int = 10,
) -> LoadedModel:
    """Load one model-pool member using its declared backend."""
    source = _source(spec, cache_dir)
    if spec.backend == "external":
        base = spec.base
        auxiliary = spec.auxiliary
        if cache_dir is not None:
            if spec.base and _shared_path(cache_dir, spec.base).exists():
                base = str(_shared_path(cache_dir, spec.base))
            if spec.auxiliary and _shared_path(cache_dir, spec.auxiliary).exists():
                auxiliary = str(_shared_path(cache_dir, spec.auxiliary))
        external = ExternalCheckpoint(
            checkpoint=source,
            base=base,
            auxiliary=auxiliary,
            runtime_repo=spec.runtime_repo,
        )
        return LoadedModel(spec, external, local_path=Path(source) if Path(source).exists() else None)
    if spec.backend == "pyg":
        return _load_pyg(
            spec, in_channels, hidden_channels, out_channels, device_map, dtype
        )
    if spec.backend == "timm":
        return _load_timm(spec, source, device_map, dtype)
    if spec.backend in {"image_classification", "zero_shot_image"}:
        return _load_image(spec, source, device_map, dtype)
    if spec.backend == "snapshot":
        path = Path(source)
        return LoadedModel(spec, path, local_path=path if path.exists() else None)
    if spec.backend == "adapter":
        # LLaGA repositories publish projectors rather than standalone models.  Load
        # the declared base and expose the projector path to the graph adapter.
        base_spec = ModelSpec(spec.name + "-base", spec.base or "", "transformers", spec.task)
        base_source = spec.base or ""
        if cache_dir is not None and spec.base:
            cached_base = _shared_path(cache_dir, spec.base)
            if cached_base.exists():
                base_source = str(cached_base)
        loaded = _load_transformers(base_spec, base_source, device_map, dtype)
        loaded.spec = spec
        loaded.local_path = Path(source) if Path(source).exists() else None
        return loaded
    return _load_transformers(spec, source, device_map, dtype)


class ModelPool:
    """An appendix model pool that yields loaded members lazily."""

    def __init__(self, task: str, cache_dir: str | Path | None = None):
        self.task = task
        self.cache_dir = cache_dir
        self.specs = load_model_pool(task)

    def __len__(self) -> int:
        return len(self.specs)

    def iter_load(self, **kwargs: Any) -> Iterator[LoadedModel]:
        for spec in self.specs:
            yield load_model(spec, self.cache_dir, **kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("--task", choices=["text2sql", "image", "node", "all"], default="all")
    p_download = sub.add_parser("download")
    p_download.add_argument("--task", choices=["text2sql", "image", "node", "all"], default="all")
    p_download.add_argument("--cache-dir", default="checkpoints")
    p_download.add_argument("--dry-run", action="store_true")
    p_load = sub.add_parser("load", help="sequentially load members into the runtime")
    p_load.add_argument("--task", choices=["text2sql", "image", "node"], required=True)
    p_load.add_argument("--cache-dir", default="checkpoints")
    p_load.add_argument("--max-models", type=int)
    p_load.add_argument("--device-map", default="auto")
    p_load.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="bfloat16")
    p_load.add_argument("--seed", type=int, default=42)
    p_runtime = sub.add_parser("runtimes", help="prepare official external graph runtimes")
    p_runtime.add_argument("--task", choices=["node"], default="node")
    p_runtime.add_argument("--cache-dir", default="checkpoints")
    p_runtime.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "download":
        download_pool(args.task, args.cache_dir, args.dry_run)
        return
    if args.command == "load":
        from .reproducibility import seed_everything

        seed_everything(args.seed)
        specs = load_model_pool(args.task)[: args.max_models]
        for index, spec in enumerate(specs, start=1):
            print(f"[load] {index}/{len(specs)} {spec.name}")
            loaded = load_model(
                spec, args.cache_dir, device_map=args.device_map, dtype=args.dtype
            )
            print(f"  ready backend={spec.backend} type={type(loaded.model).__name__}")
            del loaded
        return
    if args.command == "runtimes":
        prepare_runtimes(args.task, args.cache_dir, args.dry_run)
        return
    tasks = ("text2sql", "image", "node") if args.task == "all" else (args.task,)
    for task in tasks:
        specs = load_model_pool(task)
        print(f"{task}: {len(specs)}")
        for spec in specs:
            print(f"  {spec.name:42s} {spec.backend:20s} {spec.id}")


if __name__ == "__main__":
    main()
