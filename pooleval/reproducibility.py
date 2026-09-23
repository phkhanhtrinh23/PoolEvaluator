"""Reproducible runtime configuration for inference and estimator execution."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed supported runtimes and request deterministic PyTorch execution.

    PyTorch is optional for the CPU-only estimator. Setting ``PYTHONHASHSEED``
    affects child processes started after this call, while ``random.seed`` seeds
    Python's RNG in the current process.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        # Required by deterministic CUDA matrix multiplication on CUDA 10.2+.
        # It must be present before the first CUDA operation.
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":16:8", ":4096:8"}:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return seed


def seed_data_loader_worker(_worker_id: int) -> None:
    """Seed Python and NumPy inside a PyTorch ``DataLoader`` worker."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to seed a DataLoader worker") from exc

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def data_loader_seed_options(seed: int) -> dict[str, Any]:
    """Return deterministic keyword arguments for a PyTorch ``DataLoader``."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to configure a DataLoader") from exc

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return {"worker_init_fn": seed_data_loader_worker, "generator": generator}
