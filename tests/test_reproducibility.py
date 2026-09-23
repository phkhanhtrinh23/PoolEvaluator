import random

import numpy as np
import pytest

from pooleval.reproducibility import data_loader_seed_options, seed_everything


def test_seed_everything_repeats_python_and_numpy():
    seed_everything(19)
    first = (random.random(), np.random.random())
    seed_everything(19)
    second = (random.random(), np.random.random())
    assert first == second


def test_seed_everything_repeats_torch_and_builds_loader_options():
    torch = pytest.importorskip("torch")
    seed_everything(23)
    first = torch.rand(3)
    seed_everything(23)
    assert torch.equal(first, torch.rand(3))

    options = data_loader_seed_options(23)
    assert options["generator"].initial_seed() == 23
    assert callable(options["worker_init_fn"])


def test_bfloat16_falls_back_to_float32_on_cpu():
    torch = pytest.importorskip("torch")
    from pooleval.models import resolve_inference_dtype

    with pytest.warns(RuntimeWarning, match="using float32"):
        assert resolve_inference_dtype("bfloat16", device_map=None) == torch.float32


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
