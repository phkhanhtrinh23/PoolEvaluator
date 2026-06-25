"""PoolEval: label-free joint evaluation of a Text-to-SQL model pool via
anchored, correlation-aware latent-correctness inference."""
from .config import Config
from .inference import PoolEval
from .data import simulate, PoolRun
from . import metrics, kernel

__all__ = ["Config", "PoolEval", "simulate", "PoolRun", "metrics", "kernel"]
__version__ = "0.1.0"
