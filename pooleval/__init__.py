"""PoolEval-SQL: label-free joint evaluation and ranking of a Text-to-SQL model
pool via anchored, correlation-aware latent-correctness inference.

The paper names the framework PoolEval-SQL; this package is imported as `pooleval`
and its estimator is the `PoolEval` class."""
from .config import Config
from .inference import PoolEval
from .active import ActivePoolEval, ActiveConfig, SimulatedJudge
from .data import simulate, PoolRun
from . import metrics, kernel

__all__ = ["Config", "PoolEval", "ActivePoolEval", "ActiveConfig", "SimulatedJudge",
           "simulate", "PoolRun", "metrics", "kernel"]
__version__ = "0.1.0"
