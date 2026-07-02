"""Shared helpers for pool-evaluation baselines.

Baselines that use agreement see results compared at canonicalization level LA1
(the standard benchmark comparator); PoolEval-SQL additionally uses the LA2
multi-instance kernel, which is part of its contribution.
"""
import numpy as np
from pooleval import kernel


def obs_at(run, cfg, level="LA1"):
    return kernel.apply(run.true_class, cfg, level=level,
                        rng=np.random.default_rng(cfg.seed + 7))


def majority_accuracy(obs):
    """Fraction each model agrees with the per-item execution-majority answer."""
    M, N = obs.shape
    acc = np.zeros(M)
    maj = np.zeros(N, dtype=obs.dtype)
    for i in range(N):
        vals, counts = np.unique(obs[:, i], return_counts=True)
        maj[i] = vals[np.argmax(counts)]
    for m in range(M):
        acc[m] = np.mean(obs[m] == maj)
    return acc, maj
