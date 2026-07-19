"""Graded execution-equivalence kernel.

Real PoolEval-SQL compares query RESULT SETS on the live database along a
canonicalization ladder: LA0 exact multiset match, LA1 canonicalized (sort/normalize
types/round floats/ignore column names -- the standard benchmark comparator), LA2
multi-instance (re-run on t constraint-preserving sub-instances). We simulate the
*measured equivalence error* of each level: LA0 has high precision but poor recall
(false disagreements from ordering/types), LA1 recovers recall, LA2 raises precision
by removing coincidental same-result matches.

`apply` perturbs the gold result-classes into OBSERVED classes with the level's
precision/recall, so downstream estimators see realistic equivalence noise.
"""
import numpy as np


_LEVELS = {"LA0": ("recall_LA0", "precision_LA0"),
           "LA1": ("recall_LA1", "precision_LA1"),
           "LA2": ("recall_LA2", "precision_LA2")}


def apply(true_class, cfg, level="LA2", rng=None):
    # Real-data mode: true_class already holds measured result-equivalence classes,
    # so there is nothing to simulate -- the observation IS the true class.
    if getattr(cfg, "real_data", False):
        return true_class.copy()
    rng = rng or np.random.default_rng(cfg.seed + 99)
    rk, pk = _LEVELS[level]
    recall, precision = getattr(cfg, rk), getattr(cfg, pk)
    M, N = true_class.shape
    obs = true_class.copy()

    # recall < 1: split truly-equal results into distinct surface forms
    # (false disagreement). Give those cells fresh unique ids.
    split = rng.random((M, N)) < (1 - recall)
    uid = 5_000_000
    idx = np.argwhere(split)
    for (m, i) in idx:
        obs[m, i] = uid
        uid += 1

    # precision < 1: merge some truly-distinct results into a shared spurious id
    # (false agreement) per item.
    merge = rng.random((M, N)) < (1 - precision)
    for i in range(N):
        cells = np.where(merge[:, i])[0]
        if len(cells) >= 2:
            obs[cells, i] = 9_000_000 + i
    return obs


def measure(true_class, obs_class):
    """Precision/recall of observed equivalence vs gold, over all model pairs."""
    M, N = true_class.shape
    tp = fp = fn = 0
    for i in range(N):
        tc, oc = true_class[:, i], obs_class[:, i]
        for a in range(M):
            for b in range(a + 1, M):
                gold = tc[a] == tc[b]
                obs = oc[a] == oc[b]
                if obs and gold:
                    tp += 1
                elif obs and not gold:
                    fp += 1
                elif (not obs) and gold:
                    fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall
