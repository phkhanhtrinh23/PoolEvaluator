"""Confidence-based label-free baselines used by MetaEvaluator and GNNEvaluator.

These are the methods PoolEval must actually beat in the vision / graph domains.
Unlike the pool baselines in `baselines/`, they score each model INDEPENDENTLY
from its own softmax confidences -- they never look at the rest of the pool. That
is precisely the axis PoolEval trades against: no pool needed, but no cross-model
information either, so their errors do not cancel in a ranking.

  DoC     Guillory et al. 2021 -- Difference of Confidence
  ATC-MC  Garg et al. 2022 -- Average Thresholded Confidence, max-softmax score
  ATC-NE  Garg et al. 2022 -- ATC with negative-entropy score
  GDE     Jiang et al. 2022 -- Generalized Disagreement Equality (needs 2 seeds)
"""
import numpy as np


def _entropy_score(p):
    return (p * np.log(np.clip(p, 1e-12, None))).sum(1)      # negative entropy


def doc(prob_s, prob_t, src_acc):
    """acc_t ~= acc_s - (mean source confidence - mean target confidence)."""
    return float(np.clip(src_acc - (prob_s.max(1).mean() - prob_t.max(1).mean()), 0, 1))


def atc(prob_s, prob_t, src_correct, score="mc"):
    """Threshold t s.t. the source score-below-t rate matches the source error rate;
    predicted target accuracy = fraction of target scores at or above t."""
    f = (lambda p: p.max(1)) if score == "mc" else _entropy_score
    ss, st = f(prob_s), f(prob_t)
    err = 1.0 - float(np.mean(src_correct))
    t = np.quantile(ss, np.clip(err, 0, 1))
    return float(np.mean(st >= t))


def gde(pred_a, pred_b):
    """Disagreement between two independently-seeded runs estimates the error."""
    return float(1.0 - np.mean(pred_a != pred_b))


def confidence_baselines(pool):
    """-> {method: [M] accuracy estimates} for a pool dict from build_pool.

    Returns {} when the task cannot supply per-item softmaxes. Knowledge-graph
    completion is the case: a distribution over 14541 entities per query is
    neither affordable to store nor comparable to a 10-way softmax, so DoC and
    ATC are simply not defined on the same footing there.
    """
    P_s, P_t = pool.get("prob_s"), pool.get("prob_t")
    if P_s is None or P_t is None or np.ndim(P_s) != 3:
        return {}
    gold_s = pool["src_gold_val"]
    M = P_t.shape[0]
    out = {"DoC": np.zeros(M), "ATC-MC": np.zeros(M), "ATC-NE": np.zeros(M)}
    for m in range(M):
        correct_s = P_s[m].argmax(1) == gold_s
        out["DoC"][m] = doc(P_s[m], P_t[m], correct_s.mean())
        out["ATC-MC"][m] = atc(P_s[m], P_t[m], correct_s, "mc")
        out["ATC-NE"][m] = atc(P_s[m], P_t[m], correct_s, "ne")
    return out
