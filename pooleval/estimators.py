"""Design-based accuracy estimators for a probability sample of expert-validated items.

The model in :mod:`pooleval.validated_em` produces per-cell correctness probabilities.
Averaging them gives an accuracy estimate whose bias is entirely inherited from the
model: if the pool's errors are correlated in a way the model gets wrong, the average is
wrong and nothing in the unlabeled data reveals it.

A probability sample of validated items fixes that, but only under one condition, and it
is not negotiable.  Horvitz-Thompson is an ESTIMATOR, not a sampling strategy.  It undoes
a selection bias whose magnitude it knows -- the inclusion probability -- and an item that
had NO chance of being selected is unrecoverable at any weight, because ``1/0`` is not a
number.  So a deterministic "take the top 40 by score" cannot be HT-corrected afterwards:
those items have ``pi = 1`` and every other item has ``pi = 0``.  The design must carry
known, strictly positive ``pi_i`` from the start.  See
:func:`pooleval.validated_em.sampling_probabilities` for how an acquisition score is
turned into one.

Conventions.  ``values[j, k] = 1`` iff model ``j`` is correct on the ``k``-th SAMPLED
item; ``pi[k]`` is that item's inclusion probability; ``N`` is the size of the full
target set.  All estimators return one accuracy per model.
"""
import numpy as np

__all__ = ["horvitz_thompson", "ht_variance", "model_assisted",
           "model_assisted_variance", "inverse_variance_fuse"]


def _check(values, pi):
    values = np.atleast_2d(np.asarray(values, dtype=float))
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 1 or values.shape[1] != len(pi):
        raise ValueError("values must be [M, n_sampled] aligned with pi")
    if np.any(pi <= 0) or np.any(pi > 1):
        raise ValueError("inclusion probabilities must lie in (0, 1]")
    return values, pi


def horvitz_thompson(values, pi, N):
    """``mu_HT = (1/N) sum_{i in S} Z_i / pi_i`` -- unbiased for any design with pi > 0.

    Unbiased because ``E[I_i] = pi_i`` exactly cancels the weight:
    ``E[mu_HT] = (1/N) sum_i Z_i pi_i / pi_i = (1/N) sum_i Z_i = mu``.  Note this holds
    whatever the model believes; it is a property of the sampling design alone.
    """
    values, pi = _check(values, pi)
    return (values / pi[None, :]).sum(axis=1) / float(N)


def ht_variance(values, pi, N):
    """Poisson-design variance of :func:`horvitz_thompson`.

    Exact when inclusion indicators are independent.  Under sampling WITHOUT replacement
    the true variance is smaller (the joint inclusion probabilities contribute a negative
    term), so this is a conservative bound rather than a lie in the dangerous direction.
    """
    values, pi = _check(values, pi)
    return ((values ** 2) * ((1.0 - pi) / pi ** 2)[None, :]).sum(axis=1) / float(N) ** 2


def model_assisted(model_pred, values, sampled_index, pi, N=None):
    """Model prediction over everything, plus an HT-corrected residual over the sample.

        ``mu_hat = (1/N) sum_i p_i  +  (1/N) sum_{i in S} (Z_i - p_i) / pi_i``

    Unbiased for the same reason HT is -- the correction term has expectation
    ``(1/N) sum_i (Z_i - p_i)`` -- so a bad model costs variance, never bias.  When the
    model is good the residuals are near zero and the variance is far below plain HT's.
    This is the estimator to report: it uses the model where the model is trustworthy and
    lets the design carry the part that is not.
    """
    model_pred = np.atleast_2d(np.asarray(model_pred, dtype=float))
    sampled_index = np.asarray(sampled_index, dtype=int)
    values, pi = _check(values, pi)
    N = model_pred.shape[1] if N is None else int(N)
    residual = values - model_pred[:, sampled_index]
    return model_pred.mean(axis=1) + (residual / pi[None, :]).sum(axis=1) / float(N)


def model_assisted_variance(model_pred, values, sampled_index, pi, N=None):
    """Poisson-design variance of :func:`model_assisted`, driven by the RESIDUALS.

    This is the formal statement of "a good model buys precision, not correctness": the
    bias is zero either way, and only this variance improves as ``p_i`` approaches
    ``Z_i``.
    """
    model_pred = np.atleast_2d(np.asarray(model_pred, dtype=float))
    sampled_index = np.asarray(sampled_index, dtype=int)
    values, pi = _check(values, pi)
    N = model_pred.shape[1] if N is None else int(N)
    residual = values - model_pred[:, sampled_index]
    return ((residual ** 2) * ((1.0 - pi) / pi ** 2)[None, :]).sum(axis=1) / float(N) ** 2


def inverse_variance_fuse(estimates, variances, eps=1e-12):
    """Precision-weighted combination of independent estimates of the same quantity.

    ``mu = sum_k mu_k / v_k  /  sum_k 1 / v_k``, the minimum-variance unbiased linear
    combination, with ``1 / sum_k (1 / v_k)`` as its variance.  Used to combine the
    model-based accuracy with the design-based one: whichever is sharper dominates, and
    neither has to be chosen in advance.

    Every input must already be UNBIASED for this to be honest.  Fusing a biased model
    estimate with an unbiased design estimate buys precision by importing bias, which is
    exactly the trade this module exists to avoid -- so pass the model estimate only when
    you are willing to defend it.
    """
    est = np.atleast_2d(np.asarray(estimates, dtype=float))
    var = np.atleast_2d(np.asarray(variances, dtype=float))
    if est.shape != var.shape:
        raise ValueError("estimates and variances must have the same shape")
    weight = 1.0 / np.clip(var, eps, None)
    total = weight.sum(axis=0)
    return (est * weight).sum(axis=0) / total, 1.0 / total
