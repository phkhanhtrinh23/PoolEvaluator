"""Collision-aware binary EM for the closed-label-space domain ports.

Implements the case-3 formulation of ``Trinh_proof.tex`` ("Collisions: Which
Closed Forms Survive") for image / node classification: estimate the per-group
collision rate ``gamma_g`` without touching the target labels, then fit
:class:`~pooleval.new_formulation.CollisionAwareNewFormulationPoolEval`.

WHY GAMMA IS NEEDED HERE AT ALL.  ``pooleval.latent`` never collapses the pool.
It keeps the full multiclass observation matrix, so a collision between two wrong
answers is *directly observed*: both models land on the same key of the per-item
class score.  The binary formulation instead reduces the pool to
``C[j,i] = 1{model j agrees with the pseudo-label}``, which discards *which*
wrong answer each model gave.  The proof's

    gamma_g = P(wrong model and wrong pseudo-label emit the SAME wrong answer)

is exactly the statistic that reduction destroys, so it must be supplied back as
a parameter.  Setting gamma=1 asserts two wrong answers always coincide -- true
for K=2, false for K=5 (graph) or K=10 (vision).

WHERE GAMMA COMES FROM.  Measuring it on the target would leak the labels the
whole setup withholds.  It is measured instead on the SOURCE validation split,
whose labels are legitimately available, through the identical pipeline: run
PoolEval on the source predictions, take its argmax pseudo-label, and count how
often a wrong model matches a wrong pseudo-label.  This is the domain analogue of
the leave-one-dataset-out estimate in ``zoo/collision_formulation_real.py``.

THE NULL.  Under independent errors over K classes, a wrong model lands on the
pseudo-label's wrong class with probability ``1/(K-1)``.  A ``gamma_hat`` well
above that null is shared error -- the signal MetaEvaluator and GNNEvaluator
chase with learned shift descriptors, read off the pool itself instead.
"""
import numpy as np

from ..inference import PoolEval
from ..new_formulation import (CollisionAwareNewFormulationPoolEval,
                               NewFormulationPoolEval)
from .adapter import from_predictions


def alpha_effective_size(run, default=120.0):
    """Beta-prior strength s_j implied by the declared prior sd: s = p(1-p)/sd^2."""
    variance = np.asarray(run.prior_sigma, dtype=float) ** 2
    numer = np.asarray(run.prior, dtype=float) * (1.0 - np.asarray(run.prior, dtype=float))
    valid = (variance > 0) & (numer > 0)
    sizes = numer[valid] / variance[valid]
    return float(np.median(sizes)) if sizes.size else default


def pseudo_labels(run, cfg, out=None):
    """Argmax of PoolEval's latent posterior -- the pseudo-label the binary model conditions on."""
    if out is None:
        out = PoolEval(cfg).evaluate(run, obs=run.true_class)
    return out, np.asarray([max(p, key=p.get) for p in out["latent_post"]],
                           dtype=run.true_class.dtype)


def gamma_from_run(run, pseudo, smoothing=1.0):
    """Per-group collision rate, conditioned on BOTH model and pseudo-label being wrong.

    Class 0 is the repo's "correct" code, so ``!= 0`` is "wrong" and equality of
    two nonzero codes is a collision on the same wrong answer.
    """
    G = int(run.n_groups)
    hits = np.zeros(G)
    eligible = np.zeros(G)
    pseudo_wrong = pseudo != 0
    for m in range(run.M):
        g = int(run.group[m])
        valid = (run.true_class[m] != 0) & pseudo_wrong
        eligible[g] += int(valid.sum())
        hits[g] += int((valid & (run.true_class[m] == pseudo)).sum())
    gamma = (hits + smoothing) / (eligible + 2.0 * smoothing)
    beta = float((int((pseudo == 0).sum()) + smoothing) / (run.N + 2.0 * smoothing))
    return dict(gamma=gamma, beta=beta, hits=hits, eligible=eligible)


def source_gamma(pool, cfg, smoothing=1.0):
    """gamma_g and beta measured on the SOURCE validation split (no target labels)."""
    if pool.get("pred_s") is not None:
        pred_s = np.asarray(pool["pred_s"])          # task stores predictions directly
    else:
        pred_s = np.asarray(pool["prob_s"]).argmax(-1)
    run_s = from_predictions(pred_s, np.asarray(pool["src_gold_val"]),
                             np.asarray(pool["group"]), prior=np.asarray(pool["prior"]))
    cfg_s = _resize(cfg, run_s)
    _, pseudo_s = pseudo_labels(run_s, cfg_s)
    stats = gamma_from_run(run_s, pseudo_s, smoothing=smoothing)
    stats["split"] = "source-val"
    return stats


def target_gamma(pool, cfg, run, pseudo, smoothing=1.0):
    """ORACLE diagnostic only: the same statistic measured on the target labels."""
    stats = gamma_from_run(run, pseudo, smoothing=smoothing)
    stats["split"] = "target (oracle)"
    return stats


def _resize(cfg, run):
    import dataclasses
    return dataclasses.replace(cfg, M=run.M, N=run.N, n_groups=run.n_groups)


def collision_estimates(pool, cfg, run, pseudo_out=None, beta_strength=120.0,
                        smoothing=1.0, include_oracle=True):
    """Fit the binary and collision-aware estimators on one domain pool.

    Returns ``(estimates, diagnostics)`` where ``estimates`` maps a method label to
    an ``[M]`` accuracy vector, directly comparable with everything ``score_pool``
    already produces.
    """
    out, pseudo = pseudo_labels(run, cfg, out=pseudo_out)
    strength = alpha_effective_size(run)
    n_classes = int(np.asarray(pool["n_classes"]))
    null = 1.0 / max(n_classes - 1, 1)

    src = source_gamma(pool, cfg, smoothing=smoothing)
    est = {}

    # gamma = 1: the pre-collision binary model. Asserts two wrong answers always
    # coincide, which is only true for a two-class label space.
    est["NF binary EM (g=1)"] = NewFormulationPoolEval(cfg).evaluate(
        run, obs=run.true_class, pseudo_out=out)["acc"]

    def fit(gamma, beta_init):
        return CollisionAwareNewFormulationPoolEval(
            cfg, gamma, strength, beta_init=beta_init,
            beta_strength=beta_strength).evaluate(
                run, obs=run.true_class, pseudo_out=out)["acc"]

    # gamma = 1/(K-1): the independent-error null, available with no data at all.
    est["NF collision (g=null)"] = fit(np.full(run.n_groups, null), src["beta"])
    # gamma from the source split: the deployable estimator.
    est["NF collision (g=source)"] = fit(src["gamma"], src["beta"])

    diag = dict(null=null, n_classes=n_classes, alpha_strength=strength,
                source=_jsonable(src))
    if include_oracle:
        tgt = target_gamma(pool, cfg, run, pseudo, smoothing=smoothing)
        est["NF collision (g=oracle)"] = fit(tgt["gamma"], tgt["beta"])
        diag["target"] = _jsonable(tgt)
    return est, diag


def _jsonable(stats):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in stats.items()}
