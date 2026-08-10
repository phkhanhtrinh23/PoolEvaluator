"""Closed-form pseudo-label agreement EM from ``new_formulation/``.

This estimator intentionally lives beside, rather than replacing, :class:`PoolEval`.
It uses the existing estimator only to construct one pseudo-label per item from the
graded execution kernel, meta-dataset accuracy priors, provenance discount, and
execution verifier.  Given those fixed pseudo-labels it fits the binary model from
the new formulation with its exact closed-form EM updates.
"""
import numpy as np

from . import kernel as kernelmod
from .inference import PoolEval


def _clip_probability(x, eps):
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def agreement_em(C, alpha_init, beta_init=None, max_iters=200, tol=1e-8,
                 eps=1e-6):
    """Fit the binary pseudo-label agreement model by closed-form EM.

    Parameters
    ----------
    C : array-like, shape (M, N)
        ``C[j, i] = 1`` iff model ``j`` agrees with the fixed pseudo-label on item
        ``i``.
    alpha_init : array-like, shape (M,)
        Initial model accuracies. In PoolEval these come from a labeled train/meta
        dataset and are not re-fused after initialization.
    beta_init : float, optional
        Initial pseudo-label quality. If omitted, use the expected agreement quality
        under ``alpha_init`` and the binary conditional-agreement model.

    Returns a dictionary containing the fitted ``alpha``, ``beta``, posterior
    correctness matrix ``tau``, observed log-likelihood trace, and iteration count.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.size == 0:
        raise ValueError("C must be a non-empty 2D agreement matrix")
    if not np.all((C == 0.0) | (C == 1.0)):
        raise ValueError("C must contain only binary values")

    M, N = C.shape
    alpha = _clip_probability(alpha_init, eps)
    if alpha.shape != (M,):
        raise ValueError(f"alpha_init must have shape ({M},)")

    if beta_init is None:
        # Expected probability that pseudo-label and truth agree, initialized with
        # P(Z_ji=1)=alpha_j and the observed C. This is exactly the beta M-step with
        # tau initialized to alpha, not a target-label estimate.
        tau0 = np.broadcast_to(alpha[:, None], (M, N))
        beta = float(np.mean(tau0 * C + (1.0 - tau0) * (1.0 - C)))
    else:
        beta = float(beta_init)
    beta = float(_clip_probability(beta, eps))

    trace = []
    tau = np.empty_like(C)
    for iteration in range(1, int(max_iters) + 1):
        old_alpha = alpha.copy()
        old_beta = beta

        # E-step, Eq. (4): posterior P(Z_ji=1 | C_ji, alpha_j, beta).
        logit_alpha = np.log(alpha / (1.0 - alpha))[:, None]
        logit_beta = np.log(beta / (1.0 - beta))
        logits = logit_alpha + (2.0 * C - 1.0) * logit_beta
        tau = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))

        # M-step: exact closed-form maximizers from the new formulation.
        alpha = _clip_probability(tau.mean(axis=1), eps)
        beta = float(_clip_probability(
            np.mean(tau * C + (1.0 - tau) * (1.0 - C)), eps
        ))

        # Marginal observed-data likelihood, useful for verifying EM monotonicity.
        p_c1 = alpha[:, None] * beta + (1.0 - alpha[:, None]) * (1.0 - beta)
        p_obs = np.where(C == 1.0, p_c1, 1.0 - p_c1)
        trace.append(float(np.log(np.clip(p_obs, eps, 1.0)).sum()))

        delta = max(float(np.max(np.abs(alpha - old_alpha))),
                    abs(beta - old_beta))
        if delta < tol:
            break

    # Recompute tau at the returned parameters so all outputs describe one state.
    logits = (np.log(alpha / (1.0 - alpha))[:, None]
              + (2.0 * C - 1.0) * np.log(beta / (1.0 - beta)))
    tau = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
    sigma = np.sqrt(np.clip(alpha * (1.0 - alpha), eps, None) / N)
    return dict(alpha=alpha, beta=beta, tau=tau, a_sigma=sigma,
                log_likelihood=np.asarray(trace), n_iters=iteration)


class NewFormulationPoolEval:
    """Two-stage estimator: old scoring pseudo-labels, then closed-form EM."""

    def __init__(self, cfg, max_iters=200, tol=1e-8, beta_init=None):
        self.cfg = cfg
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.beta_init = beta_init

    def evaluate(self, run, obs=None, pseudo_out=None, alpha_init=None):
        """Estimate model accuracy using fixed old-method pseudo-labels.

        ``run.prior`` is expected to contain per-model accuracy measured on the
        labeled train/meta dataset. The test labels are never consumed here.
        ``pseudo_out`` can be supplied when the caller already evaluated the old
        method, avoiding duplicate work and guaranteeing a paired comparison.
        """
        cfg = self.cfg
        if obs is None:
            obs = kernelmod.apply(run.true_class, cfg, level=cfg.kernel_level,
                                  rng=np.random.default_rng(cfg.seed + 7))
        if pseudo_out is None:
            pseudo_out = PoolEval(cfg).evaluate(run, obs=obs)

        post = pseudo_out["latent_post"]
        if len(post) != obs.shape[1]:
            raise ValueError("pseudo-label posterior length does not match obs")
        pseudo_label = np.asarray([max(p, key=p.get) for p in post], dtype=obs.dtype)
        C = (obs == pseudo_label[None, :]).astype(float)

        init = run.prior if alpha_init is None else alpha_init
        fitted = agreement_em(C, init, beta_init=self.beta_init,
                              max_iters=self.max_iters, tol=self.tol)
        acc = fitted.pop("alpha")
        order = np.argsort(-acc)
        z = 1.645
        half = z * fitted["a_sigma"]
        fitted.update(acc=acc, ranking=order, pseudo_label=pseudo_label, C=C,
                      obs=obs, pseudo_source_acc=pseudo_out["acc"],
                      lo=np.clip(acc - half, 0.0, 1.0),
                      hi=np.clip(acc + half, 0.0, 1.0))
        return fitted
