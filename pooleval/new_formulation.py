"""Closed-form pseudo-label agreement EM from ``new_formulation/``.

This estimator intentionally lives beside, rather than replacing, :class:`PoolEval`.
It uses the existing estimator only to construct one pseudo-label per item from the
graded execution kernel, meta-dataset accuracy priors, provenance discount, and
execution verifier.  Given those fixed pseudo-labels it fits the binary model from
the new formulation with its exact closed-form EM updates.
"""
import numpy as np
from scipy.optimize import minimize_scalar

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


def collision_agreement_em(C, group, gamma_group, alpha_prior,
                           alpha_strength, beta_init=0.7, beta_strength=0.0,
                           max_iters=200,
                           tol=1e-8, eps=1e-6):
    """Collision-aware binary EM for multiclass wrong results.

    ``gamma_group[g]`` is fixed from labeled source/meta data. ``alpha_prior`` is
    retained in the objective as a Beta prior with effective sample size
    ``alpha_strength``; it is not merely an initializer. The alpha M-step remains
    closed form. The beta M-step is a bounded one-dimensional maximization because
    the original beta closed form is invalid once gamma is introduced.
    """
    C = np.asarray(C, dtype=float)
    group = np.asarray(group, dtype=int)
    gamma_group = _clip_probability(gamma_group, eps)
    prior = _clip_probability(alpha_prior, eps)
    M, N = C.shape
    if group.shape != (M,) or prior.shape != (M,):
        raise ValueError("group and alpha_prior must have one entry per model")
    if group.min() < 0 or group.max() >= len(gamma_group):
        raise ValueError("gamma_group does not cover every provenance group")

    strength = np.broadcast_to(np.asarray(alpha_strength, dtype=float), (M,))
    if np.any(strength < 0):
        raise ValueError("alpha_strength must be non-negative")
    alpha = prior.copy()
    beta = float(_clip_probability(beta_init, eps))
    beta_prior = beta
    beta_strength = float(beta_strength)
    if beta_strength < 0:
        raise ValueError("beta_strength must be non-negative")
    gamma = gamma_group[group][:, None]
    trace = []

    def log_likelihood(a, b):
        p1 = a[:, None] * b + (1.0 - a[:, None]) * (1.0 - b) * gamma
        obs = np.where(C == 1.0, p1, 1.0 - p1)
        # Beta(a0,b0) with a0=1+s*pi, b0=1+s*(1-pi).
        anchor = np.sum(strength * (prior * np.log(a)
                        + (1.0 - prior) * np.log(1.0 - a)))
        beta_anchor = beta_strength * (beta_prior * np.log(b)
                      + (1.0 - beta_prior) * np.log(1.0 - b))
        return float(np.log(np.clip(obs, eps, 1.0)).sum() + anchor + beta_anchor)

    for iteration in range(1, int(max_iters) + 1):
        old_alpha = alpha.copy()
        old_beta = beta

        like_z1 = np.where(C == 1.0, beta, 1.0 - beta)
        p_z0_c1 = (1.0 - beta) * gamma
        like_z0 = np.where(C == 1.0, p_z0_c1, 1.0 - p_z0_c1)
        numer = alpha[:, None] * like_z1
        tau = numer / np.clip(numer + (1.0 - alpha[:, None]) * like_z0,
                              eps, None)

        alpha = _clip_probability(
            (tau.sum(axis=1) + strength * prior) / (N + strength), eps
        )

        def negative_q(candidate):
            b = float(candidate)
            z1 = C * np.log(b) + (1.0 - C) * np.log(1.0 - b)
            wrong_agree = np.clip((1.0 - b) * gamma, eps, 1.0 - eps)
            z0 = C * np.log(wrong_agree) + (1.0 - C) * np.log(1.0 - wrong_agree)
            anchor = beta_strength * (beta_prior * np.log(b)
                     + (1.0 - beta_prior) * np.log(1.0 - b))
            return -float(np.sum(tau * z1 + (1.0 - tau) * z0) + anchor)

        opt = minimize_scalar(negative_q, bounds=(eps, 1.0 - eps),
                              method="bounded", options={"xatol": 1e-10})
        beta = float(opt.x)
        trace.append(log_likelihood(alpha, beta))
        delta = max(float(np.max(np.abs(alpha - old_alpha))),
                    abs(beta - old_beta))
        if delta < tol:
            break

    like_z1 = np.where(C == 1.0, beta, 1.0 - beta)
    p_z0_c1 = (1.0 - beta) * gamma
    like_z0 = np.where(C == 1.0, p_z0_c1, 1.0 - p_z0_c1)
    numer = alpha[:, None] * like_z1
    tau = numer / np.clip(numer + (1.0 - alpha[:, None]) * like_z0, eps, None)
    sigma = np.sqrt(np.clip(alpha * (1.0 - alpha), eps, None)
                    / (N + strength))
    return dict(alpha=alpha, beta=beta, tau=tau, gamma_group=gamma_group,
                a_sigma=sigma, log_posterior=np.asarray(trace), n_iters=iteration)


class CollisionAwareNewFormulationPoolEval:
    """Case-3 correction with source-estimated collision rates and alpha anchors."""

    def __init__(self, cfg, gamma_group, alpha_strength, beta_init=0.7,
                 beta_strength=0.0,
                 max_iters=200, tol=1e-8):
        self.cfg = cfg
        self.gamma_group = np.asarray(gamma_group, dtype=float)
        self.alpha_strength = alpha_strength
        self.beta_init = float(beta_init)
        self.beta_strength = float(beta_strength)
        self.max_iters = int(max_iters)
        self.tol = float(tol)

    def evaluate(self, run, obs=None, pseudo_out=None):
        if obs is None:
            obs = kernelmod.apply(run.true_class, self.cfg,
                                  level=self.cfg.kernel_level,
                                  rng=np.random.default_rng(self.cfg.seed + 7))
        if pseudo_out is None:
            pseudo_out = PoolEval(self.cfg).evaluate(run, obs=obs)
        pseudo_label = np.asarray(
            [max(post, key=post.get) for post in pseudo_out["latent_post"]],
            dtype=obs.dtype,
        )
        C = (obs == pseudo_label[None, :]).astype(float)
        fitted = collision_agreement_em(
            C, run.group, self.gamma_group, run.prior, self.alpha_strength,
            beta_init=self.beta_init, beta_strength=self.beta_strength,
            max_iters=self.max_iters, tol=self.tol,
        )
        acc = fitted.pop("alpha")
        fitted.update(acc=acc, ranking=np.argsort(-acc), C=C, obs=obs,
                      pseudo_label=pseudo_label,
                      pseudo_source_acc=pseudo_out["acc"])
        return fitted
