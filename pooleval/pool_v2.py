"""The version-2 model of ``proof/Trinh_pool_version_2.tex``: three closed-form M-steps.

Differences from :mod:`pooleval.validated_em`, all three substantive:

1.  The pseudo-label is LEAVE-ONE-OUT.  ``yhat[i, j]`` is the majority vote over every
    model EXCEPT ``j``, so ``C[i, j]`` never compares model ``j`` against a consensus it
    helped to form.  This is the standard remedy for the self-reference that makes
    "pseudo-label correct" and "model correct" dependent.
2.  ``beta`` is PER MODEL.  ``beta[j] = P(C = 1 | Z = 1)`` is model j's agreement rate when
    it is right, rather than one scalar shared by the pool.
3.  ``gamma`` is FITTED, not measured.  It has its own closed-form M-step, so the model
    needs no labeled split at all -- it is fully unsupervised.

Model, for each item i and model j:

    Z[i,j] ~ Bernoulli(alpha[j])
    C[i,j] | Z[i,j]=1 ~ Bernoulli(beta[j])
    C[i,j] | Z[i,j]=0 ~ Bernoulli(gamma[j])

with the three M-steps

    alpha[j] = mean_i q[i,j]
    beta[j]  = sum_i q[i,j] C[i,j] / sum_i q[i,j]
    gamma[j] = sum_i (1-q[i,j]) C[i,j] / sum_i (1-q[i,j])
"""
import numpy as np

EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def loo_agreement(obs):
    """``C[j, i] = 1(r_i^j == majority vote of the OTHER models)``.

    A negative class id is an execution error unique to one cell, so it can never be the
    majority answer and never matches another cell; such cells are excluded from the vote
    and score ``C = 0``.
    """
    obs = np.asarray(obs)
    M, N = obs.shape
    C = np.zeros((M, N), dtype=float)
    for i in range(N):
        col = obs[:, i]
        valid = col >= 0
        for j in range(M):
            others = col[valid & (np.arange(M) != j)]
            if others.size == 0:
                continue                      # no one else answered: no evidence
            vals, cnt = np.unique(others, return_counts=True)
            winner = vals[cnt.argmax()]       # ties -> smallest id, deterministic
            C[j, i] = float(col[j] == winner)
    return C


def em_v2(C, init=None, max_iters=500, tol=1e-10, seed=0,
          prior=None, prior_strength=0.0):
    """Fit ``(alpha, beta, gamma)`` by the three closed-form M-steps.

    Returns a dict with the parameters, the responsibilities ``q``, the observed-data
    log-likelihood trace and the iteration count.  ``init`` may supply any of
    ``alpha``/``beta``/``gamma``; anything absent is drawn from the default start
    (alpha 0.7, beta 0.9, gamma 0.3), which encodes the model's own ordering assumption
    ``beta > gamma``.

    ``prior``/``prior_strength`` add the Beta anchor of ``validated_em``:

        alpha[j] = (sum_i q[i,j] + s_j pi_j) / (n + s_j)

    The anchor does not make the model identifiable -- with alpha pinned, a_j still leaves
    one equation in the two unknowns (beta, gamma).  It pins the ONE coordinate that is
    reported, which is what the restart spread below actually measures.
    """
    C = np.asarray(C, dtype=float)
    M, N = C.shape
    init = dict(init or {})
    alpha = _clip(np.broadcast_to(init.get("alpha", 0.7), (M,)).astype(float)).copy()
    beta = _clip(np.broadcast_to(init.get("beta", 0.9), (M,)).astype(float)).copy()
    gamma = _clip(np.broadcast_to(init.get("gamma", 0.3), (M,)).astype(float)).copy()

    trace = []
    for it in range(1, int(max_iters) + 1):
        a0, b0, g0 = alpha.copy(), beta.copy(), gamma.copy()

        # E-step: q[j,i] = P(Z=1 | C)
        like1 = alpha[:, None] * np.where(C == 1.0, beta[:, None], 1.0 - beta[:, None])
        like0 = (1.0 - alpha[:, None]) * np.where(C == 1.0, gamma[:, None],
                                                  1.0 - gamma[:, None])
        denom = np.clip(like1 + like0, EPS, None)
        q = like1 / denom
        trace.append(float(np.log(denom).sum()))

        # M-step: all three in closed form
        if prior is None:
            alpha = _clip(q.mean(axis=1))
        else:
            sj = np.broadcast_to(np.asarray(prior_strength, float), (M,))
            alpha = _clip((q.sum(axis=1) + sj * np.asarray(prior, float)) / (N + sj))
        beta = _clip((q * C).sum(axis=1) / np.clip(q.sum(axis=1), EPS, None))
        gamma = _clip(((1.0 - q) * C).sum(axis=1)
                      / np.clip((1.0 - q).sum(axis=1), EPS, None))

        if max(np.abs(alpha - a0).max(), np.abs(beta - b0).max(),
               np.abs(gamma - g0).max()) < tol:
            break
    return dict(alpha=alpha, beta=beta, gamma=gamma, q=q, acc=alpha,
                log_likelihood=np.asarray(trace), n_iters=it)
