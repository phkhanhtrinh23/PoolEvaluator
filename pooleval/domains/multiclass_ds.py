"""Textbook multiclass Dawid--Skene for closed label spaces, with exact closed forms.

These are the estimators the collision-aware BINARY model is compared against in
`docs/multiclass_ds.md`.  Both run on RAW predicted labels (0..K-1), never on the
repo's gold-relative `true_class` encoding: that encoding is a per-item bijection,
which preserves equality (all `latent.py` needs) but destroys class identity
ACROSS items, which a confusion matrix depends on.

ONE-COIN.  P(obs[j,i]=k | z_i=c) = a_j if k==c else (1-a_j)/(K-1).
  E-step   log t_i(c) = log p_c + sum_j 1[obs=c] * log( a_j (K-1) / (1-a_j) )
                        (+ a c-independent constant that the softmax removes)
  M-step   a_j = (1/N) sum_i t_i(obs[j,i]),      p_c = mean_i t_i(c)
Both are closed form.  The model has no gamma: it *implies* one, namely
P(two wrong answers coincide) = 1/(K-1), which `results/ds_assumption.json`
measures to be wrong by 2.8-6.4x on these pools.

FULL.  P(obs[j,i]=k | z_i=c) = pi_j[c,k], a whole confusion matrix per model.
  E-step   log t_i(c) = log p_c + sum_j log pi_j[c, obs[j,i]]
  M-step   pi_j[c,k] = sum_i t_i(c) 1[obs=k] / sum_i t_i(c)      (row-normalised counts)
Also closed form.  This RELAXES uniform errors -- it can learn that class 4 is
usually mistaken for 9 -- but it still assumes models are conditionally
independent given z, so it cannot represent models failing TOGETHER.
"""
import numpy as np


def _softmax_rows(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def _accuracy_from_posterior(post, obs):
    """a_j = (1/N) sum_i t_i(obs[j,i]) -- posterior probability model j was right."""
    N = obs.shape[1]
    return np.array([post[np.arange(N), obs[j]].mean() for j in range(obs.shape[0])])


# Above this many bytes we refuse to build a dense structure. A full confusion
# matrix needs M*K*K floats, which is 20 GB at M=12, K=14541 (knowledge-graph
# completion) -- that is not a tuning problem, it is the estimator being
# inapplicable, and callers should report it as such rather than crash.
DENSE_BUDGET_BYTES = 2 * 1024 ** 3


def dense_cost(M, K, per_model_matrix=True):
    """Bytes a dense implementation would need, for the applicability check."""
    return int(M) * int(K) * (int(K) if per_model_matrix else 1) * 8


def one_coin_ds(obs, n_classes, max_iters=200, tol=1e-8, eps=1e-6):
    """Classic one-coin Dawid--Skene. Closed-form E- and M-steps.

    Dispatches to a sparse implementation when K is too large to hold an
    [M, N, K] indicator; see `one_coin_ds_sparse` for what that costs.
    """
    obs = np.asarray(obs, dtype=np.int64)
    if dense_cost(obs.shape[0] * obs.shape[1], n_classes,
                  per_model_matrix=False) > DENSE_BUDGET_BYTES:
        return one_coin_ds_sparse(obs, n_classes, max_iters=max_iters,
                                  tol=tol, eps=eps)
    M, N = obs.shape
    K = int(n_classes)
    onehot = np.zeros((M, N, K), dtype=np.float64)
    for j in range(M):
        onehot[j, np.arange(N), obs[j]] = 1.0

    a = np.full(M, 0.7)
    p = np.full(K, 1.0 / K)
    trace = []
    for it in range(1, max_iters + 1):
        a_prev = a.copy()
        # E-step: only the "model j voted for c" term depends on c.
        w = np.log(np.clip(a * (K - 1) / (1.0 - a), eps, None))       # [M]
        logits = np.log(np.clip(p, eps, None))[None, :] + np.tensordot(w, onehot, axes=(0, 0))
        post = _softmax_rows(logits)                                   # [N,K]
        # M-step: both are ratios of expected counts.
        a = np.clip(_accuracy_from_posterior(post, obs), eps, 1.0 - eps)
        p = np.clip(post.mean(axis=0), eps, None); p /= p.sum()
        # observed-data log-likelihood, for monotonicity checks
        ll = np.log(np.clip(p, eps, None))[None, :] + np.tensordot(
            np.log(np.clip(a * (K - 1) / (1.0 - a), eps, None)), onehot, axes=(0, 0))
        trace.append(float(np.log(np.exp(ll - ll.max(1, keepdims=True)).sum(1)).sum()
                           + ll.max(1).sum()))
        if np.max(np.abs(a - a_prev)) < tol:
            break
    return dict(acc=a, post=post, prior=p, n_iters=it,
                latent_hat=post.argmax(1), log_likelihood=np.asarray(trace))


def one_coin_ds_sparse(obs, n_classes, max_iters=200, tol=1e-8, eps=1e-6):
    """One-coin Dawid--Skene for a label space too large to enumerate.

    The E-step of section 3 is

        log t_i(c) = log p_c + sum_{j : obs[j,i] = c} w_j,
        w_j = log[ a_j (K-1) / (1 - a_j) ],

    so only the classes some model actually voted for get a contribution; every
    other class carries the bare prior. That means the posterior never has to be
    materialised over all K classes -- per item it is fully described by the <= M
    distinct votes plus one aggregate for the K - n_cand unvoted classes, which
    still carry real mass and cannot simply be dropped.

    ONE SIMPLIFICATION, STATED PLAINLY. The class prior p_c is held UNIFORM at
    1/K rather than estimated. Estimating 14541 class frequencies from 20k items
    is hopeless anyway, and it is what lets the unvoted classes collapse into a
    single count. The accuracy M-step is unchanged and still exact.
    """
    obs = np.asarray(obs, dtype=np.int64)
    M, N = obs.shape
    K = int(n_classes)
    a = np.full(M, 0.7)
    for it in range(1, max_iters + 1):
        a_prev = a.copy()
        w = np.log(np.clip(a * (K - 1) / (1.0 - a), eps, None))          # [M]
        tau_at_obs = np.zeros((M, N))
        latent = np.empty(N, dtype=np.int64)
        for i in range(N):
            col = obs[:, i]
            cands, inv = np.unique(col, return_inverse=True)
            score = np.bincount(inv, weights=w, minlength=len(cands))     # sum_j w_j
            ex = np.exp(score - score.max())
            # unvoted classes: (K - n_cand) of them, each at the bare prior
            unvoted = (K - len(cands)) * np.exp(-score.max())
            Z = ex.sum() + unvoted
            post = ex / Z
            tau_at_obs[:, i] = post[inv]
            latent[i] = cands[int(np.argmax(post))]
        a = np.clip(tau_at_obs.mean(axis=1), eps, 1.0 - eps)
        if np.max(np.abs(a - a_prev)) < tol:
            break
    return dict(acc=a, post=None, prior=None, n_iters=it, latent_hat=latent,
                log_likelihood=np.asarray([]), sparse=True)


def full_ds(obs, n_classes, max_iters=200, tol=1e-8, eps=1e-6, smoothing=1.0):
    """Full Dawid--Skene: one confusion matrix per model. Closed-form M-step.

    Raises MemoryError when the confusion matrices cannot be held; a K x K table
    per model is O(K^2) and there is no sparse rescue, because the whole point of
    the estimator is to fill in every (true class, predicted class) cell.
    """
    obs = np.asarray(obs, dtype=np.int64)
    need = dense_cost(obs.shape[0], n_classes)
    if need > DENSE_BUDGET_BYTES:
        raise MemoryError(
            f"full Dawid--Skene needs {need / 1024**3:.1f} GB for "
            f"{obs.shape[0]} models x {n_classes}^2 confusion cells "
            f"(budget {DENSE_BUDGET_BYTES / 1024**3:.1f} GB). The estimator is "
            f"not applicable at this label-space size.")
    M, N = obs.shape
    K = int(n_classes)
    onehot = np.zeros((M, N, K), dtype=np.float64)
    for j in range(M):
        onehot[j, np.arange(N), obs[j]] = 1.0

    # init from majority vote, the standard Dawid--Skene initialisation
    votes = onehot.sum(axis=0)
    post = votes / np.clip(votes.sum(1, keepdims=True), eps, None)
    p = np.full(K, 1.0 / K)
    pi = np.zeros((M, K, K))
    for it in range(1, max_iters + 1):
        post_prev = post.copy()
        # M-step: row-normalised expected counts.
        for j in range(M):
            counts = post.T @ onehot[j] + smoothing                   # [K,K]
            pi[j] = counts / counts.sum(axis=1, keepdims=True)
        p = np.clip(post.mean(axis=0), eps, None); p /= p.sum()
        # E-step.
        logits = np.log(p)[None, :].repeat(N, axis=0)
        logpi = np.log(np.clip(pi, eps, None))
        for j in range(M):
            logits += logpi[j][:, obs[j]].T                           # [N,K]
        post = _softmax_rows(logits)
        if np.max(np.abs(post - post_prev)) < tol:
            break
    acc = np.clip(_accuracy_from_posterior(post, obs), eps, 1.0 - eps)
    return dict(acc=acc, post=post, prior=p, confusion=pi, n_iters=it,
                latent_hat=post.argmax(1))


def implied_gamma(n_classes):
    """The collision rate one-coin DS assumes without saying so."""
    return 1.0 / (int(n_classes) - 1)


def ds_estimates(pool):
    """-> {label: [M] accuracy}. Omits full DS wherever it is not well posed.

    TWO distinct reasons it can be omitted, and they are not the same failure:

    1. K is too large to hold the confusion matrices (knowledge-graph completion).
       A memory limit -- the model is well defined, the machine is too small.

    2. The label space is UNBOUNDED, so class ids do not persist across items
       (image captioning, Text2SQL). This is worse than a memory limit: the
       estimator is meaningless, not merely expensive. pi_j[c,k] asks "when the
       truth is class c, how often does model j say k", but if "class 3" names a
       different caption cluster on every image there is no such quantity. The
       ids in that case come from per-item clustering, whose numbering depends on
       the order captions are listed in -- permuting the pool changes full DS's
       answer by ~2x on identical data, while one-coin (which only tests ids for
       equality) is stable. Callers flag this with pool["unbounded"] = True.
    """
    pred = np.asarray(pool["pred"])
    K = int(np.asarray(pool["n_classes"]))
    out = {"DS one-coin (exact)": one_coin_ds(pred, K)["acc"]}
    if bool(pool.get("unbounded", False)):
        print("  [DS full omitted] label space is unbounded: class ids are "
              "per-item and do not persist across items, so a confusion matrix "
              "has no meaning here (its answer depends on cluster numbering).")
        return out
    try:
        out["DS full (confusion)"] = full_ds(pred, K)["acc"]
    except MemoryError as e:
        print(f"  [DS full omitted] {e}")
    return out
