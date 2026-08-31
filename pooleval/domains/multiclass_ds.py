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


def one_coin_ds(obs, n_classes, max_iters=200, tol=1e-8, eps=1e-6):
    """Classic one-coin Dawid--Skene. Closed-form E- and M-steps."""
    obs = np.asarray(obs, dtype=np.int64)
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


def full_ds(obs, n_classes, max_iters=200, tol=1e-8, eps=1e-6, smoothing=1.0):
    """Full Dawid--Skene: one confusion matrix per model. Closed-form M-step."""
    obs = np.asarray(obs, dtype=np.int64)
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
    """-> {label: [M] accuracy} for the two textbook multiclass estimators."""
    pred = np.asarray(pool["pred"])
    K = int(np.asarray(pool["n_classes"]))
    return {"DS one-coin (exact)": one_coin_ds(pred, K)["acc"],
            "DS full (confusion)": full_ds(pred, K)["acc"]}
