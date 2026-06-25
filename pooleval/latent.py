"""Correlation-aware latent-correctness EM (the core of PoolEval).

Maximizes the anchored posterior of Eq. (objective): an agreement likelihood that
scores every model against a SHARED latent answer (IRT ability/difficulty), a
provenance-group shared-error factor that down-weights within-group agreement, a
seen-prior anchor, and an execution verifier -- fused by precision (inverse
variance). Returns per-model accuracy, difficulties, group loadings, the soft
latent posteriors, the effective-independent-model count M_eff, and a collusion
flag.

Flags on Config toggle the requirement components for ablations:
  use_prior (req2), use_verifier (req2), use_correlation (req3),
  fusion in {precision, prior_only, agreement_only}.
"""
import numpy as np
from .data import sigmoid


def _logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def run_em(obs, run, cfg, verbose=False, init=None, max_iters=None, tol=1e-4):
    M, N = obs.shape
    group = run.group
    G = run.n_groups
    prior = run.prior
    prior_sigma = run.prior_sigma

    # init
    init = init or {}
    a = np.asarray(init.get("a", prior.copy() if cfg.use_prior else np.full(M, 0.6)),
                   dtype=float).copy()
    b = np.asarray(init.get("b", np.zeros(N)), dtype=float).copy()
    u = np.asarray(init.get("u", np.zeros(G)), dtype=float).copy()

    # precompute per-item candidate classes
    item_classes = [np.unique(obs[:, i]) for i in range(N)]
    gmask = [group == g for g in range(G)]

    agree = np.zeros((M, N))
    n_iters = cfg.em_iters if max_iters is None else int(max_iters)
    n_iters = max(0, n_iters)
    for it in range(n_iters):
        a_prev = a.copy()
        b_prev = b.copy()
        u_prev = u.copy()
        # reliability weight = estimated accuracy, always positive and bounded in
        # (0,1). Bounded (not logit) avoids the high-a -> peaked-consensus -> higher-a
        # runaway; keeping it strictly positive avoids the opposite collapse where
        # sub-0.5 models get zero weight and the consensus loses all information.
        w = np.clip(a, 0.05, 0.99)

        # ---------- E-step (vectorized per item over models) ----------
        # Each model's contribution to its own voted class is its reliability w_m,
        # down-weighted by the provenance-group discount 1/(1+u_g*(n_{g,k}-1)) where
        # n_{g,k} counts same-group models voting the same class -> near-clones that
        # collude on a wrong answer count as (almost) one vote.
        latent_post = []
        for i in range(N):
            col = obs[:, i]
            # count same-(group,class) members for each model
            n_gk = np.ones(M)
            if cfg.use_correlation:
                key = {}
                for m in range(M):
                    key.setdefault((group[m], col[m]), []).append(m)
                for members in key.values():
                    if len(members) > 1:
                        for m in members:
                            n_gk[m] = len(members)
            disc = 1.0 / (1.0 + u[group] * (n_gk - 1)) if cfg.use_correlation else 1.0
            contrib = w * disc
            # aggregate contributions per class
            ks = item_classes[i]
            score = {int(k): 0.0 for k in ks}
            for m in range(M):
                score[int(col[m])] += contrib[m]
            if cfg.use_verifier:
                vg = int(run.verifier_guess[i])
                if vg in score:
                    score[vg] += cfg.verifier_strength
            sc = np.array([score[int(k)] for k in ks])
            sc = sc - sc.max()
            p = np.exp(sc); p /= p.sum()
            post = {int(k): pv for k, pv in zip(ks, p)}
            latent_post.append(post)
            agree[:, i] = np.array([post[int(c)] for c in col])

        # ---------- M-step ----------
        # accuracy = fraction of items whose answer matches the HARD latent estimate
        # (argmax posterior). Hard matching keeps the absolute level calibrated
        # (soft averaging shrinks every model toward 0.5); the latent estimate is the
        # anchored, group-discounted consensus, so it beats plain majority exactly on
        # colluded items -- the source of PoolEval's accuracy edge over B2/B3.
        latent_hat = np.array([max(post, key=post.get) for post in latent_post])
        correct = (obs == latent_hat[None, :]).astype(float)        # [M,N]
        a_agree = correct.mean(axis=1)
        a_sig = np.sqrt(np.clip(a_agree * (1 - a_agree), 1e-4, None) / N)
        if not cfg.use_prior or cfg.fusion == "agreement_only":
            a = a_agree.copy()
        elif cfg.fusion == "prior_only":
            a = prior.copy()
        else:  # precision (inverse-variance) fusion
            vp = prior_sigma ** 2 + 1e-6
            va = a_sig ** 2 + 1e-6
            a = (prior / vp + a_agree / va) / (1.0 / vp + 1.0 / va)
        a = np.clip(a, 0.02, 0.98)

        # difficulty from item-level agreement
        item_agree = correct.mean(axis=0)
        b = -_logit(np.clip(item_agree, 0.05, 0.95)) * 0.5

        # group loadings from within- vs cross-group wrong-answer agreement
        if cfg.use_correlation:
            u = _estimate_loadings(obs, agree, group, G)

        if verbose and (it % 10 == 0 or it == cfg.em_iters - 1):
            print(f"  [EM] it {it:2d}  mean a={a.mean():.3f}")

        delta = max(np.max(np.abs(a - a_prev)),
                    np.max(np.abs(b - b_prev)),
                    np.max(np.abs(u - u_prev)))
        if delta < tol:
            n_iters = it + 1
            break
    else:
        n_iters = n_iters

    cbar, Meff = _effective_models(obs, agree, M, group)
    flag = _collusion_flag(obs, group, run)
    return dict(acc=a, b=b, u=u, agree=agree, latent_post=latent_post,
                cbar=cbar, Meff=Meff, collusion=flag, a_sigma=a_sig,
                n_iters=n_iters)


def _estimate_loadings(obs, agree, group, G):
    """Excess within-group agreement on WRONG answers => shared-error loading."""
    M, N = obs.shape
    err = agree < 0.5      # model likely wrong here
    u = np.zeros(G)
    for g in range(G):
        idx = np.where(group == g)[0]
        if len(idx) < 2:
            continue
        within = cross = 0.0
        wc = cc = 0
        for a_i in range(len(idx)):
            for b_i in range(a_i + 1, len(idx)):
                ma, mb = idx[a_i], idx[b_i]
                both_err = err[ma] & err[mb]
                same = (obs[ma] == obs[mb]) & both_err
                within += same.sum(); wc += both_err.sum()
        within_rate = within / wc if wc else 0.0
        # cross-group baseline: same model vs other-group models
        others = np.where(group != g)[0]
        for ma in idx:
            for mb in others:
                both_err = err[ma] & err[mb]
                same = (obs[ma] == obs[mb]) & both_err
                cross += same.sum(); cc += both_err.sum()
        cross_rate = cross / cc if cc else 0.0
        u[g] = np.clip((within_rate - cross_rate) * 5.0, 0.0, 10.0)
    return u


def _effective_models(obs, agree, M, group=None):
    """Effective-independent-model count = survey-sampling design effect for the
    mean PROVENANCE error-correlation. Collusion shows up as near-clones giving the
    SAME wrong answer (not merely co-failing on hard items), so we measure, per
    pair, the shared-wrong-answer rate among items where both err, and subtract the
    cross-group baseline (coincidental shared wrongs). The provenance excess defines
    c_bar; M_eff = M / (1 + c_bar (M-1)) in [1, M]."""
    wrong = agree < 0.5

    def shared_rate(a, b):
        both = wrong[a] & wrong[b]
        nb = both.sum()
        if nb == 0:
            return 0.0
        return float(((obs[a] == obs[b]) & both).sum() / nb)

    within, cross = [], []
    for a in range(M):
        for b in range(a + 1, M):
            r = shared_rate(a, b)
            (within if (group is not None and group[a] == group[b]) else cross
             ).append(r)
    baseline = np.mean(cross) if cross else 0.0
    npairs = M * (M - 1) / 2
    excess = sum(max(0.0, r - baseline) for r in within)  # provenance excess only
    cbar = float(np.clip(excess / npairs, 0.0, 1.0)) if npairs else 0.0
    Meff = M / (1.0 + cbar * (M - 1))
    return cbar, Meff


def _collusion_flag(obs, group, run):
    """Within-group agreement high but cross-group + verifier low."""
    M, N = obs.shape
    within = cross = wc = cc = 0
    for a in range(M):
        for b in range(a + 1, M):
            same = (obs[a] == obs[b]).sum()
            if group[a] == group[b]:
                within += same; wc += N
            else:
                cross += same; cc += N
    wr = within / wc if wc else 0
    cr = cross / cc if cc else 0
    return bool(wr > 0.5 and cr < 0.3 and run.verifier_correct.mean() < 0.5)
