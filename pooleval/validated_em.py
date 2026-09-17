"""Validated-EM: the labeled-statistics reformulation plus the i-EM judge loop.

This module implements the variant of ``Trinh_proof.tex`` in which the two nuisance
quantities are no longer free parameters fitted on unlabeled target data but are
MEASURED on a pre-built labeled split (the train side of the target dataset), and in
which expert validation is spent by information gain, following Hung et al.,
*Minimizing Efforts in Validating Crowd Answers* (i-EM).

Two things change relative to :mod:`pooleval.new_formulation`.

1.  ``e`` -- the pairwise correlated-error matrix.  ``e[j, k]`` is the fraction of
    labeled items on which models ``j`` and ``k`` return the SAME answer and that
    answer is WRONG.  It replaces the hand-declared provenance grouping: instead of
    asserting that two models are near-clones because they share a base checkpoint,
    we measure how often they actually fail together in the same way.  It enters the
    latent-answer posterior as a per-vote discount (see :func:`latent_posterior`),
    and it reduces EXACTLY to the group discount of :mod:`pooleval.latent` when the
    excess collision rate is constant within a group and zero across groups.

2.  ``gamma`` -- redefined as ``gamma[j] = P(C = 0 | Z = 0)``, the probability that a
    WRONG model disagrees with the pseudo-label.  In the collision formulation this
    quantity was ``d_j(beta) = 1 - (1 - beta) * gamma_j^{collision}``, a function of
    the free parameter ``beta``.  Here it is measured once on the labeled split and
    held fixed.  Two consequences:

      * the alpha M-step is unchanged, exactly as in the .tex (the collision terms
        never touched ``P(Z | alpha)``), and
      * the beta M-step becomes closed form again.  Freezing the ``Z = 0`` branch
        removes ``beta`` from it, so ``Q_beta`` collapses to a weighted Bernoulli
        log-likelihood whose maximiser is a ratio of expected counts.  The collision
        model needed a bounded 1-D numerical maximisation here; this one does not.

Both statistics are refreshed as expert validation reveals target-domain labels.

Conventions.  ``obs`` is an ``[M, N]`` matrix of result-equivalence class ids: ``M``
models, ``N`` items.  On a LABELED split class ``0`` means "matches gold", a positive
id is a distinct wrong result shared across models, and a negative id is an execution
error unique to that cell (so it can never collide).  The estimator itself never
reads class ``0`` as privileged -- only the labeled-statistics helpers do.
"""
import numpy as np

EPS = 1e-6


def _clip(p, eps=EPS):
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


# --------------------------------------------------------------------------- #
#  1.  Statistics measured on a pre-built labeled split                        #
# --------------------------------------------------------------------------- #
def wrong_collision_counts(true_class):
    """Raw counts behind ``e`` on a labeled split.

    Returns ``(same_wrong, n_items)`` where ``same_wrong[j, k]`` counts items on
    which models ``j`` and ``k`` produced the same answer and that answer was wrong.
    A negative class id is an execution error unique to one cell, so requiring a
    strictly positive id is what stops two independent crashes from counting as a
    collision.  The diagonal counts model ``j``'s own shared-able wrong answers.
    """
    tc = np.asarray(true_class)
    wrong = tc > 0                                     # [M, N]
    same = tc[:, None, :] == tc[None, :, :]            # [M, M, N]
    both_wrong = wrong[:, None, :] & wrong[None, :, :]
    return (same & both_wrong).sum(axis=2).astype(float), int(tc.shape[1])


def wrong_collision_matrix(true_class):
    """``e[j, k]`` = fraction of labeled items where ``j`` and ``k`` are wrong together
    on the same answer.  This is the user's N-by-N group-correlated-error matrix."""
    counts, n = wrong_collision_counts(true_class)
    return counts / max(n, 1)


def excess_collision(e, group=None):
    """Collision in EXCESS of what independent models coincide on by chance.

    The discount must penalise only the correlation a shared provenance adds; two
    unrelated models still land on the same wrong answer sometimes (there are only so
    many plausible wrong tables).  The chance level is the mean off-diagonal rate --
    restricted to CROSS-group pairs when a grouping is supplied, which is the same
    baseline :func:`pooleval.latent._estimate_loadings` subtracts.  The diagonal is
    zeroed: a model never discounts itself.
    """
    e = np.asarray(e, dtype=float)
    M = e.shape[0]
    off = ~np.eye(M, dtype=bool)
    if group is not None:
        group = np.asarray(group)
        cross = off & (group[:, None] != group[None, :])
        pool = e[cross] if cross.any() else e[off]
    else:
        pool = e[off]
    baseline = float(pool.mean()) if pool.size else 0.0
    out = np.clip(e - baseline, 0.0, None)
    np.fill_diagonal(out, 0.0)
    return out


def pseudo_label_quality(true_class, pseudo_label):
    """``beta`` on a labeled split: the fraction of items whose pseudo-label is gold.

    ``pseudo_label[i]`` is an observed class id, so it is correct exactly when some
    model answering that class was scored correct, i.e. when the id is ``0``.
    """
    return float(np.mean(np.asarray(pseudo_label) == 0))


def gamma_counts(true_class, pseudo_label, mode="model_wrong"):
    """Raw counts behind ``gamma[j] = P(C = 0 | Z = 0)`` on a labeled split.

    ``C = 0`` always means "this model's prediction differs from the pseudo-label".
    The three modes differ only in what ``Z = 0`` is taken to condition on:

    ``"both_wrong"``   -- the model is wrong AND the pseudo-label is wrong.  This is
        the event whose counting the spec describes ("prediction and pseudolabel is
        different and both wrong"), and it is the one statistic the binary reduction
        actually destroys: ``gamma^{both} = 1 - gamma^{collision}`` of the .tex.  Note
        that it is a CONDITIONAL-ON-BOTH-WRONG rate, so plugging it straight into the
        E-step as ``P(C = 0 | Z = 0)`` is only exact when the pseudo-label is always
        wrong; the exact identity is ``P(C = 0 | Z = 0) = beta + (1 - beta) gamma^{both}``.

    ``"model_wrong"``  -- the model is wrong, marginalising over whether the
        pseudo-label happens to be right.  This matches ``Z_i^j`` as defined in
        ``Trinh_proof.tex`` ("classifier j is actually correct on instance i"), so it
        is the reading under which the E-step below is exactly the model's posterior,
        with no beta correction needed.

    ``"pseudo_wrong"`` -- the pseudo-label is wrong, the literal reading of "the
        pseudolabel is different from the true label".

    ``"collision"``    -- the ORIGINAL .tex quantity: given both the model and the
        pseudo-label are wrong, how often do they produce the SAME wrong answer.  It
        counts AGREEMENT, not disagreement, so it is the complement of ``both_wrong``
        on the same denominator: ``gamma^{coll} = 1 - gamma^{both}``.  It is NOT
        ``P(C = 0 | Z = 0)`` and must never be handed to the E-step directly; the
        E-step consumes ``d_j(beta) = 1 - (1 - beta) gamma^{coll}``, which depends on
        the LIVE beta and therefore costs the closed-form beta M-step.

    Returns ``(numerator, denominator)``, both length ``M``.
    """
    tc = np.asarray(true_class)
    yhat = np.asarray(pseudo_label)
    disagree = tc != yhat[None, :]                     # C = 0
    model_wrong = tc != 0
    pseudo_wrong = np.broadcast_to(yhat != 0, tc.shape)
    if mode in ("both_wrong", "collision", "collision_frozen"):
        cond = model_wrong & pseudo_wrong
    elif mode == "model_wrong":
        cond = model_wrong
    elif mode == "pseudo_wrong":
        cond = pseudo_wrong
    else:
        raise ValueError(f"unknown gamma mode {mode!r}")
    hit = (~disagree) if mode in ("collision", "collision_frozen") else disagree
    # the collision modes count AGREEMENT; the others count disagreement
    return (hit & cond).sum(axis=1).astype(float), cond.sum(axis=1).astype(float)


def pairwise_wrong_counts(true_class):
    """Counts behind the PAIRWISE route to ``P(C = 1 | Z = 0)``.

    Returns ``(same, wrong)`` where ``same[j, k]`` counts items on which ``j`` and ``k``
    gave the same answer and ``j`` was wrong, and ``wrong[j]`` counts items ``j`` got
    wrong.  Note ``r^j = r^k`` with ``j`` wrong forces ``k`` wrong too, so no separate
    both-wrong condition is needed -- which is exactly why this conditioning needs no
    ``beta`` correction.  The diagonal satisfies ``same[j, j] = wrong[j]``, i.e. a
    classifier always agrees with itself, and that term belongs: if ``j``'s own answer is
    the pseudo-label then ``C = 1`` with certainty.
    """
    tc = np.asarray(true_class)
    wrong = tc != 0                                        # [M, N]
    same = (tc[:, None, :] == tc[None, :, :]) & wrong[:, None, :]
    return same.sum(axis=2).astype(float), wrong.sum(axis=1).astype(float)


def gamma_from_pairwise(same, wrong, smoothing=1.0):
    """``gamma_j = 1 - mean_k P(r^j = r^k | r^j wrong)`` -- the row functional of the
    correlated-error matrix, averaged over classifiers INCLUDING ``j`` itself.

    Unlike the counted route this is invariant to the EM state: it is a function of the
    answers and the gold labels only, never of ``alpha`` or of the pseudo-labels those
    imply.  The cost is a systematic downward bias in ``P(C = 1 | Z = 0)`` -- the
    pseudo-label is the WINNER of a weighted vote, so agreeing with it is agreeing with
    the modal wrong answer, while a row mean asks about a typical one.
    """
    same = np.asarray(same, dtype=float)
    wrong = np.asarray(wrong, dtype=float)
    M = same.shape[0]
    agree = (same + smoothing) / (wrong[:, None] + 2.0 * smoothing)
    return _clip(1.0 - agree.mean(axis=1))


def gamma_from_counts(numerator, denominator, smoothing=1.0):
    """Laplace-smoothed ``gamma``; a model with no eligible labeled item falls back to 1/2."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return _clip((numerator + smoothing) / (denominator + 2.0 * smoothing))


def gamma_to_conditional(gamma_both, beta):
    """Convert a both-wrong rate into the ``P(C = 0 | Z = 0)`` the E-step needs.

    Given the model is wrong, the pseudo-label is right with probability ``beta`` and
    the two then necessarily differ; otherwise they differ with probability
    ``gamma^{both}``.  Hence ``beta + (1 - beta) gamma^{both}``, which is exactly
    ``d_j(beta) = 1 - (1 - beta) gamma^{collision}`` of the .tex under
    ``gamma^{both} = 1 - gamma^{collision}``.
    """
    g = _clip(gamma_both)
    return _clip(beta + (1.0 - beta) * g)


class LabeledStatistics:
    """``e`` and ``gamma``: measured on the labeled split, then refreshed by the expert.

    The source split fixes the starting values.  After EM has run and the expert has
    answered -- and BEFORE the warm-started re-solve -- the same two statistics are
    recomputed on the target items the expert has revealed so far (``temp``) and
    blended into the running values:

        new = (old + temp) / 2

    a one-line exponential filter: each update halves the weight of everything that
    came before, so the source-domain numbers hand over to target-domain ones as
    validation accumulates without ever discarding them outright.  Entries whose
    ``temp`` is undefined (a model with no eligible validated item yet) keep their old
    value, so the blend is a no-op there rather than a pull toward the 1/2 fallback.

    ``update_rule="counts"`` is the alternative: pool the validated items into the
    source counts and recompute, which weights every labeled item equally instead of
    geometrically favouring the newest.  Kept so the two can be compared.

    Validated items enter ``gamma`` through the pseudo-label the consensus WOULD have
    produced without the pin, not through the pinned label -- pinning makes the
    pseudo-label equal to the truth by construction, which would drive ``gamma`` to 1
    and say nothing about the unvalidated items it is used on.
    """

    def __init__(self, true_class_src, pseudo_label_src, group=None,
                 gamma_mode="model_wrong", smoothing=1.0, update_rule="average",
                 temp_scope="validated_all"):
        tc = np.asarray(true_class_src)
        yhat = np.asarray(pseudo_label_src)
        self.M = int(tc.shape[0])
        self.group = None if group is None else np.asarray(group)
        self.gamma_mode = gamma_mode
        self.smoothing = float(smoothing)
        self.update_rule = update_rule
        self.temp_scope = temp_scope

        self.collision_counts, self.n_labeled = wrong_collision_counts(tc)
        if gamma_mode in ("pairwise", "pairwise_calibrated"):
            self.pair_same, self.pair_wrong = pairwise_wrong_counts(tc)
            self.gamma_num, self.gamma_den = gamma_counts(tc, yhat, mode="model_wrong")
        else:
            self.pair_same = self.pair_wrong = None
            self.gamma_num, self.gamma_den = gamma_counts(tc, yhat, mode=gamma_mode)
        self.pseudo_hits = float(np.sum(yhat == 0))
        self.pseudo_total = int(tc.shape[1])

        self._e = self.collision_counts / max(self.n_labeled, 1)
        self._gamma = (gamma_from_pairwise(self.pair_same, self.pair_wrong, self.smoothing)
                       if gamma_mode in ("pairwise", "pairwise_calibrated")
                       else gamma_from_counts(self.gamma_num, self.gamma_den, self.smoothing))
        self._pseudo_acc = float((self.pseudo_hits + self.smoothing)
                                 / (self.pseudo_total + 2.0 * self.smoothing))
        self.source_e = self._e.copy()
        self.source_gamma = self._gamma.copy()
        self.source_pseudo_accuracy = self._pseudo_acc
        self.records = []
        self.history = []

    @classmethod
    def random(cls, M, group=None, seed=None, gamma_mode="model_wrong",
               pi_range=(0.05, 0.95), e_off=(0.0, 0.5), e_diag=(0.0, 0.8), **kw):
        """Starting statistics drawn at random, with NO labeled split.

        Every value this class normally measures -- ``e``, ``gamma``, ``pseudo_accuracy``
        -- comes from one artifact: the M models run over a labeled subset.  That run is
        what costs money and latency at deployment, so the real question is not which
        prior to compute but whether to pay for the subset at all.  This constructor is
        the "don't pay" option.

        What it costs, measured (``docs/random_init.md``, MAE in accuracy points, mean
        over six cases, ten draws):

        ================  =====  =====
        randomised        b0     b40
        ================  =====  =====
        beta              +0.03  +0.32
        e                 +0.20  -0.16
        gamma             +4.45  +0.25
        ================  =====  =====

        ``beta`` is free because its closed-form M-step has a unique maximiser and reaches
        it from any start; ``e`` is nearly free because it only reweights votes through a
        bounded monotone discount and rarely changes which answer wins; ``gamma`` costs
        real accuracy at the start but the expert loop's ``(old + temp) / 2`` blend repairs
        it within forty labels, since EM never touches it and so every point of that
        recovery is attributable to validation.

        The one prior that does NOT recover is the anchor on ``alpha``, which is not held
        here -- it is the ``prior`` / ``prior_strength`` pair passed to :func:`validated_em`.
        Randomising that costs ~9.8 points at b0 and still ~2.7 at b40.  It is also a
        per-model mean, so it needs far fewer labeled items than the M x M ``e``: a few
        hundred is enough.  The cheap-but-good deployment mode is therefore a small subset
        spent on ``alpha`` alone, with this constructor supplying the rest.

        One caveat that cuts the other way: on both vision cases a random anchor BEAT the
        measured one, because the labeled split was distribution-shifted from the target
        pool.  A measured prior is only worth its cost when the subset resembles the data
        being evaluated; under shift the free option is also the more accurate one.
        """
        rng = np.random.default_rng(seed)
        tc = np.zeros((int(M), 1), dtype=int)
        self = cls(tc, np.zeros(1, dtype=int), group=group, gamma_mode=gamma_mode, **kw)

        # Erase the dummy evidence so ``update_rule="counts"`` pools the validated items
        # into an empty prior rather than into a fabricated all-correct item.
        self.n_labeled = 0
        self.collision_counts = np.zeros((self.M, self.M), dtype=float)
        self.gamma_num = np.zeros(self.M, dtype=float)
        self.gamma_den = np.zeros(self.M, dtype=float)
        if self.pair_same is not None:
            self.pair_same = np.zeros((self.M, self.M), dtype=float)
            self.pair_wrong = np.zeros((self.M, self.M), dtype=float)
        self.pseudo_hits, self.pseudo_total = 0.0, 0

        lo, hi = pi_range
        self._gamma = rng.uniform(lo, hi, size=self.M)
        self._pseudo_acc = float(rng.uniform(lo, hi))
        a = rng.uniform(e_off[0], e_off[1], size=(self.M, self.M))
        a = (a + a.T) / 2.0
        np.fill_diagonal(a, rng.uniform(e_diag[0], e_diag[1], size=self.M))
        self._e = a

        self.source_e = self._e.copy()
        self.source_gamma = self._gamma.copy()
        self.source_pseudo_accuracy = self._pseudo_acc
        return self

    # -- current values ------------------------------------------------------
    @property
    def e(self):
        return self._e

    @property
    def gamma(self):
        return self._gamma

    def _pair_gamma(self):
        """The uncalibrated pairwise row functional, from the running pair counts."""
        return gamma_from_pairwise(self.pair_same, self.pair_wrong, self.smoothing)

    @property
    def pseudo_accuracy(self):
        """``beta_hat``: the measured fraction of items whose pseudo-label is gold.

        Deliberately NOT called ``beta``.  The model's ``beta`` is a free parameter fitted
        by the M-step from soft posteriors, with no access to any label; this is a plain
        count against gold on data that has labels.  They estimate the same underlying
        quantity, which is why one seeds the other, but only this one is measured, and
        only this one is exposed to how the validated items were chosen.  Its live job is
        the conversion in :meth:`conditional_gamma`.
        """
        return self._pseudo_acc

    @property
    def n_validated(self):
        return len(self.records)

    def e_excess(self):
        return excess_collision(self._e, self.group)

    def conditional_gamma(self):
        """``P(C = 0 | Z = 0)`` as the E-step consumes it, per :func:`gamma_to_conditional`."""
        if self.gamma_mode == "collision":
            raise ValueError(
                "gamma_mode='collision' has no beta-free P(C=0|Z=0): the E-step must "
                "build d_j(beta) = 1 - (1 - beta) gamma^coll from the LIVE beta each "
                "sweep. validated_em() handles this; do not call conditional_gamma().")
        if self.gamma_mode == "collision_frozen":
            # d_j(beta_hat) = 1 - (1 - beta_hat) gamma^coll.  Algebraically identical to
            # the both_wrong conversion, since gamma^coll = 1 - gamma^both; verified equal
            # to machine precision on all six cases (docs/collision_gamma.md, Part D).
            return _clip(1.0 - (1.0 - self._pseudo_acc) * _clip(self._gamma))
        if self.gamma_mode == "both_wrong":
            return gamma_to_conditional(self._gamma, self._pseudo_acc)
        return self._gamma          # model_wrong and pairwise are already P(C=0|Z=0)

    # -- growth --------------------------------------------------------------
    def add_validated(self, answers, truth, consensus_label):
        """Fold one expert-validated item in and refresh ``e``, ``gamma`` and ``beta``."""
        self.records.append((np.asarray(answers).copy(), int(truth),
                             int(consensus_label)))
        temp_e, temp_gamma, temp_pseudo, defined = self._temp()
        if self.update_rule == "average":
            self._e = 0.5 * (self._e + temp_e)
            blended = 0.5 * (self._gamma + temp_gamma)
            self._gamma = _clip(np.where(defined, blended, self._gamma))
            self._pseudo_acc = 0.5 * (self._pseudo_acc + temp_pseudo)
        elif self.update_rule == "counts":
            self._pool_counts()
        else:
            raise ValueError(f"unknown update rule {self.update_rule!r}")
        self.history.append(dict(n_validated=len(self.records),
                                 e_mean=float(self._e.mean()),
                                 gamma_mean=float(self._gamma.mean()),
                                 pseudo_accuracy=float(self._pseudo_acc)))

    def _records(self):
        return self.records[-1:] if self.temp_scope == "latest" else self.records

    def _temp(self):
        """``(e, gamma, beta, gamma_defined)`` measured on the validated items alone."""
        recs = self._records()
        answers = np.stack([r[0] for r in recs])              # [K, M]
        truth = np.array([r[1] for r in recs])
        consensus = np.array([r[2] for r in recs])
        K = len(recs)

        wrong = answers != truth[:, None]                     # [K, M]
        same = (answers[:, :, None] == answers[:, None, :]) & (answers[:, :, None] >= 0)
        temp_e = (same & wrong[:, :, None] & wrong[:, None, :]).sum(axis=0) / K

        disagree = answers != consensus[:, None]
        pseudo_wrong = (consensus != truth)[:, None]
        if self.gamma_mode in ("both_wrong", "collision", "collision_frozen"):
            cond = wrong & pseudo_wrong
        elif self.gamma_mode == "model_wrong":
            cond = wrong
        else:
            cond = np.broadcast_to(pseudo_wrong, wrong.shape)
        hit = (~disagree) if self.gamma_mode in ("collision", "collision_frozen") else disagree
        num = (hit & cond).sum(axis=0).astype(float)
        den = cond.sum(axis=0).astype(float)
        if self.gamma_mode == "pairwise":
            # same row functional, recomputed on the validated items alone
            ps = (answers[:, :, None] == answers[:, None, :]) & wrong[:, :, None]
            temp_gamma = gamma_from_pairwise(ps.sum(axis=0).astype(float),
                                             wrong.sum(axis=0).astype(float),
                                             self.smoothing)
            den = wrong.sum(axis=0).astype(float)
        elif self.gamma_mode == "pairwise_calibrated":
            # Same rule as every other statistic here: temp is computed on the
            # EXPERT-VALIDATED ITEMS ALONE, then blended (old + temp) / 2.
            #
            # The pairwise row functional on those items supplies the SHAPE -- which
            # classifier disagrees more than which -- because it recovers the ordering
            # well (correlation ~0.9-0.98 with truth). Its LEVEL is biased, because the
            # pseudo-label is the winner of a weighted vote while a row mean asks about a
            # typical classifier, so the level is taken from the directly-counted rate on
            # the same items, where the expert's answer serves as truth.
            ps = (answers[:, :, None] == answers[:, None, :]) & wrong[:, :, None]
            shape = gamma_from_pairwise(ps.sum(axis=0).astype(float),
                                        wrong.sum(axis=0).astype(float), self.smoothing)
            level = gamma_from_counts((disagree & wrong).sum(axis=0).astype(float),
                                      wrong.sum(axis=0).astype(float), self.smoothing)
            temp_gamma = _clip(shape + (float(np.mean(level)) - float(np.mean(shape))))
            den = wrong.sum(axis=0).astype(float)
        else:
            temp_gamma = gamma_from_counts(num, den, self.smoothing)
        return temp_e, temp_gamma, float(np.mean(consensus == truth)), den > 0

    def _pool_counts(self):
        recs = self.records
        answers = np.stack([r[0] for r in recs])
        truth = np.array([r[1] for r in recs])
        consensus = np.array([r[2] for r in recs])
        wrong = answers != truth[:, None]
        same = (answers[:, :, None] == answers[:, None, :]) & (answers[:, :, None] >= 0)
        counts = self.collision_counts + (same & wrong[:, :, None]
                                          & wrong[:, None, :]).sum(axis=0)
        self._e = counts / (self.n_labeled + len(recs))

        disagree = answers != consensus[:, None]
        pseudo_wrong = (consensus != truth)[:, None]
        if self.gamma_mode in ("both_wrong", "collision", "collision_frozen"):
            cond = wrong & pseudo_wrong
        elif self.gamma_mode == "model_wrong":
            cond = wrong
        else:
            cond = np.broadcast_to(pseudo_wrong, wrong.shape)
        self._gamma = gamma_from_counts(
            self.gamma_num + (disagree & cond).sum(axis=0),
            self.gamma_den + cond.sum(axis=0), self.smoothing)
        self._pseudo_acc = float(
            (self.pseudo_hits + float(np.sum(consensus == truth)) + self.smoothing)
            / (self.pseudo_total + len(recs) + 2.0 * self.smoothing))


# --------------------------------------------------------------------------- #
#  2.  Latent-answer posterior with the measured pairwise discount             #
# --------------------------------------------------------------------------- #
def vote_discount(column, e_excess):
    """Per-model discount ``1 / (1 + sum of excess collision with co-voters)``.

    A model's vote is worth less when the models voting the SAME class are the ones
    it is already known to fail with.  Setting ``e_excess[j, k] = u_g`` for same-group
    pairs and ``0`` otherwise reproduces ``1 / (1 + u_g (n_{g,k} - 1))`` exactly, so
    :mod:`pooleval.latent`'s group discount is the special case in which correlation
    is declared by provenance rather than measured.
    """
    column = np.asarray(column)
    same = column[:, None] == column[None, :]
    np.fill_diagonal(same, False)
    return 1.0 / (1.0 + (same * e_excess).sum(axis=1))


class LatentPlan:
    """Everything about the latent E-step that does NOT change between EM sweeps.

    ``obs`` is fixed and so is ``e_excess`` between judge calls, so each vote's
    correlation discount and each item's candidate list are computed once.  Only the
    reliability weights ``w = alpha`` move from sweep to sweep.

    The per-item candidate lists have ragged lengths, so they are stored FLAT: every
    (item, candidate class) pair gets one slot, ``offsets`` marks where each item's
    slots begin, and ``slot[m, i]`` says which slot model ``m``'s vote at item ``i``
    lands in.  A whole E-step is then one ``np.add.at`` into the flat score vector
    plus segmented max/sum reductions -- no Python loop over items.  That matters:
    information gain re-solves the model once per (candidate item, candidate answer)
    pair, so this inner loop runs tens of thousands of times per judge call.

    Rebuild the plan when the labeled statistics grow (``e_excess`` changes) -- once
    per expert overrule, not once per iteration.
    """

    def __init__(self, obs, e_excess):
        obs = np.asarray(obs)
        self.obs = obs
        self.M, self.N = obs.shape
        self.discount = np.empty((self.M, self.N))
        classes, counts = [], np.empty(self.N, dtype=np.int64)
        slot = np.empty((self.M, self.N), dtype=np.int64)
        base = 0
        for i in range(self.N):
            col = obs[:, i]
            self.discount[:, i] = vote_discount(col, e_excess)
            ks, inv = np.unique(col, return_inverse=True)
            classes.append(ks)
            counts[i] = len(ks)
            slot[:, i] = base + inv
            base += len(ks)
        self.flat_classes = np.concatenate(classes) if classes else np.empty(0, np.int64)
        self.counts = counts
        self.offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self.slot = slot
        self.total = int(base)
        self.seg_index = np.arange(self.total)
        self.item_classes = classes

    # -- one E-step, fully vectorised ---------------------------------------
    def _flat_scores(self, alpha, verifier_guess=None, verifier_strength=0.0):
        w = np.clip(np.asarray(alpha, dtype=float), 0.05, 0.99)
        contrib = w[:, None] * self.discount
        score = np.zeros(self.total)
        np.add.at(score, self.slot.ravel(), contrib.ravel())
        if verifier_strength and verifier_guess is not None:
            hit = self.flat_classes == np.repeat(np.asarray(verifier_guess), self.counts)
            score = score + verifier_strength * hit
        return score

    def _softmax(self, score):
        seg_max = np.repeat(np.maximum.reduceat(score, self.offsets), self.counts)
        p = np.exp(score - seg_max)
        return p / np.repeat(np.add.reduceat(p, self.offsets), self.counts)

    def solve(self, alpha, constraints=None, verifier_guess=None,
              verifier_strength=0.0):
        """``(hard labels, H(P), flat probabilities)`` in one pass.

        Pinned items contribute a one-hot distribution, hence zero entropy, and their
        label is the pinned class -- which may be a class no model produced.
        """
        score = self._flat_scores(alpha, verifier_guess, verifier_strength)
        p = self._softmax(score)
        # argmax per item: lowest slot index among the maxima, matching np.unique order
        seg_max = np.repeat(np.maximum.reduceat(score, self.offsets), self.counts)
        masked = np.where(score >= seg_max, self.seg_index, self.total)
        labels = self.flat_classes[np.minimum.reduceat(masked, self.offsets)]
        ent = -np.add.reduceat(np.where(p > 0.0, p * np.log(np.maximum(p, 1e-300)), 0.0),
                               self.offsets)
        if constraints:
            idx = np.fromiter(constraints.keys(), dtype=np.int64, count=len(constraints))
            labels = labels.copy()
            labels[idx] = np.fromiter(constraints.values(), dtype=labels.dtype,
                                      count=len(constraints))
            ent = ent.copy()
            ent[idx] = 0.0
        return labels, float(ent.sum()), p

    def posterior(self, alpha, constraints=None, verifier_guess=None,
                  verifier_strength=0.0):
        """``U[i]`` as a dict per item. Slower than :meth:`solve`; for reporting."""
        p = self._softmax(self._flat_scores(alpha, verifier_guess, verifier_strength))
        constraints = dict(constraints or {})
        out = []
        for i in range(self.N):
            if i in constraints:
                out.append({int(constraints[i]): 1.0})
                continue
            lo = self.offsets[i]
            hi = lo + self.counts[i]
            out.append({int(k): float(v)
                        for k, v in zip(self.flat_classes[lo:hi], p[lo:hi])})
        return out


def latent_posterior(obs, alpha, e_excess, constraints=None, verifier_guess=None,
                     verifier_strength=0.0):
    """``U[i]`` -- the posterior over an item's true answer, as a dict per item.

    Votes are summed on the accuracy scale (a model contributes its reliability,
    discounted for correlation) and turned into a distribution by a softmax, matching
    :func:`pooleval.latent.run_em`.  Items in ``constraints`` are pinned one-hot, Eq.
    (4) of Hung et al.: their latent answer is no longer estimated.  A pinned class
    may be absent from ``obs[:, i]``, in which case every model scores wrong there.
    """
    return LatentPlan(obs, e_excess).posterior(alpha, constraints, verifier_guess,
                                               verifier_strength)


def hard_labels(posterior):
    return np.array([max(p, key=p.get) for p in posterior], dtype=np.int64)


def item_entropy(post):
    p = np.fromiter(post.values(), dtype=float)
    p = p[p > 0.0]
    return float(-(p * np.log(p)).sum())


def set_entropy(posterior):
    """``H(P) = sum_o H(o)``, Eq. (7) of Hung et al."""
    return float(sum(item_entropy(p) for p in posterior))


# --------------------------------------------------------------------------- #
#  3.  The EM itself                                                           #
# --------------------------------------------------------------------------- #
def correctness_em(C, gamma, prior, prior_strength, beta_init=None,
                   beta_prior=None, beta_strength=0.0, max_iters=200, tol=1e-8):
    """The alpha/beta EM alone, with the pseudo-labels (hence ``C``) held fixed.

    This is the direct replacement for :func:`pooleval.new_formulation.collision_agreement_em`:
    same data, same alpha M-step, but ``P(C = 0 | Z = 0)`` is the measured ``gamma``
    instead of ``1 - (1 - beta) gamma^{collision}``.  Because ``beta`` no longer
    appears in the ``Z = 0`` branch, ``Q_beta`` is a plain weighted Bernoulli
    log-likelihood and its maximiser is the ratio of expected counts below -- no
    bounded 1-D search.  ``trace`` is the anchored log-posterior, which EM increases.
    """
    C = np.asarray(C, dtype=float)
    M, N = C.shape
    g = _clip(gamma)[:, None]
    pi = _clip(prior)
    s = np.broadcast_to(np.asarray(prior_strength, dtype=float), (M,)).astype(float)
    beta = float(_clip(0.7 if beta_init is None else beta_init))
    b0 = float(_clip(beta if beta_prior is None else beta_prior))
    bs = float(beta_strength)

    def log_posterior(a, b):
        p1 = a[:, None] * b + (1.0 - a[:, None]) * (1.0 - g)
        obs = np.where(C == 1.0, p1, 1.0 - p1)
        anchor = float(np.sum(s * (pi * np.log(a) + (1.0 - pi) * np.log(1.0 - a))))
        anchor += bs * (b0 * np.log(b) + (1.0 - b0) * np.log(1.0 - b))
        return float(np.log(np.clip(obs, EPS, None)).sum() + anchor)

    alpha = pi.copy()
    trace = []
    for iteration in range(1, int(max_iters) + 1):
        a_prev, b_prev = alpha.copy(), beta
        like_z1 = np.where(C == 1.0, beta, 1.0 - beta)
        like_z0 = np.where(C == 1.0, 1.0 - g, g)
        numer = alpha[:, None] * like_z1
        tau = numer / np.clip(numer + (1.0 - alpha[:, None]) * like_z0, EPS, None)
        alpha = _clip((tau.sum(axis=1) + s * pi) / (N + s))
        beta = float(_clip((float((tau * C).sum()) + bs * b0)
                           / (float(tau.sum()) + bs)))
        trace.append(log_posterior(alpha, beta))
        if max(float(np.max(np.abs(alpha - a_prev))), abs(beta - b_prev)) < tol:
            break
    return dict(alpha=alpha, acc=alpha, beta=beta, tau=tau,
                log_posterior=np.asarray(trace), n_iters=iteration,
                a_sigma=np.sqrt(np.clip(alpha * (1.0 - alpha), EPS, None) / (N + s)))


def validated_em(obs, stats, prior, prior_strength, constraints=None,
                 init=None, beta_strength=0.0, beta_max=1.0 - EPS,
                 max_iters=200, tol=1e-8,
                 verifier_guess=None, verifier_strength=0.0, use_discount=True,
                 plan=None, report_posterior=True, pseudo_mode="joint"):
    """Joint EM over the latent answers and the per-model accuracies.

    One sweep is

      1.  latent E-step -- ``U(o, l)`` from correlation-discounted votes, pinned
          one-hot on validated items;
      2.  ``C[j, i] = 1(obs[j, i] == argmax_l U(i, l))``;
      3.  correctness E-step -- ``tau = P(Z = 1 | C, alpha, beta, gamma)``, with
          ``tau`` OBSERVED (0/1) on validated items, since the truth is known there;
      4.  M-step -- both maximisers in closed form.

    ``prior`` / ``prior_strength`` are the Beta anchor ``Beta(1 + s pi, 1 + s(1-pi))``
    of the .tex, so the alpha update is ``(sum_i tau + s pi) / (N + s)``.

    ``pseudo_mode`` decides whether step 1 runs every sweep:

    ``"fixed"``  -- the STRICT reading of ``Trinh_proof.tex``.  The pseudo-labels, and
        hence ``C``, are computed ONCE at the incoming ``alpha`` and then held constant,
        so the inner loop is a genuine EM over ``(alpha, beta)`` on a fixed likelihood:
        its observed log-likelihood is monotone, and it converges to a stationary point.
        This is what :func:`correctness_em` does, with the expert loop wrapped around it.

    ``"joint"``  -- pseudo-labels are re-derived each sweep, so the estimate of WHICH
        answer is right and the estimate of WHO is reliable improve together.  This is
        a two-block coordinate ascent, not an EM on one likelihood, so monotonicity is
        NOT guaranteed -- the latent step is a reliability-weighted consensus, not the
        E-step of a joint generative model.  It is the default because the alternative
        freezes in whatever the initial anchor believed, but the two are reported
        side by side rather than one being assumed better.
    """
    obs = np.asarray(obs)
    M, N = obs.shape
    pi = _clip(prior)
    s = np.broadcast_to(np.asarray(prior_strength, dtype=float), (M,)).astype(float)
    if np.any(s < 0):
        raise ValueError("prior_strength must be non-negative")
    collision_mode = getattr(stats, "gamma_mode", None) == "collision"
    # In collision mode gamma is P(C=1|Z=0, both wrong) and the E-step needs
    # d_j(beta) = 1 - (1-beta) gamma^coll, rebuilt every sweep from the live beta.
    gamma = (_clip(stats.gamma)[:, None] if collision_mode
             else _clip(stats.conditional_gamma())[:, None])
    if plan is None:
        e_excess = stats.e_excess() if use_discount else np.zeros((M, M))
        plan = LatentPlan(obs, e_excess)
    constraints = dict(constraints or {})

    init = dict(init or {})
    alpha = _clip(init.get("alpha", pi)).copy()
    beta = float(_clip(init.get("beta", stats.pseudo_accuracy)))
    beta_prior = float(_clip(stats.pseudo_accuracy))
    beta_strength = float(beta_strength)

    if pseudo_mode not in ("joint", "fixed"):
        raise ValueError(f"unknown pseudo_mode {pseudo_mode!r}")
    frozen = None
    if pseudo_mode == "fixed":
        yhat, entropy, _ = plan.solve(alpha, constraints, verifier_guess,
                                      verifier_strength)
        frozen = (yhat, entropy, (obs == yhat[None, :]).astype(float))

    trace = []
    for iteration in range(1, int(max_iters) + 1):
        alpha_prev, beta_prev = alpha.copy(), beta

        if frozen is None:
            yhat, entropy, _ = plan.solve(alpha, constraints, verifier_guess,
                                          verifier_strength)
            C = (obs == yhat[None, :]).astype(float)
        else:
            yhat, entropy, C = frozen

        # P(C | Z = 1) uses beta; P(C | Z = 0) uses the measured, frozen gamma.
        like_z1 = np.where(C == 1.0, beta, 1.0 - beta)
        if collision_mode:
            # P(C=1|Z=0) = (1-beta) gamma^coll, so P(C=0|Z=0) = d_j(beta).
            agree_z0 = _clip((1.0 - beta) * gamma)
            like_z0 = np.where(C == 1.0, agree_z0, 1.0 - agree_z0)
        else:
            like_z0 = np.where(C == 1.0, 1.0 - gamma, gamma)
        numer = alpha[:, None] * like_z1
        tau = numer / np.clip(numer + (1.0 - alpha[:, None]) * like_z0, EPS, None)
        # On a validated item the true answer is known, so Z is observed, not inferred.
        for i, label in constraints.items():
            tau[:, i] = (obs[:, i] == label).astype(float)

        alpha = _clip((tau.sum(axis=1) + s * pi) / (N + s))
        if collision_mode:
            # The Z=0 branch still contains beta, so Q_beta is not a Bernoulli and the
            # ratio-of-counts maximiser is invalid. This is the bounded 1-D numerical
            # M-step the collision formulation is stuck with.
            beta = _collision_beta_mstep(tau, C, gamma, beta_prior,
                                         beta_strength, beta_max)
        else:
            # Closed-form beta: freezing the Z=0 branch leaves a weighted Bernoulli.
            beta = float(_clip((float((tau * C).sum()) + beta_strength * beta_prior)
                               / (float(tau.sum()) + beta_strength)))

        obs_like = alpha[:, None] * like_z1 + (1.0 - alpha[:, None]) * like_z0
        trace.append(float(np.log(np.clip(obs_like, EPS, None)).sum()))
        if max(float(np.max(np.abs(alpha - alpha_prev))),
               abs(beta - beta_prev)) < tol:
            break

    if frozen is None:
        yhat, entropy, _ = plan.solve(alpha, constraints, verifier_guess,
                                      verifier_strength)
    posterior = (plan.posterior(alpha, constraints, verifier_guess, verifier_strength)
                 if report_posterior else None)
    return dict(alpha=alpha, plan=plan, acc=alpha, beta=beta, gamma=stats.gamma.copy(),
                tau=tau, posterior=posterior, pseudo_label=yhat,
                ranking=np.argsort(-alpha), entropy=entropy,
                a_sigma=np.sqrt(np.clip(alpha * (1.0 - alpha), EPS, None) / (N + s)),
                log_likelihood=np.asarray(trace), n_iters=iteration)


def _collision_beta_mstep(tau, C, gamma_coll, beta_prior, beta_strength,
                          beta_max=1.0 - EPS):
    """Bounded 1-D maximisation of ``Q(beta)`` under the collision parameterisation.

    With ``P(C=1|Z=0) = (1-beta) gamma^coll`` the expected complete-data log-likelihood

        Q(beta) = sum tau [C log beta + (1-C) log(1-beta)]
                + sum (1-tau) [C log((1-beta)gamma) + (1-C) log(1 - (1-beta)gamma)]

    contains ``beta`` in BOTH branches.  The second one contributes
    ``log(1 - gamma + beta gamma)``, whose derivative ``gamma / d(beta)`` is rational in
    ``beta``; clearing denominators gives a polynomial whose degree grows with the number
    of distinct ``gamma`` values, so there is no closed form.  A golden-section search on
    the unit interval is used instead -- correct, but one numerical solve per sweep.
    """
    from scipy.optimize import minimize_scalar

    def negative_q(b):
        b = float(b)
        z1 = C * np.log(b) + (1.0 - C) * np.log(1.0 - b)
        agree = np.clip((1.0 - b) * gamma_coll, EPS, 1.0 - EPS)
        z0 = C * np.log(agree) + (1.0 - C) * np.log(1.0 - agree)
        anchor = beta_strength * (beta_prior * np.log(b)
                                  + (1.0 - beta_prior) * np.log(1.0 - b))
        return -float(np.sum(tau * z1 + (1.0 - tau) * z0) + anchor)

    opt = minimize_scalar(negative_q, bounds=(EPS, float(beta_max)), method="bounded",
                          options={"xatol": 1e-10})
    return float(_clip(opt.x))


# --------------------------------------------------------------------------- #
#  4.  Information gain over the whole answer set (Hung et al. Eqs. 7-10)      #
# --------------------------------------------------------------------------- #
def _refit(obs, stats, prior, prior_strength, constraints, init, iters, **kw):
    return validated_em(obs, stats, prior, prior_strength, constraints=constraints,
                        init=init, max_iters=iters, **kw)


def acquisition_scores(obs, stats, prior, prior_strength, state, constraints,
                       candidates, ig_iters=6, min_prob=1e-3, **kw):
    """Both acquisition criteria from one set of hypothetical re-solves.

    The expensive part -- refitting once per (candidate item, candidate answer) pair --
    is shared, so the two objectives cost the same and can be compared honestly.

    ``entropy_gain``  Hung et al. Eq. (9): ``H(P) - sum_l U(o,l) H(P_l)``.  It asks
        which query most reduces uncertainty about EVERY label.  That is the right
        objective for recovering labels and the wrong one for estimating a mean.

    ``mean_gain``     the variance-reduction criterion for the quantity actually being
        estimated.  Writing ``alpha_j`` for model j's accuracy and ``V = sum_j
        Var(alpha_j | D)``, the law of total variance gives

            A(q) = V - E[V | after querying q] = sum_j Var_l( E[alpha_j | D, e(q)=l] )

        i.e. how much learning the answer at ``q`` could MOVE the accuracy estimate.
        With two possible answers this is exactly ``p(1-p) (m_1 - m_0)^2``: an item
        uncertainty factor times a GLOBAL LEVERAGE factor.  Entropy sampling sees only
        the first.  When errors are correlated, a confidently-wrong item can have tiny
        ``p(1-p)`` and enormous leverage -- revealing it exposes a shared failure mode
        that moves many other items at once -- which is precisely the case label
        entropy is blind to.
    """
    H_now = state["entropy"]
    init = dict(alpha=state["alpha"], beta=state["beta"])
    kw = dict(kw, plan=state.get("plan"), report_posterior=False)
    out = {}
    for o in candidates:
        post = state["posterior"][o]
        kept = {int(l): float(pv) for l, pv in post.items() if pv >= min_prob}
        total = sum(kept.values())
        if not kept or total <= 0:
            out[int(o)] = dict(entropy_gain=0.0, mean_gain=0.0, n_outcomes=0)
            continue
        probs, alphas, entropies = [], [], []
        for label, prob in kept.items():
            trial = dict(constraints)
            trial[int(o)] = int(label)
            fit = _refit(obs, stats, prior, prior_strength, trial, init, ig_iters, **kw)
            probs.append(prob / total)
            alphas.append(fit["alpha"])
            entropies.append(fit["entropy"])
        w = np.asarray(probs)
        A = np.asarray(alphas)                       # [n_outcomes, M]
        mean = w @ A
        var = w @ (A ** 2) - mean ** 2               # Var_l(E[alpha_j | Z_q]) per model
        out[int(o)] = dict(
            entropy_gain=float(H_now - float(w @ np.asarray(entropies))),
            mean_gain=float(np.sum(np.clip(var, 0.0, None))),
            n_outcomes=len(w))
    return out


def information_gain(obs, stats, prior, prior_strength, state, constraints,
                     candidates, ig_iters=6, min_prob=1e-3, **kw):
    """``IG(o) = H(P) - sum_l U(o, l) H(P_l)``, Eq. (9). See :func:`acquisition_scores`."""
    scores = acquisition_scores(obs, stats, prior, prior_strength, state, constraints,
                                candidates, ig_iters, min_prob, **kw)
    return {o: v["entropy_gain"] for o, v in scores.items()}


def mean_information_gain(obs, stats, prior, prior_strength, state, constraints,
                          candidates, ig_iters=6, min_prob=1e-3, **kw):
    """``A_mu(q) = sum_j Var_l(E[alpha_j | D, e(q)=l])``. See :func:`acquisition_scores`."""
    scores = acquisition_scores(obs, stats, prior, prior_strength, state, constraints,
                                candidates, ig_iters, min_prob, **kw)
    return {o: v["mean_gain"] for o, v in scores.items()}


def sampling_probabilities(scores, items, epsilon=0.2, temperature=None):
    """Turn an acquisition score into a proper sampling distribution.

    Horvitz-Thompson is an ESTIMATOR, not a sampling strategy: it can only undo a bias
    it knows the size of, and an item that had zero chance of selection is unrecoverable
    at any weight.  So a deterministic ``argmax`` cannot be HT-corrected after the fact
    -- the design has to carry known, strictly positive inclusion probabilities from the
    start.  This mixes a softmax over the scores with a uniform floor,

        p_i = (1 - eps) softmax(A_i / T)_i + eps / n,

    so high-value items are still strongly preferred while every item keeps
    ``p_i >= eps/n > 0``: the positivity condition HT needs.  ``temperature`` defaults to
    the score scale, which keeps the softmax from saturating when scores are tiny.
    """
    items = list(items)
    a = np.array([float(scores[i]) for i in items])
    if temperature is None:
        spread = float(a.max() - a.min())
        temperature = spread / 3.0 if spread > 0 else 1.0
    z = (a - a.max()) / max(temperature, 1e-12)
    soft = np.exp(z)
    soft /= soft.sum()
    p = (1.0 - epsilon) * soft + epsilon / len(items)
    return p / p.sum()


# --------------------------------------------------------------------------- #
#  5.  Experts                                                                 #
# --------------------------------------------------------------------------- #
class OracleExpert:
    """Picks the correct answer whenever the pool produced it.

    "Choose among the latent answers" is a RESTRICTED choice: the expert is shown the
    distinct results the pool returned and selects one.  When no model is correct the
    correct answer is simply not on the menu; ``allow_none`` decides whether the
    expert may then reject every candidate (returning a fresh class id that no model
    matches -- the candidate-coverage fix) or must abstain and waste the call.
    """

    name = "oracle"

    def __init__(self, true_class, allow_none=False):
        self.true_class = np.asarray(true_class)
        self.allow_none = bool(allow_none)
        self.calls = 0
        self._fresh = 20_000_000

    def _correct_class(self, obs, i):
        correct = np.where(self.true_class[:, i] == 0)[0]
        if len(correct) == 0:
            return None
        vals, counts = np.unique(obs[correct, i], return_counts=True)
        return int(vals[np.argmax(counts)])

    def query(self, obs, i):
        self.calls += 1
        answer = self._correct_class(obs, i)
        if answer is None and self.allow_none:
            self._fresh += 1
            return self._fresh
        return answer


class NoisyExpert(OracleExpert):
    """An expert that picks the right candidate only ``accuracy`` of the time.

    A wrong call still pins the item, so this measures how much of the gain survives
    an LLM judge that is good but not perfect -- the realistic case.
    """

    name = "noisy"

    def __init__(self, true_class, accuracy=0.9, seed=0, allow_none=False):
        super().__init__(true_class, allow_none=allow_none)
        self.accuracy = float(accuracy)
        self.rng = np.random.default_rng(seed)

    def query(self, obs, i):
        self.calls += 1
        truth = self._correct_class(obs, i)
        options = np.unique(obs[:, i])
        if truth is not None and self.rng.random() < self.accuracy:
            return truth
        wrong = [int(k) for k in options if k != truth]
        if not wrong:
            return truth
        return int(self.rng.choice(wrong))


class JudgeExpert:
    """Adapter for a real LLM judge, e.g. :class:`zoo.judge.RealJudge`.

    The loop only needs ``query(obs, i) -> class id or None``; the zoo judge takes
    ``(run, obs, i)`` and already implements the right protocol -- it shows the LLM
    the question, the schema, and one executed-result preview per distinct answer the
    pool produced, and lets it pick one or reject them all. No gold label is shown,
    so "expert" here is a strong label-free verifier, not an oracle.
    """

    name = "llm"

    def __init__(self, judge, run):
        self.judge = judge
        self.run = run

    @property
    def calls(self):
        return getattr(self.judge, "calls", 0)

    def query(self, obs, i):
        return self.judge.query(self.run, obs, int(i))


# --------------------------------------------------------------------------- #
#  6.  The validation loop                                                     #
# --------------------------------------------------------------------------- #
def run_validation(obs, stats, prior, prior_strength, expert, budget=20,
                   select="info_gain", ig_candidates=40, ig_iters=6,
                   warm_iters=200, clamp_on_confirm=False,
                   update_stats_on_confirm=True, seed=0, max_iters=200,
                   tol=1e-8, log=None, pilot=0, epsilon=0.2, temperature=None,
                   **kw):
    """Run EM, then spend ``budget`` expert calls one at a time.

    Per the pipeline: solve to convergence, pick the single highest-information
    object, ask the expert to choose among that object's candidate answers, and

      * if the expert CONFIRMS the current pseudo-label, leave the parameters alone
        and move on to the next object (no re-solve -- this is the shortcut that makes
        a confirming call cheap), or
      * if the expert OVERRULES it, pin the item, refresh ``e`` and ``gamma`` with the
        newly revealed label, and re-solve warm-started until convergence.

    ``select`` is one of

    ``"info_gain"``         Hung et al. Eq. (10): argmax of the LABEL-entropy gain.
    ``"mean_gain"``         argmax of ``A_mu``, the variance reduction of the accuracy
                            estimate itself -- the right objective when the target is a
                            mean rather than a set of labels.
    ``"mean_gain_sampled"`` the same score, but SAMPLED from a softmax with a uniform
                            floor instead of maximised, so every item keeps a known
                            positive inclusion probability and the sample can carry a
                            design-based estimator.
    ``"entropy"``           per-object uncertainty, the paper's own baseline.
    ``"random"``            uniform, the unbiased-by-construction reference.

    ``pilot`` reserves the first calls for a simple random sample of the whole target
    set.  Those items have exactly known inclusion probability ``pilot / N``, which makes
    them a valid AUDIT: :mod:`pooleval.estimators` can turn them into an accuracy
    estimate that is unbiased whatever the model believes.  The remaining budget is spent
    actively and improves the model, not the audit.  Keeping the two roles separate is
    what makes the final number defensible -- an adaptively chosen item has no clean
    marginal inclusion probability, so it cannot do audit duty.
    """
    obs = np.asarray(obs)
    N = obs.shape[1]
    rng = np.random.default_rng(seed)
    constraints = {}
    validated = set()
    inclusion = {}                       # item -> probability it was drawn with
    answers = {}                         # item -> the expert's answer, confirmed or not
    pilot = int(min(pilot, budget, N))
    pilot_items = (list(rng.choice(N, size=pilot, replace=False)) if pilot else [])
    pilot_pi = pilot / float(N) if pilot else None

    state = validated_em(obs, stats, prior, prior_strength, max_iters=max_iters,
                         tol=tol, **kw)
    history = [dict(n_labels=0, acc=state["alpha"].copy(), beta=state["beta"],
                    entropy=state["entropy"], confirmed=0, overruled=0)]
    confirmed = overruled = abstained = 0

    while len(validated) < budget:
        pool = [i for i in range(N) if i not in validated]
        if not pool:
            break
        if pilot_items:                              # spend the audit first
            choice = int(pilot_items.pop())
            inclusion[choice] = pilot_pi
        elif select == "random":
            choice = int(rng.choice(pool))
            inclusion[choice] = len(pool) and 1.0 / len(pool)
        elif select == "entropy":
            choice = max(pool, key=lambda i: item_entropy(state["posterior"][i]))
        elif select in ("info_gain", "mean_gain", "mean_gain_sampled"):
            ranked = sorted(pool, key=lambda i: -item_entropy(state["posterior"][i]))
            cands = ranked if ig_candidates is None else ranked[:int(ig_candidates)]
            scored = acquisition_scores(obs, stats, prior, prior_strength, state,
                                        constraints, cands, ig_iters=ig_iters, **kw)
            key = "entropy_gain" if select == "info_gain" else "mean_gain"
            gains = {o: v[key] for o, v in scored.items()}
            if select == "mean_gain_sampled":
                items = list(gains)
                p = sampling_probabilities(gains, items, epsilon, temperature)
                pick = int(rng.choice(len(items), p=p))
                choice = int(items[pick])
                # the per-draw probability, NOT the marginal inclusion probability:
                # adaptive sampling without replacement makes the marginal depend on the
                # whole history. Recorded for diagnostics; the audit is what the
                # design-based estimate should rest on.
                inclusion[choice] = float(p[pick])
            else:
                choice = max(gains, key=gains.get)
        else:
            raise ValueError(f"unknown selection rule {select!r}")

        consensus = int(state["pseudo_label"][choice])
        answer = expert.query(obs, choice)
        validated.add(choice)
        if answer is None:                      # no candidate was right, call wasted
            abstained += 1
            if log:
                log(f"  item {choice:4d}  expert abstained")
            continue

        answers[choice] = int(answer)
        agrees = int(answer) == consensus
        if update_stats_on_confirm or not agrees:
            stats.add_validated(obs[:, choice], int(answer), consensus)
        if agrees:
            confirmed += 1
            if clamp_on_confirm:
                constraints[choice] = int(answer)
                state = validated_em(obs, stats, prior, prior_strength,
                                     constraints=constraints,
                                     init=dict(alpha=state["alpha"],
                                               beta=state["beta"]),
                                     max_iters=warm_iters, tol=tol, **kw)
        else:
            overruled += 1
            constraints[choice] = int(answer)
            state = validated_em(obs, stats, prior, prior_strength,
                                 constraints=constraints,
                                 init=dict(alpha=state["alpha"], beta=state["beta"]),
                                 max_iters=warm_iters, tol=tol, **kw)
        if log:
            log(f"  item {choice:4d}  consensus={consensus:>10d}  "
                f"expert={int(answer):>10d}  {'confirm' if agrees else 'OVERRULE'}")
        history.append(dict(n_labels=len(validated), acc=state["alpha"].copy(),
                            beta=state["beta"], entropy=state["entropy"],
                            confirmed=confirmed, overruled=overruled))

    audit = sorted(i for i in validated if inclusion.get(i) == pilot_pi) if pilot else []
    state.update(constraints=constraints, validated=sorted(validated),
                 history=history, confirmed=confirmed, overruled=overruled,
                 abstained=abstained, expert_calls=expert.calls,
                 inclusion=inclusion, audit=audit, audit_pi=pilot_pi,
                 answers=answers,
                 e=stats.e, e_excess=stats.e_excess(), gamma_final=stats.gamma)
    return state
