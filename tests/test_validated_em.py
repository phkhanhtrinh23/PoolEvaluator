"""Numerical re-derivation of every claim in the validated-EM reformulation.

Nothing here is asserted from the algebra alone: each maximiser is checked against a
brute-force optimum, each reduction against the formula it is supposed to reduce to.
"""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from pooleval.new_formulation import collision_agreement_em
from pooleval.validated_em import (EPS, LabeledStatistics, LatentPlan, NoisyExpert,
                                   OracleExpert, _clip, correctness_em, excess_collision,
                                   gamma_counts, gamma_from_counts,
                                   acquisition_scores, information_gain, item_entropy,
                                   latent_posterior, mean_information_gain,
                                   run_validation, set_entropy, validated_em,
                                   vote_discount, wrong_collision_counts,
                                   wrong_collision_matrix)


# --------------------------------------------------------------------------- #
#  e -- the pairwise correlated-error matrix                                   #
# --------------------------------------------------------------------------- #
def test_collision_matrix_matches_the_definition_item_by_item():
    #            i0  i1  i2  i3
    tc = np.array([[0,  1,  2,  1],      # m0
                   [1,  1,  2,  0],      # m1
                   [0,  1,  3, -7]])     # m2
    e = wrong_collision_matrix(tc)
    M, N = tc.shape
    want = np.zeros((M, M))
    for j in range(M):
        for k in range(M):
            hits = sum(1 for i in range(N)
                       if tc[j, i] == tc[k, i] and tc[j, i] > 0 and tc[k, i] > 0)
            want[j, k] = hits / N
    assert np.allclose(e, want)
    # m0 and m1 share wrong answers on i1 and i2 -> 2/4.
    assert e[0, 1] == pytest.approx(0.5)
    # the diagonal is just "wrong on a shareable class".
    assert e[0, 0] == pytest.approx(3 / 4)


def test_execution_errors_never_collide():
    """Negative ids are per-cell error markers; two crashes are not an agreement."""
    tc = np.array([[-1, -1], [-1, -1]])
    counts, n = wrong_collision_counts(tc)
    assert counts.sum() == 0.0 and n == 2


def test_excess_collision_subtracts_the_cross_group_chance_level():
    e = np.array([[0.4, 0.30, 0.10],
                  [0.30, 0.4, 0.12],
                  [0.10, 0.12, 0.4]])
    group = np.array([0, 0, 1])
    ex = excess_collision(e, group)
    cross = (0.10 + 0.12 + 0.10 + 0.12) / 4      # the four cross-group off-diagonals
    assert ex[0, 1] == pytest.approx(0.30 - cross)
    assert np.allclose(np.diag(ex), 0.0)         # a model never discounts itself
    assert ex.min() >= 0.0                       # only excess correlation penalised


# --------------------------------------------------------------------------- #
#  the discount reduces to the group formula it generalises                    #
# --------------------------------------------------------------------------- #
def test_vote_discount_reproduces_the_group_loading_formula_exactly():
    """With a constant within-group excess u and zero across groups, the pairwise
    discount must equal pooleval.latent's 1 / (1 + u (n_{g,k} - 1))."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        M = int(rng.integers(4, 10))
        group = rng.integers(0, 3, size=M)
        column = rng.integers(0, 3, size=M)
        u = float(rng.uniform(0.0, 2.0))
        e_excess = u * (group[:, None] == group[None, :]).astype(float)
        np.fill_diagonal(e_excess, 0.0)

        got = vote_discount(column, e_excess)
        n_gk = np.array([np.sum((group == group[m]) & (column == column[m]))
                         for m in range(M)], dtype=float)
        want = 1.0 / (1.0 + u * (n_gk - 1.0))
        assert np.allclose(got, want)


def test_zero_correlation_leaves_every_vote_at_full_weight():
    column = np.array([1, 1, 1, 2])
    assert np.allclose(vote_discount(column, np.zeros((4, 4))), 1.0)


# --------------------------------------------------------------------------- #
#  gamma                                                                       #
# --------------------------------------------------------------------------- #
def test_gamma_both_wrong_mode_is_one_minus_the_collision_rate():
    """Among items where the model AND the pseudo-label are both wrong, C = 0 exactly
    when the two wrong answers FAIL to collide."""
    tc = np.array([[1, 2, 3, 4]])
    yhat = np.array([1, 5, 3, 6])          # all wrong; collides with the model on i0,i2
    num, den = gamma_counts(tc, yhat, mode="both_wrong")
    assert den[0] == 4 and num[0] == 2     # differs on i1 and i3
    assert gamma_from_counts(num, den, smoothing=0.0)[0] == pytest.approx(0.5)


def test_conditional_gamma_reproduces_d_of_beta_from_the_tex():
    """P(C=0|Z=0) = beta + (1-beta) gamma^both, and with gamma^both = 1 - gamma^coll
    that is exactly d_j(beta) = 1 - (1-beta) gamma^coll."""
    from pooleval.validated_em import gamma_to_conditional
    beta = 0.64
    gamma_coll = np.array([0.1, 0.5, 0.9])
    got = gamma_to_conditional(1.0 - gamma_coll, beta)
    assert np.allclose(got, 1.0 - (1.0 - beta) * gamma_coll)


def test_gamma_model_wrong_mode_counts_the_right_events():
    tc = np.array([[0, 1, 2, 3],
                   [1, 1, 0, 3]])
    yhat = np.array([0, 1, 0, 9])
    num, den = gamma_counts(tc, yhat, mode="model_wrong")
    # model 0 is wrong on i1,i2,i3 (classes 1,2,3); it disagrees with yhat on i2,i3.
    assert den[0] == 3 and num[0] == 2
    # model 1 is wrong on i0,i1,i3 (classes 1,1,3); it disagrees with yhat on i0,i3.
    assert den[1] == 3 and num[1] == 2
    g = gamma_from_counts(num, den, smoothing=1.0)
    assert g[0] == pytest.approx(3 / 5) and g[1] == pytest.approx(3 / 5)


def test_gamma_pseudo_wrong_mode_conditions_on_the_pseudo_label_instead():
    tc = np.array([[0, 1, 2, 3],
                   [1, 1, 0, 3]])
    yhat = np.array([0, 1, 0, 9])          # wrong on i1 and i3 (ids 1 and 9 != 0)
    num, den = gamma_counts(tc, yhat, mode="pseudo_wrong")
    assert np.all(den == 2)
    # model 0 disagrees with yhat on i3 among those two items; model 1 likewise.
    assert num[0] == 1 and num[1] == 1


def test_gamma_falls_back_to_one_half_when_a_model_is_never_wrong():
    assert gamma_from_counts([0.0], [0.0], smoothing=1.0)[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
#  the M-steps are the true maximisers of Q                                    #
# --------------------------------------------------------------------------- #
def test_alpha_m_step_is_the_argmax_of_q_alpha():
    """Q_alpha = A log a + B log(1-a) with A = s pi + sum tau, B = s(1-pi) + N - sum tau
    (the .tex lemma). Its maximiser must be A / (A + B) = (sum tau + s pi) / (N + s)."""
    rng = np.random.default_rng(1)
    tau_row = rng.uniform(0.05, 0.95, size=60)
    s, pi, N = 120.0, 0.72, 60
    closed = (tau_row.sum() + s * pi) / (N + s)

    def neg_q(a):
        a = float(np.clip(a, EPS, 1 - EPS))
        return -float(tau_row.sum() * np.log(a) + (1 - tau_row).sum() * np.log(1 - a)
                      + s * (pi * np.log(a) + (1 - pi) * np.log(1 - a)))

    numeric = minimize_scalar(neg_q, bounds=(EPS, 1 - EPS), method="bounded",
                              options={"xatol": 1e-12}).x
    assert closed == pytest.approx(numeric, abs=1e-7)


def test_beta_m_step_is_closed_form_and_maximises_q_beta():
    """The collision model needed a bounded 1-D search here. Freezing the Z=0 branch
    removes beta from it, so the maximiser is a ratio of expected counts."""
    rng = np.random.default_rng(2)
    M, N = 6, 40
    C = (rng.random((M, N)) < 0.6).astype(float)
    tau = rng.uniform(0.05, 0.95, size=(M, N))
    b0, bs = 0.8, 30.0

    def neg_q(b):
        b = float(np.clip(b, EPS, 1 - EPS))
        return -float((tau * (C * np.log(b) + (1 - C) * np.log(1 - b))).sum()
                      + bs * (b0 * np.log(b) + (1 - b0) * np.log(1 - b)))

    numeric = minimize_scalar(neg_q, bounds=(EPS, 1 - EPS), method="bounded",
                              options={"xatol": 1e-12}).x
    closed = (float((tau * C).sum()) + bs * b0) / (float(tau.sum()) + bs)
    assert closed == pytest.approx(numeric, abs=1e-7)


def test_correctness_em_increases_the_anchored_log_posterior():
    rng = np.random.default_rng(3)
    C = (rng.random((8, 120)) < 0.65).astype(float)
    out = correctness_em(C, gamma=rng.uniform(0.5, 0.95, size=8),
                         prior=rng.uniform(0.4, 0.9, size=8), prior_strength=100.0,
                         beta_init=0.7, beta_strength=40.0)
    trace = out["log_posterior"]
    assert np.all(np.diff(trace) > -1e-8)
    assert out["n_iters"] < 200            # it converges rather than hitting the cap


# --------------------------------------------------------------------------- #
#  relation to the collision formulation it replaces                           #
# --------------------------------------------------------------------------- #
def test_e_step_matches_the_collision_model_when_gamma_equals_d_of_beta():
    """gamma_new = 1 - (1 - beta) gamma_collision is exactly d_j(beta) from the .tex,
    so at that value the two posteriors over Z must coincide cell for cell."""
    rng = np.random.default_rng(4)
    M, N = 5, 30
    C = (rng.random((M, N)) < 0.6).astype(float)
    alpha = rng.uniform(0.3, 0.9, size=M)
    beta = 0.73
    gamma_coll = rng.uniform(0.1, 0.9, size=M)
    gamma_new = 1.0 - (1.0 - beta) * gamma_coll

    like_z1 = np.where(C == 1.0, beta, 1.0 - beta)
    old_p_c1_z0 = (1.0 - beta) * gamma_coll[:, None]
    old_like_z0 = np.where(C == 1.0, old_p_c1_z0, 1.0 - old_p_c1_z0)
    new_like_z0 = np.where(C == 1.0, 1.0 - gamma_new[:, None], gamma_new[:, None])
    assert np.allclose(old_like_z0, new_like_z0)

    num = alpha[:, None] * like_z1
    tau_old = num / (num + (1 - alpha[:, None]) * old_like_z0)
    tau_new = num / (num + (1 - alpha[:, None]) * new_like_z0)
    assert np.allclose(tau_old, tau_new)


def test_gamma_one_recovers_the_original_no_collision_model():
    """gamma^{collision} = 1 means two wrong answers always coincide, so
    P(C = 0 | Z = 0) = 1 - (1 - beta) = beta and the .tex's original binary model is
    the special case gamma_new = beta."""
    rng = np.random.default_rng(5)
    C = (rng.random((4, 50)) < 0.6).astype(float)
    beta = 0.8
    gamma_new = np.full(4, 1.0 - (1.0 - beta) * 1.0)
    assert np.allclose(gamma_new, beta)
    like_z0 = np.where(C == 1.0, 1.0 - gamma_new[:, None], gamma_new[:, None])
    original = np.where(C == 1.0, 1.0 - beta, beta)   # P(C=1|Z=0) = 1-beta
    assert np.allclose(like_z0, original)


# --------------------------------------------------------------------------- #
#  latent posterior, entropy, information gain                                 #
# --------------------------------------------------------------------------- #
def test_constraints_pin_the_latent_answer_to_one_hot():
    obs = np.array([[1, 1], [2, 1], [1, 3]])
    post = latent_posterior(obs, np.full(3, 0.7), np.zeros((3, 3)),
                            constraints={0: 99})
    assert post[0] == {99: 1.0}
    assert item_entropy(post[0]) == 0.0
    assert set(post[1]) == {1, 3}


def test_unanimous_items_have_zero_entropy_and_split_items_have_more():
    obs = np.array([[1, 1], [1, 2], [1, 3]])
    post = latent_posterior(obs, np.full(3, 0.7), np.zeros((3, 3)))
    assert item_entropy(post[0]) == pytest.approx(0.0)
    assert item_entropy(post[1]) > 0.9
    assert set_entropy(post) == pytest.approx(item_entropy(post[1]))


def test_information_gain_is_near_zero_for_a_unanimous_item():
    """A unanimous item has a one-hot U(o, .), so there is only one answer to pretend
    and its own entropy is already 0. The gain is not EXACTLY zero -- pinning still
    makes tau hard there, which nudges every alpha and hence every other item -- but
    it must be orders of magnitude below a genuinely contested item's."""
    rng = np.random.default_rng(6)
    obs = rng.integers(1, 4, size=(3, 8))
    obs[:, 0] = 1                                   # item 0 unanimous
    obs[:, 1] = np.array([1, 2, 3])                 # item 1 three-way split
    stats = _toy_stats(obs.shape[0])
    prior = np.full(3, 0.7)
    state = validated_em(obs, stats, prior, 50.0, max_iters=30)
    gains = information_gain(obs, stats, prior, 50.0, state, {}, [0, 1], ig_iters=30)
    assert abs(gains[0]) < 0.01
    assert gains[1] > 10 * abs(gains[0])


def _toy_stats(M, N=40, seed=0):
    rng = np.random.default_rng(seed)
    tc = rng.integers(0, 3, size=(M, N))
    yhat = rng.integers(0, 3, size=N)
    return LabeledStatistics(tc, yhat, group=np.arange(M))


# --------------------------------------------------------------------------- #
#  growing the labeled statistics with validated items                         #
# --------------------------------------------------------------------------- #
def test_the_update_is_literally_new_equals_old_plus_temp_over_two():
    tc = np.array([[0, 1], [1, 1], [0, 0]])
    stats = LabeledStatistics(tc, np.array([0, 1]), group=np.array([0, 0, 1]))
    e_old = stats.e.copy()

    # models 0 and 1 both answer 5 while the truth is 7 -> a collision for the pair;
    # model 2 is right, so it collides with nobody.
    stats.add_validated(answers=np.array([5, 5, 7]), truth=7, consensus_label=5)
    temp_e = np.array([[1.0, 1.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0]])
    assert np.allclose(stats.e, 0.5 * (e_old + temp_e))
    assert stats.n_validated == 1


def test_a_model_with_no_eligible_validated_item_keeps_its_old_gamma():
    tc = np.array([[0, 1], [1, 1], [0, 0]])
    stats = LabeledStatistics(tc, np.array([0, 1]), group=np.array([0, 0, 1]))
    g0 = stats.gamma.copy()
    # model 2 is correct on the validated item, so "both wrong" never fires for it.
    # model 0 is wrong and DISAGREES with the consensus, so its temp gamma is 2/3.
    stats.add_validated(answers=np.array([6, 5, 7]), truth=7, consensus_label=5)
    assert stats.gamma[2] == pytest.approx(g0[2])
    assert stats.gamma[0] == pytest.approx(0.5 * (g0[0] + 2 / 3))


def test_pooled_counts_rule_weights_every_labeled_item_equally():
    tc = np.array([[0, 1], [1, 1], [0, 0]])
    avg = LabeledStatistics(tc, np.array([0, 1]), group=np.array([0, 0, 1]),
                            update_rule="average")
    cnt = LabeledStatistics(tc, np.array([0, 1]), group=np.array([0, 0, 1]),
                            update_rule="counts")
    for _ in range(4):
        for st in (avg, cnt):
            st.add_validated(np.array([5, 5, 7]), truth=7, consensus_label=5)
    # after four repeats the geometric filter is almost all the way onto temp,
    # while pooled counts still carry the two source items.
    assert avg.e[0, 1] > cnt.e[0, 1]
    assert cnt.e[0, 1] == pytest.approx(5 / 6)   # 1 source collision + 4 new, over 6 items


def test_pinning_does_not_inflate_gamma_because_the_consensus_label_is_used():
    """Using the pinned label as the pseudo-label would make every wrong model
    disagree by construction and push gamma to 1. The consensus label avoids that."""
    tc = np.array([[0, 0], [1, 1]])
    stats = LabeledStatistics(tc, np.array([0, 0]), group=np.array([0, 1]))
    for _ in range(30):
        stats.add_validated(np.array([4, 4]), truth=7, consensus_label=4)
    assert stats.gamma[1] < 0.2                      # wrong but agreeing -> low gamma


# --------------------------------------------------------------------------- #
#  experts and the validation loop                                             #
# --------------------------------------------------------------------------- #
def test_oracle_expert_picks_the_class_of_a_correct_model():
    true_class = np.array([[0, 1], [3, 1]])
    obs = np.array([[11, 22], [33, 22]])
    e = OracleExpert(true_class)
    assert e.query(obs, 0) == 11          # model 0 is correct at item 0
    assert e.query(obs, 1) is None        # nobody is correct at item 1
    assert e.calls == 2


def test_oracle_expert_can_reject_every_candidate_when_allowed():
    true_class = np.array([[1], [2]])
    obs = np.array([[11], [22]])
    answer = OracleExpert(true_class, allow_none=True).query(obs, 0)
    assert answer not in (11, 22)


def test_noisy_expert_degrades_towards_the_stated_accuracy():
    true_class = np.tile(np.array([[0], [1], [2]]), (1, 400))
    obs = np.tile(np.array([[7], [8], [9]]), (1, 400))
    e = NoisyExpert(true_class, accuracy=0.75, seed=0)
    hits = np.mean([e.query(obs, i) == 7 for i in range(400)])
    assert 0.70 < hits < 0.80


def test_validation_loop_spends_exactly_the_budget_and_only_refits_on_overrule():
    rng = np.random.default_rng(7)
    M, N = 6, 25
    true_class = rng.integers(0, 3, size=(M, N))
    obs = true_class.copy()
    stats = _toy_stats(M, seed=1)
    out = run_validation(obs, stats, np.full(M, 0.6), 40.0,
                         expert=OracleExpert(true_class), budget=5,
                         select="entropy", max_iters=30, warm_iters=10)
    assert out["expert_calls"] == 5
    assert len(out["validated"]) == 5
    assert out["confirmed"] + out["overruled"] + out["abstained"] == 5
    # only overruled items are pinned; confirmations leave the parameters alone.
    assert len(out["constraints"]) == out["overruled"]


def test_info_gain_selection_runs_and_picks_an_unvalidated_item():
    rng = np.random.default_rng(8)
    M, N = 5, 12
    true_class = rng.integers(0, 3, size=(M, N))
    obs = true_class.copy()
    stats = _toy_stats(M, seed=2)
    out = run_validation(obs, stats, np.full(M, 0.6), 30.0,
                         expert=OracleExpert(true_class), budget=3,
                         select="info_gain", ig_candidates=5, ig_iters=3,
                         max_iters=20, warm_iters=8)
    assert len(set(out["validated"])) == 3
    assert all(0 <= i < N for i in out["validated"])


def test_alpha_stays_in_the_unit_interval_and_tracks_the_anchor_without_data():
    """With N -> 0 evidence the MAP update returns the anchor itself."""
    M = 4
    obs = np.zeros((M, 1), dtype=int)
    stats = _toy_stats(M, seed=3)
    prior = np.array([0.9, 0.8, 0.4, 0.2])
    out = validated_em(obs, stats, prior, 1e6, max_iters=5)
    assert np.allclose(out["alpha"], prior, atol=1e-3)
    assert np.all((out["alpha"] > 0) & (out["alpha"] < 1))


# --------------------------------------------------------------------------- #
#  The strict .tex inner loop: pseudo-labels fixed, EM only over (alpha, beta)  #
# --------------------------------------------------------------------------- #
def test_fixed_pseudo_mode_holds_the_agreement_matrix_constant():
    """In "fixed" mode the inner loop is a real EM on one likelihood, so the observed
    log-likelihood must be monotone. In "joint" mode the pseudo-labels move between
    sweeps, so it is coordinate ascent on two blocks and monotonicity is not claimed."""
    rng = np.random.default_rng(11)
    M, N = 6, 60
    obs = rng.integers(0, 3, size=(M, N))
    stats = _toy_stats(M, seed=4)
    prior = rng.uniform(0.4, 0.8, size=M)
    out = validated_em(obs, stats, prior, 40.0, max_iters=200, pseudo_mode="fixed")
    trace = out["log_likelihood"]
    assert np.all(np.diff(trace) > -1e-8)
    assert out["n_iters"] < 200


def test_fixed_mode_agrees_with_correctness_em_on_the_same_agreement_matrix():
    """Wrapping the expert loop around correctness_em is exactly what "fixed" does."""
    rng = np.random.default_rng(12)
    M, N = 5, 50
    obs = rng.integers(0, 3, size=(M, N))
    stats = _toy_stats(M, seed=5)
    prior = rng.uniform(0.4, 0.8, size=M)
    joint = validated_em(obs, stats, prior, 30.0, max_iters=300, pseudo_mode="fixed")

    plan = LatentPlan(obs, stats.e_excess())
    yhat, _, _ = plan.solve(_clip(prior))
    C = (obs == yhat[None, :]).astype(float)
    direct = correctness_em(C, stats.conditional_gamma(), prior, 30.0,
                            beta_init=stats.pseudo_accuracy, max_iters=300)
    assert np.allclose(joint["alpha"], direct["alpha"], atol=1e-6)
    assert joint["beta"] == pytest.approx(direct["beta"], abs=1e-6)


def test_both_pseudo_modes_run_inside_the_validation_loop():
    rng = np.random.default_rng(13)
    M, N = 5, 20
    true_class = rng.integers(0, 3, size=(M, N))
    obs = true_class.copy()
    for mode in ("fixed", "joint"):
        out = run_validation(obs, _toy_stats(M, seed=6), np.full(M, 0.6), 25.0,
                             expert=OracleExpert(true_class), budget=4,
                             select="entropy", max_iters=50, warm_iters=25,
                             pseudo_mode=mode)
        assert out["expert_calls"] == 4
        assert np.all((out["acc"] > 0) & (out["acc"] < 1))


def test_unknown_pseudo_mode_is_rejected():
    with pytest.raises(ValueError):
        validated_em(np.zeros((2, 2), dtype=int), _toy_stats(2), np.full(2, 0.5),
                     10.0, pseudo_mode="nonsense")


# --------------------------------------------------------------------------- #
#  Mean-targeted acquisition                                                   #
# --------------------------------------------------------------------------- #
def test_mean_gain_reduces_to_p_times_one_minus_p_times_leverage_squared():
    """With exactly two possible answers, A_mu(q) must equal p(1-p)(m1-m0)^2 summed over
    models -- an item-uncertainty factor times a GLOBAL LEVERAGE factor."""
    rng = np.random.default_rng(20)
    M, N = 5, 30
    obs = rng.integers(0, 2, size=(M, N))         # binary answers -> two outcomes
    stats = _toy_stats(M, seed=7)
    prior = rng.uniform(0.4, 0.8, size=M)
    s = 25.0
    state = validated_em(obs, stats, prior, s, max_iters=100)

    q = 0
    post = state["posterior"][q]
    assert len(post) == 2
    scored = acquisition_scores(obs, stats, prior, s, state, {}, [q], ig_iters=20)

    # rebuild the two hypothetical fits by hand
    init = dict(alpha=state["alpha"], beta=state["beta"])
    labels = sorted(post)
    fits = [validated_em(obs, stats, prior, s, constraints={q: int(l)}, init=init,
                         max_iters=20, plan=state["plan"], report_posterior=False)
            for l in labels]
    p = post[labels[0]] / (post[labels[0]] + post[labels[1]])
    m1, m0 = fits[0]["alpha"], fits[1]["alpha"]
    closed = float(np.sum(p * (1 - p) * (m1 - m0) ** 2))
    assert scored[q]["mean_gain"] == pytest.approx(closed, rel=1e-9, abs=1e-12)


def test_mean_gain_matches_the_covariance_form():
    """A_mu(q) = Cov(mu, Z_q)^2 / Var(Z_q), since m1 - m0 = Cov / (p(1-p))."""
    rng = np.random.default_rng(21)
    p = 0.3
    m1, m0 = np.array([0.82, 0.41]), np.array([0.55, 0.39])
    direct = p * (1 - p) * (m1 - m0) ** 2
    cov = p * (1 - p) * (m1 - m0)                  # Cov(alpha_j, Z_q)
    via_cov = cov ** 2 / (p * (1 - p))
    assert np.allclose(direct, via_cov)


def test_mean_gain_is_zero_when_the_answer_cannot_change_anything():
    """A unanimous item has one possible answer, so there is no variance to reduce."""
    rng = np.random.default_rng(22)
    obs = rng.integers(0, 3, size=(4, 10))
    obs[:, 0] = 1                                  # unanimous
    stats = _toy_stats(4, seed=8)
    prior = np.full(4, 0.7)
    state = validated_em(obs, stats, prior, 20.0, max_iters=50)
    scored = acquisition_scores(obs, stats, prior, 20.0, state, {}, [0], ig_iters=10)
    assert scored[0]["n_outcomes"] == 1
    assert scored[0]["mean_gain"] == pytest.approx(0.0, abs=1e-12)


def test_the_two_objectives_rank_items_differently():
    """If label entropy and accuracy variance always agreed there would be nothing to
    choose between them. They do not."""
    rng = np.random.default_rng(23)
    M, N = 8, 40
    obs = rng.integers(0, 3, size=(M, N))
    obs[:6, 5] = 2                                 # a bloc agreeing on one answer
    stats = _toy_stats(M, seed=9)
    prior = rng.uniform(0.4, 0.8, size=M)
    state = validated_em(obs, stats, prior, 30.0, max_iters=100)
    scored = acquisition_scores(obs, stats, prior, 30.0, state, {}, list(range(N)),
                                ig_iters=6)
    by_entropy = sorted(scored, key=lambda o: -scored[o]["entropy_gain"])
    by_mean = sorted(scored, key=lambda o: -scored[o]["mean_gain"])
    assert by_entropy != by_mean


def test_every_selection_rule_spends_the_budget():
    rng = np.random.default_rng(24)
    M, N = 5, 25
    true_class = rng.integers(0, 3, size=(M, N))
    obs = true_class.copy()
    for rule in ("info_gain", "mean_gain", "mean_gain_sampled", "entropy", "random"):
        out = run_validation(obs, _toy_stats(M, seed=10), np.full(M, 0.6), 20.0,
                             expert=OracleExpert(true_class), budget=5, select=rule,
                             ig_candidates=8, ig_iters=3, max_iters=40, warm_iters=20)
        assert out["expert_calls"] == 5, rule
        assert len(out["validated"]) == 5, rule


def test_the_pilot_audit_has_an_exactly_known_inclusion_probability():
    rng = np.random.default_rng(25)
    M, N = 5, 40
    true_class = rng.integers(0, 3, size=(M, N))
    obs = true_class.copy()
    out = run_validation(obs, _toy_stats(M, seed=11), np.full(M, 0.6), 20.0,
                         expert=OracleExpert(true_class), budget=12, pilot=8,
                         select="mean_gain", ig_candidates=8, ig_iters=3,
                         max_iters=40, warm_iters=20)
    assert len(out["audit"]) == 8
    assert out["audit_pi"] == pytest.approx(8 / N)
    # every audited item carries that probability; actively chosen ones do not
    assert all(out["inclusion"][i] == pytest.approx(8 / N) for i in out["audit"])


def test_sampled_selection_records_a_positive_probability_for_every_pick():
    rng = np.random.default_rng(26)
    M, N = 5, 25
    true_class = rng.integers(0, 3, size=(M, N))
    out = run_validation(true_class.copy(), _toy_stats(M, seed=12), np.full(M, 0.6),
                         20.0, expert=OracleExpert(true_class), budget=6,
                         select="mean_gain_sampled", ig_candidates=10, ig_iters=3,
                         max_iters=40, warm_iters=20)
    assert len(out["inclusion"]) == 6
    assert all(0.0 < p <= 1.0 for p in out["inclusion"].values())


# --------------------------------------------------------------------------- #
#  The pairwise route to P(C=0|Z=0)                                            #
# --------------------------------------------------------------------------- #
def test_pairwise_counts_put_the_classifier_itself_on_the_diagonal():
    """A classifier always agrees with itself, so same[j,j] == wrong[j]. That term
    belongs: if j's own answer IS the pseudo-label then C=1 with certainty."""
    from pooleval.validated_em import pairwise_wrong_counts
    tc = np.array([[0, 1, 2], [1, 1, 0]])
    same, wrong = pairwise_wrong_counts(tc)
    assert np.allclose(np.diag(same), wrong)
    assert wrong[0] == 2 and wrong[1] == 2          # j=0 wrong on i1,i2; j=1 on i0,i1
    assert same[0, 1] == 1                          # they share the wrong answer 1 at i1


def test_pairwise_needs_no_both_wrong_condition():
    """r^j = r^k with j wrong forces k wrong too, which is why this conditioning needs
    no beta correction."""
    from pooleval.validated_em import pairwise_wrong_counts
    tc = np.array([[5, 0], [5, 3]])
    same, _ = pairwise_wrong_counts(tc)
    assert same[0, 1] == 1        # i0: both gave 5, both wrong -- counted
    assert same[1, 0] == 1        # symmetric here because both are wrong at i0


def test_pairwise_gamma_is_one_minus_the_row_mean_agreement():
    from pooleval.validated_em import gamma_from_pairwise
    same = np.array([[4.0, 2.0], [2.0, 4.0]])
    wrong = np.array([4.0, 4.0])
    g = gamma_from_pairwise(same, wrong, smoothing=0.0)
    assert g[0] == pytest.approx(1.0 - np.mean([4 / 4, 2 / 4]))


def test_pairwise_gamma_does_not_depend_on_the_pseudo_labels():
    """The structural property that motivates it: invariant to the EM state, because it
    is a function of (answers, gold) only."""
    rng = np.random.default_rng(30)
    tc = rng.integers(0, 3, size=(5, 40))
    a = LabeledStatistics(tc, rng.integers(0, 3, size=40), gamma_mode="pairwise")
    b = LabeledStatistics(tc, rng.integers(0, 3, size=40), gamma_mode="pairwise")
    assert np.allclose(a.gamma, b.gamma)
    # the counted route DOES move when the pseudo-labels move
    c = LabeledStatistics(tc, np.zeros(40, dtype=int), gamma_mode="model_wrong")
    d = LabeledStatistics(tc, np.ones(40, dtype=int), gamma_mode="model_wrong")
    assert not np.allclose(c.gamma, d.gamma)


def test_pairwise_needs_no_conversion():
    rng = np.random.default_rng(31)
    tc = rng.integers(0, 3, size=(4, 30))
    st = LabeledStatistics(tc, rng.integers(0, 3, size=30), gamma_mode="pairwise")
    assert np.allclose(st.conditional_gamma(), st.gamma)


def test_pairwise_refreshes_with_validated_items():
    tc = np.array([[0, 1], [1, 1], [0, 0]])
    st = LabeledStatistics(tc, np.array([0, 1]), group=np.array([0, 0, 1]),
                           gamma_mode="pairwise")
    before = st.gamma.copy()
    st.add_validated(np.array([5, 5, 7]), truth=7, consensus_label=5)
    assert not np.allclose(st.gamma, before)
    assert np.all((st.gamma > 0) & (st.gamma < 1))


def test_all_three_modes_run_end_to_end():
    rng = np.random.default_rng(32)
    M, N = 5, 25
    true_class = rng.integers(0, 3, size=(M, N))
    for mode in ("model_wrong", "both_wrong", "pairwise"):
        st = LabeledStatistics(_toy_stats(M, seed=13).obs if False else
                               rng.integers(0, 3, size=(M, 40)),
                               rng.integers(0, 3, size=40), gamma_mode=mode)
        out = run_validation(true_class.copy(), st, np.full(M, 0.6), 20.0,
                             expert=OracleExpert(true_class), budget=4,
                             select="entropy", max_iters=40, warm_iters=20)
        assert out["expert_calls"] == 4
        assert np.all((out["acc"] > 0) & (out["acc"] < 1)), mode
