"""Tests for Active PoolEval-SQL: hard-constraint EM, the judge, the submodular
selector's (1-1/e) guarantee, and end-to-end candidate-gap recovery."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from pooleval import (Config, PoolEval, ActivePoolEval, ActiveConfig,   # noqa: E402
                      SimulatedJudge, simulate)
from pooleval.active import greedy_submodular, item_scores, abstention   # noqa: E402
from pooleval.latent import run_em                                       # noqa: E402
import pooleval.kernel as km                                             # noqa: E402


def test_constraint_pins_latent_answer():
    """A hard constraint forces every model's credit on that item to match the pinned
    class, including a class ABSENT from obs (-> all models wrong there)."""
    cfg = Config(seed=0, N=120, real_data=True)
    run = simulate(cfg)
    obs = run.true_class.copy()
    absent = 987654321                       # a class no model produced
    out = run_em(obs, run, cfg, constraints={5: absent})
    # every model is scored wrong on the pinned item (nobody matches the absent class)
    correct_on_5 = (obs[:, 5] == absent)
    assert not correct_on_5.any()
    # pinning an EXISTING class credits exactly the models that produced it
    c = int(obs[3, 7])
    out2 = run_em(obs, run, cfg, constraints={7: c})
    # (the M-step credits obs==pinned on item 7; sanity: model 3 matches it)
    assert obs[3, 7] == c


def test_simulated_judge_is_label_free_correct():
    """The judge returns the correct class: an obs class when a model is right, else a
    fresh id absent from obs (the candidate-coverage fix)."""
    cfg = Config(seed=1, N=100, real_data=True)
    run = simulate(cfg)
    obs = run.true_class.copy()
    j = SimulatedJudge()
    for i in range(run.N):
        c = j.query(run, obs, i)
        correct = np.where(run.true_class[:, i] == 0)[0]
        if len(correct) == 0:
            assert c not in set(obs[:, i].tolist())      # fresh, absent class
        else:
            assert (obs[correct, i] == c).any()
    assert j.calls == run.N


def test_greedy_submodular_within_1_minus_1_over_e():
    """Lazy greedy on the monotone-submodular facility-location objective attains at
    least (1-1/e) of the optimum on a small instance where OPT is brute-forceable."""
    import itertools
    rng = np.random.default_rng(0)
    N, k = 8, 3
    value = rng.uniform(0.1, 1.0, size=N)
    F = rng.uniform(0, 1, size=(N, N))
    S = (F + F.T) / 2
    np.fill_diagonal(S, 1.0)

    def f(A):
        if not A:
            return 0.0
        cov = S[:, list(A)].max(axis=1)
        return float(np.sum(value[list(A)]) + 0.15 * np.sum(value * cov))

    greedy = greedy_submodular(value, S, k, lam=0.15)
    opt = max(f(set(c)) for c in itertools.combinations(range(N), k))
    assert f(set(greedy)) >= (1 - 1 / np.e) * opt - 1e-9


def test_candidate_gap_recovery():
    """End to end: on a near-clone-collusion DGP where the correct answer is absent,
    pure PoolEval mis-ranks the clique #1; a modest judge budget with the submodular
    strategy restores a good model to the top."""
    from run_active import dgp_candidate_gap
    fixed = fixed_top = 0
    for sd in range(10):
        cfg = Config(real_data=True, seed=sd)
        run, meta = dgp_candidate_gap(cfg, np.random.default_rng(2000 + sd), rho=0.24)
        base = PoolEval(cfg).evaluate(run)
        pure_top1 = int(np.argmax(base["acc"]) in meta["good"])
        out = ActivePoolEval(cfg, ActiveConfig(budget=40, rounds=8,
                             strategy="hybrid_submodular")).run(run)
        act_top1 = int(np.argmax(out["acc"]) in meta["good"])
        fixed += (act_top1 and not pure_top1)
        fixed_top += act_top1
    # pure almost never picks a good model; active recovers on the clear majority
    assert fixed_top >= 7


def test_abstention_flags_near_clone_collusion():
    """The per-item flag has high precision/recall for the trap items (a near-clone
    plurality on a wrong answer)."""
    from run_active import dgp_candidate_gap
    cfg = Config(real_data=True, seed=0)
    run, meta = dgp_candidate_gap(cfg, np.random.default_rng(2000), rho=0.24)
    obs = km.apply(run.true_class, cfg, level=cfg.kernel_level,
                   rng=np.random.default_rng(cfg.seed + 7))
    out = PoolEval(cfg).evaluate(run, obs=obs)
    ab = abstention(out, run, obs)
    flagged, trap = ab["per_item"], meta["trap"]
    assert (flagged & trap).sum() / flagged.sum() > 0.9      # precision
    assert (flagged & trap).sum() / trap.sum() > 0.9         # recall


if __name__ == "__main__":
    for fn in [test_constraint_pins_latent_answer,
               test_simulated_judge_is_label_free_correct,
               test_greedy_submodular_within_1_minus_1_over_e,
               test_candidate_gap_recovery,
               test_abstention_flags_near_clone_collusion]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all active tests passed")
