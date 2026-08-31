"""Collision-aware binary EM on a closed-label-space pool (Trinh_proof.tex case 3)."""
import dataclasses
import numpy as np

from pooleval.config import Config
from pooleval.domains.adapter import from_predictions
from pooleval.domains.collision import (collision_estimates, gamma_from_run,
                                        source_gamma)


def _pool(seed=0, M=6, N=300, K=5, gamma_true=0.8):
    """Pool whose wrong answers collide within a group at a known rate."""
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(M // 3), 3)
    gold = rng.integers(0, K, N)
    shared_wrong = np.array([rng.choice([c for c in range(K) if c != g]) for g in gold])
    acc = np.linspace(0.55, 0.9, M)

    def emit(n_items, gold_v, shared_v):
        out = np.empty((M, n_items), dtype=np.int64)
        for m in range(M):
            ok = rng.random(n_items) < acc[m]
            coll = rng.random(n_items) < gamma_true
            idio = np.array([rng.choice([c for c in range(K) if c != g]) for g in gold_v])
            out[m] = np.where(ok, gold_v, np.where(coll, shared_v, idio))
        return out

    pred = emit(N, gold, shared_wrong)
    n_val = 200
    gold_s = rng.integers(0, K, n_val)
    shared_s = np.array([rng.choice([c for c in range(K) if c != g]) for g in gold_s])
    pred_s = emit(n_val, gold_s, shared_s)
    prob_s = np.eye(K, dtype=np.float32)[pred_s]        # one-hot stand-in for logits
    return dict(pred=pred, gold=gold, group=group,
                prior=(pred_s == gold_s[None, :]).mean(1),
                verifier_guess=np.where(rng.random(N) < 0.7, gold, rng.integers(0, K, N)),
                prob_s=prob_s, src_gold_val=gold_s, n_classes=np.int64(K))


def _run_cfg(pool):
    run = from_predictions(pool["pred"], pool["gold"], pool["group"],
                           prior=pool["prior"], verifier_guess=pool["verifier_guess"])
    cfg = dataclasses.replace(Config(), real_data=True, M=run.M, N=run.N,
                              n_groups=run.n_groups)
    return run, cfg


def test_gamma_detects_shared_error_above_the_independent_null():
    pool = _pool(gamma_true=0.8)
    run, cfg = _run_cfg(pool)
    stats = source_gamma(pool, cfg)
    null = 1.0 / (int(pool["n_classes"]) - 1)
    assert np.all((stats["gamma"] > 0) & (stats["gamma"] < 1))
    assert stats["gamma"].mean() > 3 * null          # colluding pool, gamma_true=0.8
    assert np.all(stats["eligible"] > 0)


def test_gamma_is_near_the_null_when_errors_are_independent():
    pool = _pool(gamma_true=0.0, seed=3)
    run, cfg = _run_cfg(pool)
    g = source_gamma(pool, cfg)["gamma"].mean()
    null = 1.0 / (int(pool["n_classes"]) - 1)
    assert g < 3 * null                              # no collusion -> near 1/(K-1)


def test_source_gamma_never_touches_the_target_labels():
    pool = _pool()
    run, cfg = _run_cfg(pool)
    before = source_gamma(pool, cfg)["gamma"]
    scrambled = dict(pool, gold=np.random.default_rng(9).permutation(pool["gold"]))
    after = source_gamma(scrambled, cfg)["gamma"]
    assert np.allclose(before, after)


def test_gamma_from_run_counts_only_both_wrong_pairs():
    pool = _pool()
    run, _ = _run_cfg(pool)
    pseudo = run.true_class[0].copy()                # pretend model 0 is the pseudo-label
    s = gamma_from_run(run, pseudo, smoothing=0.0)
    for m in range(run.M):
        g = int(run.group[m])
        both_wrong = (run.true_class[m] != 0) & (pseudo != 0)
        assert both_wrong.sum() <= s["eligible"][g]
    assert np.all(s["hits"] <= s["eligible"])


def test_collision_estimates_produce_valid_accuracies_for_every_gamma():
    pool = _pool()
    run, cfg = _run_cfg(pool)
    est, diag = collision_estimates(pool, cfg, run)
    assert set(est) == {"NF binary EM (g=1)", "NF collision (g=null)",
                        "NF collision (g=source)", "NF collision (g=oracle)"}
    for name, acc in est.items():
        acc = np.asarray(acc)
        assert acc.shape == (run.M,), name
        assert np.all(np.isfinite(acc)) and np.all((acc > 0) & (acc < 1)), name
    assert diag["null"] == 1.0 / (int(pool["n_classes"]) - 1)
    assert diag["alpha_strength"] > 0
