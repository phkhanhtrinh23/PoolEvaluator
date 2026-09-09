"""Textbook multiclass Dawid--Skene: closed forms, monotonicity, recovery."""
import numpy as np

from pooleval.domains.multiclass_ds import (ds_estimates, full_ds,
                                            implied_gamma, one_coin_ds)


def _pool(seed=0, M=7, N=500, K=6, acc_lo=0.45, acc_hi=0.9, confusable=False):
    """Pool drawn FROM the one-coin model (or with a confusion structure)."""
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, K, N)
    acc = np.linspace(acc_lo, acc_hi, M)
    pred = np.empty((M, N), dtype=np.int64)
    for m in range(M):
        ok = rng.random(N) < acc[m]
        if confusable:                      # every class is mistaken for c+1 mod K
            wrong = (gold + 1) % K
        else:                               # uniform over the K-1 wrong classes
            wrong = np.array([rng.choice([c for c in range(K) if c != g]) for g in gold])
        pred[m] = np.where(ok, gold, wrong)
    return pred, gold, K, acc


def test_one_coin_recovers_accuracy_when_the_model_is_true():
    pred, gold, K, acc = _pool()
    out = one_coin_ds(pred, K)
    truth = (pred == gold[None, :]).mean(1)
    assert np.abs(out["acc"] - truth).mean() < 0.05
    assert (out["latent_hat"] == gold).mean() > 0.95


def test_one_coin_log_likelihood_is_monotone():
    pred, gold, K, _ = _pool(seed=1)
    ll = one_coin_ds(pred, K)["log_likelihood"]
    assert np.all(np.diff(ll) > -1e-6)


def test_full_ds_beats_one_coin_when_errors_are_not_uniform():
    """A confusion matrix can learn 'c is mistaken for c+1'; one-coin cannot."""
    pred, gold, K, _ = _pool(seed=2, confusable=True)
    truth = (pred == gold[None, :]).mean(1)
    one = np.abs(one_coin_ds(pred, K)["acc"] - truth).mean()
    fll = np.abs(full_ds(pred, K)["acc"] - truth).mean()
    assert fll < one


def test_full_ds_confusion_rows_are_distributions():
    pred, gold, K, _ = _pool(seed=3)
    pi = full_ds(pred, K)["confusion"]
    assert pi.shape == (pred.shape[0], K, K)
    assert np.allclose(pi.sum(axis=2), 1.0)
    assert np.all(pi > 0)


def test_implied_gamma_is_the_uniform_error_rate():
    assert implied_gamma(2) == 1.0
    assert abs(implied_gamma(5) - 0.25) < 1e-12
    assert abs(implied_gamma(10) - 1.0 / 9) < 1e-12


def test_ds_estimates_returns_both_variants_in_range():
    pred, gold, K, _ = _pool(seed=4)
    est = ds_estimates(dict(pred=pred, n_classes=np.int64(K)))
    assert set(est) == {"DS one-coin (exact)", "DS full (confusion)"}
    for name, a in est.items():
        assert a.shape == (pred.shape[0],) and np.all((a > 0) & (a < 1)), name


def test_estimators_never_see_the_gold_labels():
    """Both take raw predictions only -- permuting gold cannot change the fit."""
    pred, gold, K, _ = _pool(seed=5)
    a = one_coin_ds(pred, K)["acc"]
    b = one_coin_ds(pred, K)["acc"]
    assert np.allclose(a, b)
    est = ds_estimates(dict(pred=pred, n_classes=np.int64(K),
                            gold=np.random.default_rng(0).permutation(gold)))
    assert np.allclose(est["DS one-coin (exact)"], a)
