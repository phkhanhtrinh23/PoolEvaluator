"""RQ-D7: which estimator wins in which regime -- small N, large K, imbalanced classes?

The choice between PoolEval's single reliability scalar and full Dawid-Skene's
per-model confusion matrix is a bias/variance trade, and the crossover is set by
one quantity: observations per free parameter.

  PoolEval (anchor, collision-aware) : M alphas + 1 beta + G group loadings
                                       (+ G gammas FIXED from source) -- independent of K
  full DS                            : M*K*(K-1) + (K-1) -- quadratic in K

So full DS should win when data is plentiful relative to K, and lose when the
target set is small, K is large, or rare classes starve individual confusion rows.
This script measures all three axes. Scenarios 1 and 3 use the real cached pools;
scenario 2 needs K beyond what the real datasets offer, so it simulates -- and is
labelled as such.
"""
import argparse, dataclasses, json, os, sys, warnings
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pooleval.config import Config                                   # noqa: E402
from pooleval.inference import PoolEval                              # noqa: E402
from pooleval.domains.adapter import from_predictions                # noqa: E402
from pooleval.domains.multiclass_ds import full_ds, one_coin_ds      # noqa: E402

CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")
METHODS = ("PoolEval", "DS one-coin", "DS full")


def evaluate(pred, gold, group, prior, n_classes, verifier=None):
    run = from_predictions(pred, gold, group, prior=prior, verifier_guess=verifier)
    cfg = dataclasses.replace(Config(), real_data=True, M=run.M, N=run.N,
                              n_groups=run.n_groups)
    truth = run.true_acc
    est = {"PoolEval": PoolEval(cfg).evaluate(run, obs=run.true_class)["acc"],
           "DS one-coin": one_coin_ds(pred, n_classes)["acc"],
           "DS full": full_ds(pred, n_classes)["acc"]}
    return {k: dict(mae=float(np.abs(v - truth).mean()),
                    spearman=float(spearmanr(v, truth).statistic))
            for k, v in est.items()}


def _mean(reps):
    return {m: dict(mae=float(np.mean([r[m]["mae"] for r in reps])),
                    spearman=float(np.mean([r[m]["spearman"] for r in reps])))
            for m in METHODS}


def _row(label, extra, stats):
    line = f"  {label:>10} {extra:>22} "
    for m in METHODS:
        line += f"  {stats[m]['mae']:.4f}/{stats[m]['spearman']:+.3f}".rjust(22)
    return line


def _header(c1, c2):
    return (f"  {c1:>10} {c2:>22} " + " ".join(f"{m:>21}" for m in METHODS))


def load(name):
    z = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def scenario_small_n(pools, sizes=(50, 100, 200, 500, 1000), reps=5):
    """Subsample target ITEMS. Everything else is held fixed."""
    out = {}
    for name in pools:
        p = load(name)
        K, M, N = int(np.asarray(p["n_classes"])), p["pred"].shape[0], len(p["gold"])
        params = M * K * (K - 1)
        print(f"\n  {name}  (K={K}, M={M}, full N={N}; full-DS params = {params})")
        print(_header("N", "obs per DS-full param"))
        rows = {}
        for n in [s for s in sizes if s < N] + [N]:
            reps_n = []
            for r in range(reps if n < N else 1):
                idx = np.random.default_rng(r).choice(N, n, replace=False)
                reps_n.append(evaluate(p["pred"][:, idx], p["gold"][idx], p["group"],
                                       p["prior"], K, p["verifier_guess"][idx]))
            rows[n] = _mean(reps_n)
            print(_row(n, f"{n * M / params:.2f}", rows[n]))
        out[name] = dict(K=K, M=M, ds_full_params=params, rows=rows)
    return out


def scenario_large_k(ks=(5, 10, 20, 50, 100), N=400, M=9, G=3, reps=3,
                     acc_lo=0.20, acc_hi=0.50, collude=0.6):
    """SIMULATED. N held fixed; only the number of classes changes."""
    print(f"\n  simulated pool: N={N} (fixed), M={M}, {G} provenance groups, "
          f"accuracy {acc_lo}-{acc_hi}, within-group collusion {collude}")
    print(_header("K", "obs per DS-full param"))
    out = {}
    for K in ks:
        reps_k = []
        for r in range(reps):
            rng = np.random.default_rng(r)
            gold = rng.integers(0, K, N)
            acc = np.linspace(acc_lo, acc_hi, M)
            group = np.repeat(np.arange(G), M // G)
            fav = {g: (np.arange(K) + 1 + g) % K for g in range(G)}   # shared confusion
            pred = np.empty((M, N), dtype=np.int64)
            for m in range(M):
                ok = rng.random(N) < acc[m]
                coll = rng.random(N) < collude
                pred[m] = np.where(ok, gold,
                                   np.where(coll, fav[group[m]][gold], rng.integers(0, K, N)))
            prior = np.clip((pred == gold[None, :]).mean(1) + rng.normal(0, .03, M), .02, .98)
            reps_k.append(evaluate(pred, gold, group, prior, K))
        out[K] = _mean(reps_k)
        print(_row(K, f"{N * M / (M * K * (K - 1)):.2f}", out[K]))
    return out


def scenario_imbalance(pool_name, powers=(0.0, 1.5, 3.0, 5.0), n_fixed=800, reps=3):
    """Resample target items to a power-law class distribution, N HELD FIXED."""
    p = load(pool_name)
    K = int(np.asarray(p["n_classes"]))
    by = [np.where(p["gold"] == c)[0] for c in range(K)]
    print(f"\n  {pool_name}  (K={K}, N held at {n_fixed})")
    print(_header("imbalance", "rarest class n"))
    out = {}
    for power in powers:
        reps_p, cnt = [], None
        for r in range(reps):
            rng = np.random.default_rng(r)
            w = (np.arange(K) + 1.0) ** (-power); w /= w.sum()
            take = np.maximum((w * n_fixed).astype(int), 2)
            idx = np.concatenate([rng.choice(b, min(n, len(b)), replace=False)
                                  for b, n in zip(by, take)])
            cnt = np.bincount(p["gold"][idx], minlength=K)
            reps_p.append(evaluate(p["pred"][:, idx], p["gold"][idx], p["group"],
                                   p["prior"], K, p["verifier_guess"][idx]))
        ratio = float(cnt.max() / max(cnt.min(), 1))
        out[f"{ratio:.1f}x"] = dict(rarest=int(cnt.min()), **_mean(reps_p))
        print(_row(f"{ratio:.1f}x", int(cnt.min()), _mean(reps_p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/regime_scenarios.json")
    a = ap.parse_args()
    res = {}

    print("=" * 96)
    print("SCENARIO 1 -- SMALL unlabelled target set (real pools, items subsampled)")
    print("=" * 96)
    res["small_n"] = scenario_small_n(["graph_AC", "vision_mnist_usps"])

    print("\n" + "=" * 96)
    print("SCENARIO 2 -- LARGE number of classes (SIMULATED; N fixed)")
    print("=" * 96)
    res["large_k"] = scenario_large_k()

    print("\n" + "=" * 96)
    print("SCENARIO 3 -- IMBALANCED classes (real pool, N fixed)")
    print("=" * 96)
    res["imbalance"] = {n: scenario_imbalance(n) for n in ("graph_AC", "vision_mnist_usps")}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
