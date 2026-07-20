"""NE (new experiments) -- misspecification robustness + small-label oracle baseline.

These experiments are NOT in the paper. They were added during an independent
review to probe two weaknesses of the original evaluation:

  1. CIRCULARITY. The paper's simulator (`pooleval/data/simulator.py`) draws data
     from *exactly* the generative model PoolEval-SQL assumes (IRT ability/difficulty
     + provenance-group shared errors + an independent verifier + a Gaussian seen
     prior). An estimator whose model matches the data-generating process is
     guaranteed to win, so the reported margins over the baselines are partly an
     artifact of the match. Here we deliberately BREAK each assumption and re-measure.

  2. A MISSING BASELINE. The paper never compares against the obvious practical
     alternative: label a handful of items by hand and estimate accuracy directly.
     We add that oracle-sampling baseline (B6) and report the "label-equivalent
     budget" -- how many gold labels a practitioner would need for a ranking as good
     as PoolEval-SQL's label-free estimate.

Everything reuses the shipped estimator/baselines unchanged; only the DGP and the
extra baseline are new. Run:

    python experiments/run_ne_robustness.py [--seeds 20]

Writes results/ne_misspec.json and results/ne_label_budget.json.
"""
import argparse
import copy
import numpy as np

from _shared import (Config, PoolEval, simulate, metrics, all_baselines,
                     aggregate, mean_ci, save_json, METRIC_KEYS)
from pooleval.data.simulator import (PoolRun, sigmoid, default_groups)


# --------------------------------------------------------------------------- #
#  Alternative data-generating processes (DGPs) that BREAK the estimator model #
# --------------------------------------------------------------------------- #
def _finalize(cfg, true_class, group, G, rng, verifier_true_class=None,
              verifier_correlated=False):
    """Shared tail of every DGP: build priors + verifier + PoolRun, matching the
    fields the estimator reads. `verifier_correlated` makes the verifier fail on
    the SAME items the pool tends to fail on (a realistic execution checker that
    cannot catch a query that runs but returns the wrong rows)."""
    M, N = true_class.shape
    true_acc = (true_class == 0).mean(axis=1)
    b = rng.normal(0, cfg.diff_std, size=N)

    bias = rng.normal(0, cfg.prior_bias, size=M)
    prior = np.clip(true_acc + bias + rng.normal(0, cfg.prior_noise, size=M),
                    0.01, 0.99)
    prior_sigma = np.full(M, max(cfg.prior_noise, 1e-3))

    if verifier_correlated:
        # verifier is right only where the pool is mostly right (its errors are
        # correlated with pool errors, not independent) -> anchor loses its
        # collusion-resistance, the property the paper leans on.
        pool_right = (true_class == 0).mean(axis=0)          # [N] fraction correct
        p_v = np.clip(0.35 + 0.6 * pool_right, 0.0, 0.98)
        vc = rng.random(N) < p_v
    else:
        vc = rng.random(N) < cfg.verifier_acc
    verifier_guess = np.where(vc, 0, rng.integers(1, 1000, size=N))
    vc = verifier_guess == 0
    phi = rng.uniform(0.2, 1.0, size=N)
    return PoolRun(true_class, true_acc, group, prior, prior_sigma, b, phi,
                   verifier_guess, vc, M, N, G)


def dgp_baseline(cfg, rng):
    """Faithful re-implementation of the shipped simulator (the matched DGP)."""
    return simulate(cfg, rng=rng)


def dgp_non_irt(cfg, rng):
    """MISSPEC (a): no shared item-difficulty axis. Each (model,item) is an
    independent Bernoulli at the model's own accuracy -- there is no latent b_i the
    estimator can recover, so the IRT difficulty term is fitting noise."""
    M, N, G = cfg.M, cfg.N, cfg.n_groups
    group = default_groups(M, G)
    acc = rng.uniform(cfg.acc_lo, cfg.acc_hi, size=M)
    true_class = np.zeros((M, N), dtype=np.int64)
    idio = 100000
    for m in range(M):
        correct = rng.random(N) < acc[m]                     # NO difficulty coupling
        for i in range(N):
            if not correct[i]:
                r = rng.random()
                if r < cfg.collusion:
                    true_class[m, i] = 1000 + group[m] * N + i   # group-shared wrong
                else:
                    true_class[m, i] = idio; idio += 1
    return _finalize(cfg, true_class, group, G, rng)


def dgp_cross_group_collusion(cfg, rng):
    """MISSPEC (b): errors are shared ACROSS provenance groups, not within. The
    operator's group tags are then anti-informative -- the correlation the model
    discounts (within-group) is not where the real correlation lives."""
    M, N, G = cfg.M, cfg.N, cfg.n_groups
    group = default_groups(M, G)
    acc = rng.uniform(cfg.acc_lo, cfg.acc_hi, size=M)
    a_logit = np.log(acc / (1 - acc))
    b = rng.normal(0, cfg.diff_std, size=N)
    true_class = np.zeros((M, N), dtype=np.int64)
    idio = 100000
    # a global shared-wrong answer per item, emitted regardless of group
    for m in range(M):
        p = sigmoid(a_logit[m] - b)
        correct = rng.random(N) < p
        for i in range(N):
            if not correct[i]:
                r = rng.random()
                if r < cfg.collusion:
                    true_class[m, i] = 700000 + i             # CROSS-group shared
                else:
                    true_class[m, i] = idio; idio += 1
    return _finalize(cfg, true_class, group, G, rng)


def dgp_correlated_verifier(cfg, rng):
    """MISSPEC (c): the verifier's errors are correlated with the pool's errors
    (idealized-independent-oracle assumption broken). DGP is otherwise the matched
    simulator, isolating the effect of the verifier assumption."""
    base = simulate(cfg, rng=rng)
    # rebuild only the verifier so it fails where the pool fails
    pool_right = (base.true_class == 0).mean(axis=0)
    p_v = np.clip(0.35 + 0.6 * pool_right, 0.0, 0.98)
    vc = rng.random(base.N) < p_v
    vg = np.where(vc, 0, rng.integers(1, 1000, size=base.N))
    return PoolRun(base.true_class, base.true_acc, base.group, base.prior,
                   base.prior_sigma, base.b, base.phi, vg, vg == 0,
                   base.M, base.N, base.n_groups)


def dgp_heavy_gauge(cfg, rng):
    """MISSPEC (d): many more 'everyone agrees but is wrong' items (universal_error
    0.30 -> 0.60). Stresses whether the anchor still breaks the gauge when consensus
    is frequently a shared delusion."""
    c = copy.copy(cfg)
    c.universal_error = 0.60
    return simulate(c, rng=rng)


DGPS = {
    "matched (paper sim)":      dgp_baseline,
    "misspec-non-IRT":          dgp_non_irt,
    "misspec-cross-group":      dgp_cross_group_collusion,
    "misspec-corr-verifier":    dgp_correlated_verifier,
    "misspec-heavy-gauge":      dgp_heavy_gauge,
}


# --------------------------------------------------------------------------- #
#  B6 -- small-label oracle sampling (the missing practical baseline)          #
# --------------------------------------------------------------------------- #
def oracle_label_baseline(run, k, rng):
    """Label k random items by hand; estimate each model's accuracy as its
    fraction correct on those k, and rank by it. This is what a practitioner does
    without any label-free machinery."""
    N = run.N
    k = min(k, N)
    idx = rng.choice(N, size=k, replace=False)
    correct = (run.true_class[:, idx] == 0)                  # [M,k]
    return correct.mean(axis=1)


# --------------------------------------------------------------------------- #
#  Experiment drivers                                                          #
# --------------------------------------------------------------------------- #
def run_misspec(seeds):
    """For each DGP, compare PoolEval-SQL against the shipped baselines."""
    out = {}
    for dgp_name, dgp in DGPS.items():
        rows = {}
        for s in range(seeds):
            cfg = Config(seed=s)
            rng = np.random.default_rng(1000 + s)
            run = dgp(cfg, rng)
            for b in all_baselines():
                m = metrics.all_metrics(b.evaluate(run, cfg), run.true_acc)
                rows.setdefault(b.name, []).append(m)
            res = PoolEval(cfg).evaluate(run)
            rows.setdefault("PoolEval-SQL (ours)", []).append(
                metrics.all_metrics(res["acc"], run.true_acc))
        out[dgp_name] = {k: aggregate(v) for k, v in rows.items()}
    return out


def run_label_budget(seeds, ks=(5, 10, 20, 40, 80, 160)):
    """Sweep the labeling budget k for B6 and compare to PoolEval-SQL (label-free)."""
    pool_rows, b6_rows = [], {k: [] for k in ks}
    for s in range(seeds):
        cfg = Config(seed=s)
        run = simulate(cfg)
        rng = np.random.default_rng(5000 + s)
        res = PoolEval(cfg).evaluate(run)
        pool_rows.append(metrics.all_metrics(res["acc"], run.true_acc))
        for k in ks:
            # average over 5 random label subsets to reduce sampling variance
            subs = [oracle_label_baseline(run, k, np.random.default_rng(7000 + s * 97 + r))
                    for r in range(5)]
            m = [metrics.all_metrics(a, run.true_acc) for a in subs]
            b6_rows[k].append({mk: float(np.mean([mm[mk] for mm in m]))
                               for mk in METRIC_KEYS})
    pool_agg = aggregate(pool_rows)
    b6_agg = {k: aggregate(v) for k, v in b6_rows.items()}
    # label-equivalent budget: smallest k whose mean Flip <= PoolEval's mean Flip
    pool_flip = pool_agg["Flip"][0]
    equiv_k = next((k for k in ks if b6_agg[k]["Flip"][0] <= pool_flip), None)
    return {"PoolEval-SQL (ours)": {mk: pool_agg[mk] for mk in METRIC_KEYS},
            "B6 oracle-label": {str(k): {mk: b6_agg[k][mk] for mk in METRIC_KEYS}
                                for k in ks},
            "pool_flip": pool_flip,
            "label_equivalent_k": equiv_k,
            "N": Config().N}


def _print_misspec(out):
    keys = ["MAE", "Flip", "Kendall", "Top1"]
    for dgp_name, agg in out.items():
        print(f"\n=== DGP: {dgp_name} ===")
        print(f"{'Method':24s}" + "".join(f"{k:>10s}" for k in keys))
        order = ["B1 Independent", "B2 Majority/self-cons.", "B3 Dawid--Skene",
                 "B4 Agreement-on-line", "B5 LLM-as-judge", "PoolEval-SQL (ours)"]
        for name in order:
            if name not in agg:
                continue
            cells = "".join(f"{agg[name][k][0]:>10.2f}" for k in keys)
            print(f"{name:24s}{cells}")


def _print_budget(out):
    print("\n=== Label-equivalent budget (B6 oracle sampling vs PoolEval-SQL) ===")
    p = out["PoolEval-SQL (ours)"]
    print(f"PoolEval-SQL (label-free): MAE={p['MAE'][0]:.2f}  Flip={p['Flip'][0]:.3f}  "
          f"Kendall={p['Kendall'][0]:.2f}")
    print(f"{'k labels':>10s}{'MAE':>10s}{'Flip':>10s}{'Kendall':>10s}")
    for k, row in out["B6 oracle-label"].items():
        print(f"{k:>10s}{row['MAE'][0]:>10.2f}{row['Flip'][0]:>10.3f}"
              f"{row['Kendall'][0]:>10.2f}")
    ek = out["label_equivalent_k"]
    print(f"\n-> PoolEval-SQL's label-free ranking (Flip={out['pool_flip']:.3f}) "
          f"matches ~{ek} hand labels out of N={out['N']}"
          if ek else "\n-> No tested budget matched PoolEval-SQL's flip rate.")


def main(seeds=20):
    print(f"[NE] misspecification robustness ({seeds} seeds per DGP)")
    misspec = run_misspec(seeds)
    _print_misspec(misspec)
    save_json("ne_misspec.json", misspec)

    print(f"\n[NE] label-equivalent budget ({seeds} seeds)")
    budget = run_label_budget(seeds)
    _print_budget(budget)
    save_json("ne_label_budget.json", budget)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    main(**vars(ap.parse_args()))
