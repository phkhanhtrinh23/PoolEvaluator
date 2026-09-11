"""Known-smoothness Monte Carlo check on real geometry, synthetic outcomes only."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_ess_coverage import CACHE, DELTA, TARGETS, build_cache, geometry, subset_blocks
from pooleval import theory as th
from synsql.config import RESULTS


def smooth_probabilities(cosines, a):
    """p(v) = 1/2 + a <u, phi(v)> has Lipschitz constant at most a."""
    if not 0 <= a <= .5:
        raise ValueError("a must be in [0, .5]")
    cosines = np.asarray(cosines)
    if np.any(np.abs(cosines) > 1 + 1e-10):
        raise ValueError("Expected unit-vector inner products")
    return .5 + a * np.clip(cosines, -1, 1)


def violation_summary(events):
    """One-sided 95% exact binomial upper limit over independent repetitions."""
    events = np.asarray(events, dtype=bool)
    n, failures = len(events), int(events.sum())
    upper = 1.0 if failures == n else float(beta.ppf(.95, failures + 1, n - failures))
    return dict(count=failures, rate=failures / n, upper_95=upper)


def mae_summary(errors):
    # Models share a repetition. Average within repetitions before computing SE.
    replicate_mae = errors.mean(axis=1) * 100
    return dict(mae=float(replicate_mae.mean()),
                mc_se=float(replicate_mae.std(ddof=1) / np.sqrt(len(errors))))


def run(repetitions=5000, seed=20260909):
    if not Path(CACHE).exists():
        raise FileNotFoundError("Cached geometry required")
    cache = build_cache()
    gram = cache["gram_cal"]
    np.testing.assert_allclose(np.diag(gram), 1, atol=1e-10)
    models = 10
    anchors = np.linspace(0, gram.shape[0] - 1, models, dtype=int)
    rng = np.random.default_rng(seed)
    out = dict(seed=seed, repetitions=repetitions, delta=DELTA,
               synthetic_models=models, anchors=anchors.tolist(),
               outcome_source="synthetic independent Bernoulli, not actual model correctness",
               probability_model="p_j(v)=1/2+a*cos(phi(v),phi(calibration_anchor_j))",
               L="a is a valid deterministic upper bound by Cauchy-Schwarz",
               targets={})
    for ds in TARGETS:
        cos = cache[f"cos_{ds}"]
        np.testing.assert_allclose(np.diag(cache[f"self_{ds}"]), 1, atol=1e-10)
        S, B, _ = subset_blocks(cos, cache["sub_of"], cache["subsets"])
        order = th.greedy_cover(B, min(10, len(cache["subsets"])))[0]
        geometries, weights = [], np.zeros((gram.shape[0], len(order)))
        for i in range(len(order)):
            cols = np.flatnonzero(np.isin(cache["sub_of"], cache["subsets"][order[:i + 1]]))
            g = geometry(S, cols)
            geometries.append(g)
            weights[cols, i] = g["w"]
        cases = {}
        for a in (0., .25, .5):
            pcal = smooth_probabilities(gram[anchors], a)
            ptgt = smooth_probabilities(cos[:, anchors].T, a)
            theta = ptgt.mean(axis=1)
            mean_estimate = pcal @ weights
            transfer = np.array([th.transfer_term(a, g["c"]) for g in geometries])
            assert np.all(np.abs(mean_estimate - theta[:, None]) <= transfer + 1e-12)
            errors, realized_errors = [], []
            for start in range(0, repetitions, 100):
                size = min(100, repetitions - start)
                ycal = rng.random((size, *pcal.shape)) < pcal
                ytgt = rng.random((size, *ptgt.shape)) < ptgt
                estimates = ycal.astype(float) @ weights
                errors.append(np.abs(estimates - theta[None, :, None]))
                realized_errors.append(np.abs(estimates - ytgt.mean(axis=2)[:, :, None]))
            errors = np.concatenate(errors)
            realized_errors = np.concatenate(realized_errors)
            rows = []
            for i, g in enumerate(geometries):
                point = th.bound_weighted(a, g["c"], g["n_eff"], DELTA)
                simultaneous = th.bound_weighted(a, g["c"], g["n_eff"], DELTA / models)
                observed = th.bound_realized(a, g["c"], g["n_eff"], g["N"], DELTA / models)
                rows.append(dict(K=i + 1, c=g["c"], n_eff=g["n_eff"],
                                 expected_pointwise_bound=point,
                                 expected_pool_bound=simultaneous,
                                 realized_pool_bound=observed,
                                 transfer_bias_max=float(np.abs(mean_estimate[:, i] - theta).max()),
                                 expected=mae_summary(errors[:, :, i]),
                                 realized=mae_summary(realized_errors[:, :, i]),
                                 expected_pool_violations=violation_summary(
                                     (errors[:, :, i] > simultaneous).any(axis=1)),
                                 realized_pool_violations=violation_summary(
                                     (realized_errors[:, :, i] > observed).any(axis=1)),
                                 pointwise_max_model_violation_rate=float(
                                     (errors[:, :, i] > point).mean(axis=0).max())))
            cases[str(a)] = rows
        out["targets"][ds] = cases
        print("[done] controlled", ds, flush=True)
    path = Path(RESULTS) / "bound_control.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print("Saved", path)
    return out


if __name__ == "__main__":
    run()
