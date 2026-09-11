"""Budget-only matched-weight ablation using cached real-data outcomes.

Run with .venv/bin/python experiments/run_bound_threshold.py.
The fitted smoothness proxy is diagnostic, not a certified Lipschitz bound.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_ess_coverage import (
    CACHE, DELTA, TARGETS, build_cache, geometry, load_run,
    sigma_from_strength, subset_blocks, with_prior,
)
from pooleval import Config, PoolEval, theory as th
from synsql.config import RESULTS, TOP_K


def downward_crossings(bounds):
    """Adjacent budgets crossing from strictly above one to at most one."""
    return [i for i in range(1, len(bounds))
            if bounds[i - 1] > 1 and bounds[i] <= 1]


def run():
    if not Path(CACHE).exists():
        raise FileNotFoundError("Cached geometry required. No data rebuild is allowed.")
    cache = build_cache()
    Y, subsets, sub_of = cache["Y"], cache["subsets"], cache["sub_of"]
    L = th.estimate_lipschitz(cache["gram_cal"], Y, k=8, quantile=.95)
    out = dict(delta=DELTA, L_proxy=L.tolist(), selection="greedy coverage",
               weights="target matched", strength="fixed at K=5 calibration count",
               accuracy_reference="observed finite-target execution accuracy",
               bound_status="diagnostic only, smoothness and outcome independence unverified",
               legacy_bounds_reference="expected target accuracy, not the MAE reference",
               targets={})
    for ds in TARGETS:
        S, B, _ = subset_blocks(cache[f"cos_{ds}"], sub_of, subsets)
        order = th.greedy_cover(B, min(10, len(subsets)))[0]
        fixed_strength = int(np.isin(sub_of, subsets[order[:TOP_K]]).sum())
        run_obj = load_run(ds)
        truth = cache[f"acc_{ds}"]
        np.testing.assert_allclose(truth, cache[f"corr_{ds}"].mean(axis=1))
        simultaneous_delta = DELTA / (Y.shape[0] * len(order))
        rows = []
        for k in range(1, len(order) + 1):
            cols = np.flatnonzero(np.isin(sub_of, subsets[order[:k]]))
            g = geometry(S, cols)
            prior = th.weighted_estimate(Y[:, cols], g["w"])
            pe = PoolEval(Config(real_data=True)).evaluate(
                with_prior(run_obj, prior, sigma_from_strength(prior, fixed_strength)))
            transfer = 2 * L * np.sqrt(1 - g["c"])
            sampling = float(np.sqrt(np.log(2 / DELTA) / (2 * g["n_eff"])))
            bounds = transfer + sampling
            realized = np.array([th.bound_realized(lj, g["c"], g["n_eff"], g["N"], DELTA)
                                 for lj in L])
            simultaneous = np.array([
                th.bound_realized(lj, g["c"], g["n_eff"], g["N"], simultaneous_delta)
                for lj in L])
            prior_error = np.abs(prior - truth) * 100
            pe_error = np.abs(np.asarray(pe["acc"]) - truth) * 100
            rows.append(dict(K=k, n=g["n"], c=g["c"], n_eff=g["n_eff"],
                             fixed_strength=fixed_strength,
                             transfer=transfer.tolist(), sampling=sampling,
                             bounds=bounds.tolist(), bound_max=float(bounds.max()),
                             bound_min=float(bounds.min()),
                             realized_bounds=realized.tolist(),
                             realized_bound_max=float(realized.max()),
                             realized_simultaneous_bounds=simultaneous.tolist(),
                             realized_simultaneous_bound_max=float(simultaneous.max()),
                             required_neff_max=float(th.required_neff(max(L), g["c"], DELTA)),
                             prior_errors=prior_error.tolist(), pe_errors=pe_error.tolist(),
                             prior_mae=float(prior_error.mean()), pe_mae=float(pe_error.mean())))
        crossings = downward_crossings([r["bound_max"] for r in rows])
        model_crossings = []
        for j in range(Y.shape[0]):
            for i in downward_crossings([r["bounds"][j] for r in rows]):
                model_crossings.append(dict(model=j, K_before=rows[i - 1]["K"],
                                            K_after=rows[i]["K"],
                                            prior_error_change=rows[i]["prior_errors"][j] - rows[i - 1]["prior_errors"][j],
                                            pe_error_change=rows[i]["pe_errors"][j] - rows[i - 1]["pe_errors"][j]))
        out["targets"][ds] = dict(rows=rows, crossing_indices=crossings,
                                  realized_crossing_indices=downward_crossings(
                                      [r["realized_bound_max"] for r in rows]),
                                  simultaneous_scope="all models and tested budgets within this target",
                                  simultaneous_delta_per_comparison=simultaneous_delta,
                                  model_crossings=model_crossings)
        print(ds, "expected-reference diagnostic crossings:",
              [(rows[i - 1]["K"], rows[i]["K"]) for i in crossings],
              "realized-reference diagnostic crossing indices:",
              out["targets"][ds]["realized_crossing_indices"], flush=True)
    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    path = Path(RESULTS) / "bound_threshold.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print("Saved", path)
    return out


if __name__ == "__main__":
    run()
