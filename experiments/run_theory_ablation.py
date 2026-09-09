"""Head-to-head: PoolEval WITHOUT the new theory vs WITH it, one switch at a time.

`run_ess_coverage.py` measures the theory's own quantities (coverage, n_eff, bound
slack, smoothness). This script answers the only question that decides whether to
adopt any of it: **does the final accuracy error go down?**

Four independent switches, all defined by the two derivations:

  selection  distance top-k  ->  greedy coverage       (submodular selection, sec. 6)
  weights    uniform average ->  target-matched        (sec. 11 / 16)
  strength   s = n0 (a0 = 1) ->  s = c * n0            (power-prior discount, MTM08)
  s_beta     a0 * J * n0     ->  a0 * n0               (drop the nominal J, sec. 5)

BASELINE is what the paper does today: rank candidate subsets by centroid distance,
take the top k, average their labeled items uniformly, and set the anchor sigma from
the plain binomial SE over all n0 of them (which is exactly s = n0, i.e. a0 = 1).

Everything is scored against the REAL per-model accuracy on the REAL target, with the
real SynSQL probes as the calibration corpus. Metrics are MAE in accuracy points.

    python experiments/run_theory_ablation.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import Config, PoolEval, metrics, theory as th          # noqa: E402
from pooleval.new_formulation import collision_agreement_em           # noqa: E402
from synsql.config import RESULTS, TOP_K                              # noqa: E402
from experiments.run_ess_coverage import (                            # noqa: E402
    TARGETS, build_cache, load_run, with_prior, sigma_from_strength,
    subset_blocks, geometry, topk_by_distance)

# name -> (selection, weights, strength, beta_scale)
CONFIGS = [
    ("Baseline (no new theory)",      "distance", "uniform", "full",     "pair"),
    ("+ greedy coverage selection",   "greedy",   "uniform", "full",     "pair"),
    ("+ target-matched weights",      "distance", "matched", "full",     "pair"),
    ("+ coverage discount a0 = c",    "distance", "uniform", "coverage", "pair"),
    ("+ item-scaled s_beta (drop J)", "distance", "uniform", "full",     "item"),
    ("All new theory",                "greedy",   "matched", "coverage", "item"),
    # reverse ablation: which switch is the combined gain actually resting on?
    ("All new theory - greedy",       "distance", "matched", "coverage", "item"),
    ("All new theory - matched",      "greedy",   "uniform", "coverage", "item"),
    ("All new theory - discount",     "greedy",   "matched", "full",     "item"),
    ("All new theory - item s_beta",  "greedy",   "matched", "coverage", "pair"),
]
FULL = "All new theory"
BASE = "Baseline (no new theory)"


def evaluate(cfg_row, cos, Y, sub_of, subsets, S, B, run_obj, true_acc,
             C_agree, gamma_g):
    _, selection, weights, strength, beta_scale = cfg_row
    K = TOP_K
    pick = (th.greedy_cover(B, K)[0] if selection == "greedy"
            else topk_by_distance(cos, sub_of, subsets, K))
    cols = np.flatnonzero(np.isin(sub_of, subsets[pick]))
    g = geometry(S, cols)

    prior = (th.weighted_estimate(Y[:, cols], g["w"]) if weights == "matched"
             else th.uniform_estimate(Y[:, cols]))
    s = g["c"] * g["n"] if strength == "coverage" else float(g["n"])
    s_beta = s if beta_scale == "item" else s * run_obj.M

    pe = PoolEval(Config(real_data=True)).evaluate(
        with_prior(run_obj, prior, sigma_from_strength(prior, s)))
    mp = metrics.all_metrics(pe["acc"], true_acc)
    ce = collision_agreement_em(C_agree, run_obj.group, gamma_g, prior, s,
                                beta_init=0.7, beta_strength=s_beta)
    mc = metrics.all_metrics(ce["alpha"], true_acc)
    return dict(prior_mae=float(np.mean(np.abs(prior - true_acc)) * 100),
                pe_mae=float(mp["MAE"]), pe_kendall=float(mp["Kendall"]),
                pe_top1=float(mp["Top1"]), ce_mae=float(mc["MAE"]),
                ce_kendall=float(mc["Kendall"]),
                c=g["c"], n_eff=g["n_eff"], s=float(s), ess=th.prior_ess(s),
                map_weight=th.map_weight(s, run_obj.N))


def run(log=print):
    cache = build_cache(log=log)
    Y, sub_of, subsets = cache["Y"], cache["sub_of"], cache["subsets"]
    out = {"configs": [c[0] for c in CONFIGS], "targets": {}}

    for ds in TARGETS:
        if f"cos_{ds}" not in cache:
            continue
        cos, true_acc = cache[f"cos_{ds}"], cache[f"acc_{ds}"]
        run_obj = load_run(ds)
        S, B, _ = subset_blocks(cos, sub_of, subsets)
        pseudo = PoolEval(Config(real_data=True)).evaluate(run_obj)
        latent_hat = np.array([max(q, key=q.get) for q in pseudo["latent_post"]],
                              dtype=run_obj.true_class.dtype)
        C_agree = (run_obj.true_class == latent_hat[None, :]).astype(float)
        gamma_g = np.full(run_obj.n_groups, 0.15)
        out["targets"][ds] = {
            c[0]: evaluate(c, cos, Y, sub_of, subsets, S, B, run_obj, true_acc,
                           C_agree, gamma_g) for c in CONFIGS}
        log(f"[done] {ds}")

    # per-config mean over targets
    out["mean"] = {}
    for name, *_ in CONFIGS:
        rows = [out["targets"][d][name] for d in out["targets"]]
        out["mean"][name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, "theory_ablation.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    _print(out, log)
    log(f"\n[saved] {p}")
    return out


def _print(out, log):
    log("\n" + "=" * 96)
    log("PoolEval MAE (accuracy points, lower is better) -- per target")
    log("=" * 96)
    tg = list(out["targets"])
    log(f"{'configuration':32s}" + "".join(f"{d[:12]:>13s}" for d in tg) + f"{'MEAN':>8s}")
    for name, *_ in CONFIGS:
        row = [out["targets"][d][name]["pe_mae"] for d in tg]
        log(f"{name:32s}" + "".join(f"{v:>13.2f}" for v in row)
            + f"{out['mean'][name]['pe_mae']:>8.2f}")
    log("\n" + "=" * 96)
    log(f"{'configuration':32s}{'prior MAE':>11s}{'PoolEval MAE':>14s}"
        f"{'Kendall':>9s}{'Top1':>7s}{'collision MAE':>15s}")
    log("=" * 96)
    for name, *_ in CONFIGS:
        m = out["mean"][name]
        log(f"{name:32s}{m['prior_mae']:>11.2f}{m['pe_mae']:>14.2f}"
            f"{m['pe_kendall']:>9.2f}{m['pe_top1']:>7.2f}{m['ce_mae']:>15.2f}")
    base, full = out["mean"][BASE], out["mean"][FULL]
    log(f"\ndelta (all new theory - baseline):  prior MAE "
        f"{full['prior_mae']-base['prior_mae']:+.2f}   PoolEval MAE "
        f"{full['pe_mae']-base['pe_mae']:+.2f}   collision MAE "
        f"{full['ce_mae']-base['ce_mae']:+.2f}")

    # How much of the mean is one target? And how often does it actually win?
    log("\n" + "=" * 96)
    log("ROBUSTNESS -- the mean over 5 targets can be driven by a single one")
    log("=" * 96)
    log(f"{'configuration':32s}{'PE MAE':>9s}{'vs base':>9s}{'wins':>7s}"
        f"{'PE MAE excl. spider2local':>27s}")
    for name, *_ in CONFIGS:
        d = [out["targets"][t][name]["pe_mae"] - out["targets"][t][BASE]["pe_mae"]
             for t in tg]
        sub = [out["targets"][t][name]["pe_mae"] for t in tg if t != "spider2local"]
        log(f"{name:32s}{out['mean'][name]['pe_mae']:>9.2f}"
            f"{out['mean'][name]['pe_mae'] - base['pe_mae']:>+9.2f}"
            f"{sum(1 for x in d if x < -1e-9):>4d}/{len(d):<2d}"
            f"{float(np.mean(sub)):>27.2f}")


if __name__ == "__main__":
    run()
