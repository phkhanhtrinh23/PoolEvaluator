"""Does choosing ESS by the Section 19 threshold actually make B <= eps?

Fully controlled: we build the geometry so coverage c (hence the transfer term
A = 2L sqrt(1-c)) is fixed and uniform, while the effective sample size n_eff = n is
dialed directly by the number of equally weighted calibration anchors. Expected
correctness is p_j(v) = 1/2 + a <w_j, phi(v)> with ||w_j|| = 1, so L = a is a provable
Lipschitz constant (Cauchy-Schwarz). Outcomes are independent Bernoulli draws.

For a target tolerance eps, Section 19.13 says B <= eps iff A < eps and
    n_eff >= n* = log(2/delta) / (2 (eps - A)^2).
We compute n* from the provable A, then sweep n_eff across n* and check two things:
  (1) sharpness  -- the bound B(n) crosses eps exactly at n*, and
  (2) validity   -- the empirical violation rate stays <= delta at every n.
An infeasible regime (A >= eps, so n* = inf) confirms that no ESS rescues the bound.

Run with .venv/bin/python experiments/run_ess_threshold.py.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pooleval import theory as th  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DELTA = 0.05


def geometry(n_groups, per_group, s, d_extra=6, seed=0):
    """n_groups orthonormal calibration anchors; per_group targets each at cosine s to
    exactly one anchor (and 0 to the others). Coverage is uniform at (1+s)/2 and the
    matched weights are 1/n_groups each, so n_eff = n_groups exactly."""
    rng = np.random.default_rng(seed)
    n, d = n_groups, n_groups + d_extra
    anchors = np.eye(d)[:n]                       # e_1..e_n, unit and orthonormal
    tail = rng.standard_normal((n, d))
    tail[:, :n] = 0.0                             # live only in the extra dims
    tail /= np.linalg.norm(tail, axis=1, keepdims=True)
    targets = np.repeat(s * anchors + np.sqrt(1 - s ** 2) * tail, per_group, axis=0)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    S = th.shifted_cosine(targets, anchors)       # [N, n]
    assign, _ = th.best_match(S)
    _, c = th.coverage(S)
    w = th.match_weights(assign, n)
    # verify the construction did what we claimed
    assert np.all(assign == np.repeat(np.arange(n), per_group)), "best match not own anchor"
    np.testing.assert_allclose(c, (1 + s) / 2, atol=1e-9)
    np.testing.assert_allclose(th.n_eff(w), n, atol=1e-9)
    return anchors, targets, w, float(c)


def violation_rate(events):
    events = np.asarray(events, dtype=bool)
    n, k = len(events), int(events.sum())
    upper = 1.0 if k == n else float(beta.ppf(0.95, k + 1, n - k))
    return dict(rate=k / n, upper_95=upper)


def sweep(a, s, eps, ns, per_group=40, models=8, reps=4000, seed=20260910):
    rng = np.random.default_rng(seed)
    c = (1 + s) / 2
    A = th.transfer_term(a, c)                    # provable transfer term, fixed by (a, s)
    need = th.required_neff(a, c, DELTA, eps)     # Section 19.13 threshold n*
    rows = []
    for n in ns:
        anchors, targets, w, c_chk = geometry(n, per_group, s, seed=seed + n)
        W = rng.standard_normal((models, anchors.shape[1]))
        W /= np.linalg.norm(W, axis=1, keepdims=True)   # ||w_j|| = 1  =>  L = a
        pcal = np.clip(0.5 + a * (anchors @ W.T).T, 0, 1)   # [M, n]
        ptgt = np.clip(0.5 + a * (targets @ W.T).T, 0, 1)   # [M, N]
        theta = ptgt.mean(axis=1)                            # known expected target acc
        tilde = pcal @ w
        assert np.all(np.abs(tilde - theta) <= A + 1e-12), "transfer term violated"
        B = th.bound_weighted(a, c, n, DELTA)                # scalar: c, n_eff uniform
        errs = []
        for start in range(0, reps, 200):
            size = min(200, reps - start)
            ycal = (rng.random((size, models, n)) < pcal).astype(float)
            est = ycal @ w
            errs.append(np.abs(est - theta[None, :]))
        errs = np.concatenate(errs)                          # [reps, models]
        mae = float(errs.mean())
        rows.append(dict(
            n_eff=n, c=c_chk, A=A, bound=B,
            predicted_ok=bool(n >= need), empirical_bound_le_eps=bool(B <= eps),
            mae_pts=mae * 100,
            transfer_bias_max_pts=float(np.abs(tilde - theta).max() * 100),
            bound_over_mae=float(B / mae) if mae > 0 else float("inf"),
            pool_violation=violation_rate((errs > B).any(axis=1)),
            worst_model_violation=float((errs > B).mean(axis=0).max())))
    return dict(a=a, L=a, s=s, coverage=c, eps=eps, transfer_A=A,
                required_neff=(None if not np.isfinite(need) else float(need)),
                feasible=bool(A < eps), rows=rows)


def run():
    out = dict(delta=DELTA,
               probability_model="p_j(v)=1/2+a<w_j,phi(v)>, ||w_j||=1 => L=a (provable)",
               outcome_source="independent Bernoulli draws",
               threshold="Section 19.13: B<=eps iff A<eps and n_eff>=log(2/delta)/(2(eps-A)^2)",
               regimes={})
    # General eps < 1. A = 0.2 < eps = 0.5, so n* is finite; sweep across it.
    out["regimes"]["feasible_eps0.5"] = sweep(
        a=0.2, s=0.5, eps=0.5, ns=[10, 15, 18, 20, 21, 22, 25, 30, 50])
    # B in [0,1], A < 1: the A<1-then-ESS condition of Section 19.8. Here A = 0.5 and
    # n* = log(40)/(2(1-0.5)^2) = 7.38, so B<=1 must first hold at n_eff = 8. Sweep the
    # integer boundary to check the crossing is exactly at ceil(n*).
    out["regimes"]["B01_A0.5_sharp"] = sweep(
        a=0.5, s=0.5, eps=1.0, ns=[5, 6, 7, 8, 9, 12])
    # B in [0,1], A >= 1: the necessary condition A<1 FAILS (2L sqrt(1-c) = 1.14), so
    # n* = inf and no effective sample size can bring B into [0,1].
    out["regimes"]["B01_Aover1_infeasible"] = sweep(
        a=0.9, s=0.2, eps=1.0, ns=[10, 40, 160, 640])
    # General eps < 1, infeasible: A = 0.656 >= eps = 0.5, so n* = inf; no ESS helps.
    out["regimes"]["infeasible_eps0.5"] = sweep(
        a=0.5, s=0.14, eps=0.5, ns=[10, 20, 40, 80, 160, 320])
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "ess_threshold.json")
    with open(path, "w") as handle:
        handle.write(json.dumps(out, indent=2, allow_nan=False) + "\n")
    # console summary
    for name, reg in out["regimes"].items():
        need = reg["required_neff"]
        print(f"\n=== {name}: L={reg['L']} c={reg['coverage']:.3f} "
              f"A={reg['transfer_A']:.3f} eps={reg['eps']} "
              f"n*={'inf' if need is None else f'{need:.1f}'} feasible={reg['feasible']}")
        print(f"  {'n_eff':>6} {'bound':>7} {'B<=eps':>7} {'n>=n*':>6} "
              f"{'MAE_pts':>8} {'poolViol':>9} {'upper95':>8}")
        for r in reg["rows"]:
            print(f"  {r['n_eff']:6d} {r['bound']:7.3f} "
                  f"{str(r['empirical_bound_le_eps']):>7} {str(r['predicted_ok']):>6} "
                  f"{r['mae_pts']:8.2f} {r['pool_violation']['rate']:9.4f} "
                  f"{r['pool_violation']['upper_95']:8.4f}")
    print("\nSaved", path)
    return out


if __name__ == "__main__":
    run()
