"""How many labeled SUBSETS should we spend on the prior?

The prior (pi, e, gamma) is read off labeled data. Labeled data arrives in fixed-size
SUBSETS -- a batch of items the M models are run on and scored against gold. Using K
subsets instead of 1 gives a prior measured on K times more items, so it costs K times
more work and should be K times less noisy. This experiment measures both sides of that
trade so the operating point can be chosen instead of guessed.

  x  number of subsets K              (each of --subset-size items, disjoint)
  L  latency of computing the prior accuracy for each model, averaged over models
  R  MAE of the final accuracy estimate, averaged over models (mae() does that) and
     then over modalities

WHAT THE LATENCY INCLUDES. The models' outputs on the labeled split are already cached
here, so the timer covers the prior COMPUTATION -- scoring each model on each subset,
then (for the full-prior column) the MxM collision matrix e, the pseudo-labels and gamma.
It does NOT include running the M models on the subsets, which in a real deployment
dominates and also scales linearly in K. So these numbers are a lower bound on the true
cost, and the linear shape they show is the part the prior-building code contributes.

Subsets are disjoint, so the K-subset prior is exactly the accuracy over the K*size
pooled items -- averaging K equal-size disjoint subset accuracies is the pooled mean.

  python experiments/run_subset_count.py --draws 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval.validated_em import (OracleExpert, run_validation,      # noqa: E402
                                   validated_em)
from experiments.run_validated_em import _strength, build_stats, mae  # noqa: E402
from experiments.run_random_init import load_cases                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODALITY = {"text2sql": "text2sql", "vision": "vision", "graph": "graph"}


def draw_subsets(n_pool, k, size, rng):
    """K disjoint index blocks of `size` drawn without replacement from the pool."""
    idx = rng.choice(n_pool, size=k * size, replace=False)
    return idx.reshape(k, size)


def time_prior_accuracy(lab, subsets, reps):
    """Median wall-clock of computing each model's prior accuracy over the subsets,
    divided by M -> seconds PER MODEL. The loop is explicit (model by model, subset by
    subset) because that is the shape of the work a deployment does; a vectorised mean
    would time numpy's broadcasting, not the operation we are costing."""
    M = lab.shape[0]
    runs, acc = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        acc = np.empty((M, subsets.shape[0]))
        for j in range(M):
            for k, s in enumerate(subsets):
                acc[j, k] = float((lab[j, s] == 0).mean())
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs)) / M, acc.mean(axis=1)


def time_full_prior(lab, idx, group, pi, reps):
    """Median wall-clock of building e + pseudo-labels + gamma, per model."""
    M = lab.shape[0]
    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        build_stats(lab[:, idx], group, pi)
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs)) / M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-counts", nargs="+", type=int,
                    default=[1, 2, 3, 4, 6, 8, 12])
    ap.add_argument("--subset-size", type=int, default=10)
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 40])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--timing-reps", type=int, default=25)
    ap.add_argument("--n-target", type=int, default=600)
    ap.add_argument("--n-labeled", type=int, default=3000)
    ap.add_argument("--prior-noise", type=float, default=0.07)
    ap.add_argument("--ig-candidates", type=int, default=25)
    ap.add_argument("--ig-iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "subset_count.json"))
    args = ap.parse_args()

    cols = ["b0"] + [f"b{b}" for b in args.budgets]
    cases = load_cases(args)
    smallest = min(np.asarray(c["labeled"]).shape[1] for c in cases.values())
    need = max(args.subset_counts) * args.subset_size
    if need > smallest:
        sys.exit(f"K_max*size = {need} exceeds the smallest labeled pool ({smallest}); "
                 f"lower --subset-counts or --subset-size")

    payload = {}
    for name, c in cases.items():
        obs, lab = c["obs"], np.asarray(c["labeled"])
        N = obs.shape[1]
        print(f"\n  [{name}]  N={N}  M={obs.shape[0]}  labeled pool={lab.shape[1]}   "
              f"mean +/- sd over {args.draws} draws", flush=True)
        print(f"    {'K':>3s} {'items':>6s} {'t_pi(ms)':>10s} {'t_all(ms)':>10s}"
              + "".join(f"{k:>15s}" for k in cols))
        payload[name] = {}
        for k in args.subset_counts:
            got = {q: [] for q in cols}
            tp, ta = [], []
            for rep in range(args.draws):
                seed = args.seed + 977 * rep
                rng = np.random.default_rng(seed)
                subs = draw_subsets(lab.shape[1], k, args.subset_size, rng)
                idx = subs.reshape(-1)

                t_pi, pi = time_prior_accuracy(lab, subs, args.timing_reps)
                pi = np.clip(pi, 0.02, 0.98)
                t_all = time_full_prior(lab, idx, c["group"], pi, args.timing_reps)
                tp.append(t_pi * 1e3)
                ta.append((t_pi + t_all) * 1e3)

                # A K-subset prior knows only what K*size items can tell it. Its declared
                # sd is the binomial SE at that size, floored at the transfer noise so a
                # tiny subsample cannot claim an anchor stronger than domain shift allows.
                sigma = np.maximum(np.sqrt(pi * (1 - pi) / idx.size),
                                   np.asarray(c["prior_sigma"], float))
                s = _strength(pi, sigma, cap=N)

                st = build_stats(lab[:, idx], c["group"], pi)
                got["b0"].append(mae(validated_em(obs, st, pi, s,
                                                  max_iters=200)["acc"], c["true_acc"]))
                for b in args.budgets:
                    st2 = build_stats(lab[:, idx], c["group"], pi)
                    o = run_validation(obs, st2, pi, s,
                                       expert=OracleExpert(obs, allow_none=True),
                                       budget=b, select="mean_gain",
                                       ig_candidates=args.ig_candidates,
                                       ig_iters=args.ig_iters, seed=seed)
                    got[f"b{b}"].append(mae(o["acc"], c["true_acc"]))

            payload[name][str(k)] = dict(
                items=k * args.subset_size,
                t_pi_ms=[float(np.mean(tp)), float(np.std(tp))],
                t_all_ms=[float(np.mean(ta)), float(np.std(ta))],
                **{q: [float(np.mean(v)), float(np.std(v))] for q, v in got.items()})
            print(f"    {k:>3d} {k*args.subset_size:>6d} {np.mean(tp):>10.4f}"
                  f" {np.mean(ta):>10.4f}"
                  + "".join(f"{np.mean(got[q]):9.2f}+-{np.std(got[q]):<4.2f}"
                            for q in cols), flush=True)

    # ---- aggregate: per modality, then the unweighted mean over the 3 modalities ----
    mods = {}
    for name in payload:
        mods.setdefault(name.split("/")[0], []).append(name)

    agg = {}
    for k in args.subset_counts:
        per_mod = {}
        for m, names in mods.items():
            per_mod[m] = {f: float(np.mean([payload[n][str(k)][f][0] for n in names]))
                          for f in ["t_pi_ms", "t_all_ms"] + cols}
        agg[str(k)] = dict(
            items=k * args.subset_size,
            per_modality=per_mod,
            **{f: float(np.mean([per_mod[m][f] for m in per_mod]))
               for f in ["t_pi_ms", "t_all_ms"] + cols})

    print("\n" + "=" * 86)
    print(f"  MEAN over the {len(mods)} modalities "
          f"(each modality = mean of its {len(next(iter(mods.values())))} cases)")
    print("=" * 86)
    print(f"  {'K':>3s} {'items':>6s} {'t_pi(ms)':>10s} {'t_all(ms)':>10s}"
          + "".join(f"{q:>10s}" for q in cols))
    for k in args.subset_counts:
        a = agg[str(k)]
        print(f"  {k:>3d} {a['items']:>6d} {a['t_pi_ms']:>10.4f} {a['t_all_ms']:>10.4f}"
              + "".join(f"{a[q]:>10.2f}" for q in cols))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as h:
        json.dump(dict(table=payload, aggregate=agg, metadata=vars(args)), h,
                  indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
