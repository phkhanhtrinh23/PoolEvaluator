"""Deployment option: how cheap a prior can we get away with?

All four starting values -- pi, beta, gamma, e -- are read off ONE artifact: the M models
run on a labeled subset. So the deployment question is not "which prior do we compute" but
"how many labeled items do we run the models on", because that is what costs money and
latency. The ablation in docs/random_init.md says only pi carries real information, and pi
is a per-model mean, which needs far fewer items than the MxM matrix e. That suggests a
cheap middle mode: spend a small labeled subset on pi alone and randomise the rest.

Modes compared here:

  measured    every prior from the full labeled split          (most expensive)
  pi@n        pi from an n-item subsample, beta/gamma/e random  (cheap, n varies)
  random      nothing measured, no anchor                       (free)

Spread across draws is reported because the cheap modes are stochastic and a mode that is
good on average but wild across draws is not deployable.

  python experiments/run_prior_modes.py --draws 10
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval.validated_em import (OracleExpert, run_validation,      # noqa: E402
                                   validated_em)
from experiments.run_validated_em import _strength, build_stats, mae  # noqa: E402
from experiments.run_random_init import load_cases                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def randomise_rest(st, M, rng):
    """Replace every prior EXCEPT pi with a uniform draw, using the ranges from the
    random-init ablation so the two experiments stay comparable."""
    st._pseudo_acc = float(rng.uniform(0.05, 0.95))
    st._gamma = rng.uniform(0.05, 0.95, size=M)
    a = rng.uniform(0.0, 0.5, size=(M, M))
    a = (a + a.T) / 2.0
    np.fill_diagonal(a, rng.uniform(0.0, 0.8, size=M))
    st._e = a
    return st


def make_start(mode, c, rng):
    """-> (pi, anchor strength, stats) for one deployment mode."""
    M, N = c["obs"].shape
    lab = np.asarray(c["labeled"])

    if mode == "measured":
        pi = np.asarray(c["prior"], float)
        return pi, _strength(pi, c["prior_sigma"], cap=N), build_stats(
            lab, c["group"], pi)

    if mode == "random":
        pi = rng.uniform(0.15, 0.95, size=M)
        return pi, 0.0, randomise_rest(build_stats(lab, c["group"], pi), M, rng)

    # pi@n -- estimate pi on an n-item subsample of the labeled split, randomise the rest.
    n = int(mode.split("@")[1])
    if n >= lab.shape[1]:
        pi = np.asarray(c["prior"], float)
        sigma = np.asarray(c["prior_sigma"], float)
    else:
        idx = rng.choice(lab.shape[1], size=n, replace=False)
        pi = (lab[:, idx] == 0).mean(axis=1)
        pi = np.clip(pi, 0.02, 0.98)
        # Honest uncertainty for a size-n estimate: binomial SE, floored so a degenerate
        # subsample cannot manufacture an infinitely strong anchor. Never let it claim to
        # be more certain than the declared transfer noise.
        sigma = np.maximum(np.sqrt(pi * (1 - pi) / n),
                           np.asarray(c["prior_sigma"], float))
    return pi, _strength(pi, sigma, cap=N), randomise_rest(
        build_stats(lab, c["group"], pi), M, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 40])
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--n-target", type=int, default=600)
    ap.add_argument("--n-labeled", type=int, default=3000)
    ap.add_argument("--prior-noise", type=float, default=0.07)
    ap.add_argument("--ig-candidates", type=int, default=25)
    ap.add_argument("--ig-iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", nargs="+",
                    default=["measured", "pi@500", "pi@200", "pi@50", "random"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "prior_modes.json"))
    args = ap.parse_args()

    cols = ["b0"] + [f"b{b}" for b in args.budgets]
    payload, agg = {}, {m: {k: [] for k in cols} for m in args.modes}

    for name, c in load_cases(args).items():
        obs = c["obs"]
        print(f"\n  [{name}]  N={obs.shape[1]}  M={obs.shape[0]}   "
              f"mean +/- sd over {args.draws} draws", flush=True)
        print(f"    {'mode':14s}" + "".join(f"{k:>16s}" for k in cols))
        payload[name] = {}
        for mode in args.modes:
            got = {k: [] for k in cols}
            for rep in range(args.draws):
                seed = args.seed + 977 * rep
                pi, s, st = make_start(mode, c, np.random.default_rng(seed))
                got["b0"].append(mae(validated_em(obs, st, pi, s,
                                                  max_iters=200)["acc"],
                                     c["true_acc"]))
                for b in args.budgets:
                    pi2, s2, st2 = make_start(mode, c, np.random.default_rng(seed))
                    o = run_validation(obs, st2, pi2, s2,
                                       expert=OracleExpert(obs, allow_none=True),
                                       budget=b, select="mean_gain",
                                       ig_candidates=args.ig_candidates,
                                       ig_iters=args.ig_iters, seed=seed)
                    got[f"b{b}"].append(mae(o["acc"], c["true_acc"]))
            payload[name][mode] = {k: [float(np.mean(v)), float(np.std(v))]
                                   for k, v in got.items()}
            for k, v in got.items():
                agg[mode][k].append(float(np.mean(v)))
            print(f"    {mode:14s}" + "".join(
                f"{np.mean(got[k]):10.2f}+-{np.std(got[k]):<5.2f}" for k in cols),
                flush=True)

    print("\n" + "=" * 78)
    print(f"  MEANS over {len(payload)} cases")
    print("=" * 78)
    print(f"  {'mode':14s}" + "".join(f"{k:>16s}" for k in cols))
    for mode in args.modes:
        print(f"  {mode:14s}" + "".join(
            f"{np.mean(agg[mode][k]):10.2f}+-{np.std(agg[mode][k]):<5.2f}"
            for k in cols))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
