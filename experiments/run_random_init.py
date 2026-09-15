"""How much does the pipeline actually depend on the pre-built prior?

Everything the pipeline needs to start is randomised, one component at a time and then all
together, over many draws. The point is to separate two very different kinds of starting
value:

  RE-ESTIMATED BY EM        alpha (per-model accuracy) and beta (pseudo-label quality) are
                            recomputed in every M-step, so a bad start should wash out.
  NEVER RE-ESTIMATED BY EM  gamma and e are measured once on the labeled split and are only
                            ever changed by expert validation. EM never touches them, so a
                            bad start persists unless the judge repairs it.

If the prior matters mainly through alpha, randomising it should cost little and recover
with budget. If it matters through gamma and e, randomising those should be far more
damaging and recover only as the expert reveals labels. That distinction is the experiment.

Spread across draws is reported alongside the mean, because a method that is good on
average and wild across initialisations is not usable.

  python experiments/run_random_init.py --draws 10
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval.domains.adapter import encode_classes                   # noqa: E402
from pooleval.validated_em import (OracleExpert, run_validation,      # noqa: E402
                                   validated_em)
from experiments.run_validated_em import (_strength, _subsample,      # noqa: E402
                                          build_stats, mae)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")

#          name                  pi      beta     gamma    e        anchor
ARMS = [("measured (baseline)",  False,  False,   False,   False,   True),
        ("random pi",            True,   False,   False,   False,   False),
        ("random beta",          False,  True,    False,   False,   True),
        ("random gamma",         False,  False,   True,    False,   True),
        ("random e",             False,  False,   False,   True,    True),
        ("random pi+beta",       True,   True,    False,   False,   False),
        ("random gamma+e",       False,  False,   True,    True,    True),
        ("random EVERYTHING",    True,   True,    True,    True,    False)]


def make_start(arm, c, rng):
    """-> (pi, anchor strength, stats, beta_init). Only components flagged are randomised."""
    _, rnd_pi, rnd_beta, rnd_gamma, rnd_e, use_anchor = arm
    M = c["obs"].shape[0]
    pi = rng.uniform(0.15, 0.95, size=M) if rnd_pi else np.asarray(c["prior"], float)
    s = (_strength(c["prior"], c["prior_sigma"], cap=c["obs"].shape[1])
         if use_anchor else 0.0)
    # stats are always built from the labeled split, then selected pieces overwritten
    st = build_stats(c["labeled"], c["group"], pi)
    if rnd_gamma:
        st._gamma = rng.uniform(0.05, 0.95, size=M)
    if rnd_e:
        a = rng.uniform(0.0, 0.5, size=(M, M))
        a = (a + a.T) / 2.0
        np.fill_diagonal(a, rng.uniform(0.0, 0.8, size=M))
        st._e = a
    if rnd_beta:
        # validated_em initialises beta from stats.pseudo_accuracy when `init` has no
        # "beta" key, so setting it here randomises the start without colliding with the
        # `init` argument that acquisition_scores passes positionally.
        st._pseudo_acc = float(rng.uniform(0.05, 0.95))
    return pi, s, st, None


def load_cases(args):
    cases = {}
    from zoo.new_formulation_real import load_run
    src = np.load(os.path.join(ROOT, "zoo_artifacts", "source_true_class.npz"))
    for ds in ["spider", "bird"]:
        run, _ = load_run(ds)
        cases[f"text2sql/{ds}"] = dict(obs=run.true_class, labeled=src[ds],
                                       group=run.group, prior=run.prior,
                                       prior_sigma=run.prior_sigma,
                                       true_acc=run.true_acc)
    for key in ["vision_mnist_usps", "vision_mnist_svhn", "graph_AC", "graph_DA"]:
        path = os.path.join(POOL_CACHE, key + ".npz")
        if not os.path.exists(path):
            continue
        z = np.load(path, allow_pickle=True)
        pred, gold, prior = z["pred"], z["gold"], z["prior"]
        pred_s = (z["pred_s"] if "pred_s" in z.files else z["prob_s"].argmax(-1))
        ti = _subsample(pred.shape[1], args.n_target, args.seed)
        si = _subsample(pred_s.shape[1], args.n_labeled, args.seed)
        obs = encode_classes(pred[:, ti], gold[ti])
        dom = "vision/" if key.startswith("vision") else "graph/"
        cases[dom + key.split("_", 1)[1]] = dict(
            obs=obs, labeled=encode_classes(pred_s[:, si], z["src_gold_val"][si]),
            group=z["group"], prior=prior,
            prior_sigma=np.full(len(prior), args.prior_noise),
            true_acc=(obs == 0).mean(axis=1))
    return cases


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
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "random_init.json"))
    args = ap.parse_args()

    cols = ["b0"] + [f"b{b}" for b in args.budgets]
    payload = {}
    for name, c in load_cases(args).items():
        obs = c["obs"]
        print(f"\n  [{name}]  N={obs.shape[1]}  M={obs.shape[0]}   "
              f"mean +/- sd over {args.draws} draws", flush=True)
        print(f"    {'arm':22s}" + "".join(f"{k:>16s}" for k in cols))
        payload[name] = {}
        for arm in ARMS:
            got = {k: [] for k in cols}
            for rep in range(args.draws):
                rng = np.random.default_rng(args.seed + 977 * rep)
                pi, s, st, beta_init = make_start(arm, c, rng)
                got["b0"].append(mae(validated_em(obs, st, pi, s,
                                                  max_iters=200)["acc"], c["true_acc"]))
                for b in args.budgets:
                    _, _, st2, _ = make_start(arm, c, np.random.default_rng(
                        args.seed + 977 * rep))
                    o = run_validation(obs, st2, pi, s,
                                       expert=OracleExpert(obs, allow_none=True),
                                       budget=b, select="mean_gain",
                                       ig_candidates=args.ig_candidates,
                                       ig_iters=args.ig_iters,
                                       seed=args.seed + 977 * rep)
                    got[f"b{b}"].append(mae(o["acc"], c["true_acc"]))
            row = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v))) for k, v in got.items()}
            payload[name][arm[0]] = row
            print(f"    {arm[0]:22s}" + "".join(
                f"{row[k]['mean']:9.2f}+-{row[k]['sd']:<5.2f}" for k in cols), flush=True)

    print("\n" + "=" * 78 + f"\n  MEANS over {len(payload)} cases\n" + "=" * 78)
    print(f"  {'arm':22s}" + "".join(f"{k:>16s}" for k in cols))
    for arm in ARMS:
        print(f"  {arm[0]:22s}" + "".join(
            f"{np.mean([payload[n][arm[0]][k]['mean'] for n in payload]):9.2f}"
            f"+-{np.mean([payload[n][arm[0]][k]['sd'] for n in payload]):<5.2f}"
            for k in cols))
    with open(args.out, "w") as h:
        json.dump(dict(table=payload, metadata=vars(args)), h, indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
