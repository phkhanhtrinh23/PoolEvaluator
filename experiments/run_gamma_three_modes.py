"""Three routes to P(C=0|Z=0), compared where it matters: downstream MAE.

  model_wrong  count the conditional directly on the labeled split. Exact there, but a
               function of the pseudo-labels, hence of alpha -- so it can drift as EM moves.
  both_wrong   count the both-wrong rate and convert with the global beta. Measured biased
               by 0.19-0.47 because the conversion assumes an independence that fails.
  pairwise     a row functional of the correlated-error matrix, conditioned on j being
               wrong: gamma_j = 1 - mean_k P(r^j = r^k | r^j wrong), diagonal included.
               Invariant to the EM state by construction (a function of answers and gold
               only), at the cost of a downward bias -- the pseudo-label is the WINNER of a
               weighted vote, so agreeing with it is agreeing with the modal wrong answer
               while a row mean asks about a typical one.

Two questions, both previously argued rather than measured:

  A. How far does gamma actually drift DURING a judge loop? The earlier drift check ran
     with no judge calls, where alpha barely moved, so it could not see the regime where
     the invariance argument would pay.
  B. Does any of it change downstream MAE?

  python experiments/run_gamma_three_modes.py --seeds 5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval.domains.adapter import encode_classes                   # noqa: E402
from pooleval.validated_em import (LatentPlan, OracleExpert,          # noqa: E402
                                   gamma_counts, gamma_from_counts,
                                   hard_labels, run_validation, validated_em)
from experiments.run_validated_em import (_strength, _subsample,      # noqa: E402
                                          build_stats, mae)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")
MODES = ["model_wrong", "both_wrong", "pairwise", "pairwise_calibrated"]


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
    for key in ["vision_mnist_usps", "vision_mnist_svhn",
                "graph_AC", "graph_CD", "graph_DA"]:
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


def drift(name, c, args, log):
    """Question A: how far does each route's gamma move as the judge loop runs?

    `model_wrong` is recomputed at the alpha the loop has reached; `pairwise` cannot move
    for that reason at all, so any movement it shows comes only from the labels the expert
    revealed. Reporting both separates 'drift because alpha moved' from 'update because we
    learned something'.
    """
    obs, lab = c["obs"], c["labeled"]
    s = _strength(c["prior"], c["prior_sigma"], cap=obs.shape[1])
    out = {}
    for mode in ("model_wrong", "pairwise"):
        st = build_stats(lab, c["group"], c["prior"], gamma_mode=mode)
        g0 = st.gamma.copy()
        a0 = validated_em(obs, st, c["prior"], s, max_iters=200)["alpha"]
        res = run_validation(obs, st, c["prior"], s,
                             expert=OracleExpert(obs, allow_none=True),
                             budget=args.budget, select="mean_gain",
                             ig_candidates=args.ig_candidates, ig_iters=args.ig_iters,
                             seed=args.seed)
        a1 = res["acc"]
        # gamma as it would be re-measured at the alpha the loop reached
        ex = st.e_excess()
        y1 = hard_labels(LatentPlan(lab, ex).posterior(a1))
        g_re = gamma_from_counts(*gamma_counts(lab, y1, mode="model_wrong")) \
            if mode == "model_wrong" else st.gamma
        out[mode] = dict(alpha_move=float(np.abs(a1 - a0).mean()),
                         gamma_update=float(np.abs(st.gamma - g0).mean()),
                         gamma_restale=float(np.abs(g_re - st.gamma).mean()))
    log(f"  {name:20s} alpha moved {out['model_wrong']['alpha_move']:.3f}"
        f" | model_wrong: update {out['model_wrong']['gamma_update']:.3f},"
        f" staleness {out['model_wrong']['gamma_restale']:.3f}"
        f" | pairwise: update {out['pairwise']['gamma_update']:.3f},"
        f" staleness {out['pairwise']['gamma_restale']:.3f}")
    return out


def head_to_head(name, c, args, log):
    """Question B: downstream MAE for the three routes, at several budgets."""
    obs, lab = c["obs"], c["labeled"]
    s = _strength(c["prior"], c["prior_sigma"], cap=obs.shape[1])
    res = {}
    for mode in MODES:
        st0 = build_stats(lab, c["group"], c["prior"], gamma_mode=mode)
        row = {"gamma": float(st0.conditional_gamma().mean()),
               "b0": mae(validated_em(obs, st0, c["prior"], s,
                                      max_iters=200)["acc"], c["true_acc"])}
        for b in args.budgets:
            vals = []
            for rep in range(args.seeds):
                st = build_stats(lab, c["group"], c["prior"], gamma_mode=mode)
                o = run_validation(obs, st, c["prior"], s,
                                   expert=OracleExpert(obs, allow_none=True),
                                   budget=b, select="mean_gain",
                                   ig_candidates=args.ig_candidates,
                                   ig_iters=args.ig_iters,
                                   seed=args.seed + 1000 * rep)
                vals.append(mae(o["acc"], c["true_acc"]))
            row[f"b{b}"] = float(np.mean(vals))
        res[mode] = row
    cols = ["gamma", "b0"] + [f"b{b}" for b in args.budgets]
    log(f"\n  [{name}]")
    log(f"    {'mode':14s}" + "".join(f"{k:>10s}" for k in cols))
    for mode in MODES:
        log(f"    {mode:14s}" + "".join(f"{res[mode][k]:10.3f}" for k in cols))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 40])
    ap.add_argument("--budget", type=int, default=40, help="budget for the drift probe")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-target", type=int, default=600)
    ap.add_argument("--n-labeled", type=int, default=3000)
    ap.add_argument("--prior-noise", type=float, default=0.07)
    ap.add_argument("--ig-candidates", type=int, default=25)
    ap.add_argument("--ig-iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "gamma_three_modes.json"))
    args = ap.parse_args()
    cases = load_cases(args)

    print("=" * 78)
    print("  A. gamma drift during a judge loop (budget %d)" % args.budget)
    print("     'update'    = how far gamma moved from the expert's revealed labels")
    print("     'staleness' = how far it is from gamma re-measured at the alpha reached")
    print("=" * 78)
    drifts = {n: drift(n, c, args, print) for n, c in cases.items()}

    print("\n" + "=" * 78)
    print("  B. downstream MAE, accuracy points. gamma = mean P(C=0|Z=0) fed to the E-step")
    print("=" * 78)
    table = {n: head_to_head(n, c, args, print) for n, c in cases.items()}

    print("\n" + "=" * 78 + "\n  MEANS over %d cases\n" % len(cases) + "=" * 78)
    cols = ["b0"] + [f"b{b}" for b in args.budgets]
    print(f"  {'mode':14s}" + "".join(f"{c:>10s}" for c in cols))
    for m in MODES:
        print(f"  {m:14s}" + "".join(f"{np.mean([table[n][m][c] for n in table]):10.2f}"
                                     for c in cols))
    with open(args.out, "w") as h:
        json.dump(dict(drift=drifts, table=table, metadata=vars(args)), h,
                  indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
