"""Does the ``new = (old + temp) / 2`` refresh survive the new A_mu acquisition?

The refresh rule folds each expert-revealed label back into the labeled statistics before
the warm-started re-solve.  It is applied to three quantities, and they are NOT the same
kind of object:

  e      P(two models wrong on the same answer)   -- a rate over MODEL PAIRS
  gamma  P(C = 0 | Z = 0)                         -- a rate CONDITIONAL on being wrong
  beta   P(pseudo-label = truth)                  -- a raw MEAN OVER ITEMS

Only the third is an average over the item population, which makes it vulnerable to
exactly the selection bias that motivates probability sampling for the accuracy mean
itself: an acquisition rule picks items BECAUSE the consensus looks doubtful there, so
``temp_beta`` computed on them is systematically low.  On an overruled item it is 0 by
construction, and a filter that hands the newest sample half the weight drives beta
geometrically toward 0.

This script measures whether that happens under each acquisition rule, including the new
mean-targeted one, and whether the pooled-counts alternative avoids it.

  python experiments/run_refresh_rule.py --seeds 3
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
RESULTS = os.path.join(ROOT, "results")
POOL_CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")

SELECTORS = ["entropy", "info_gain", "mean_gain", "mean_gain_sampled", "random"]
RULES = [("(old+temp)/2", {}), ("pooled counts", dict(update_rule="counts"))]


def load_cases(args):
    cases = {}
    from zoo.new_formulation_real import load_run
    src = np.load(os.path.join(ROOT, "zoo_artifacts", "source_true_class.npz"))
    for ds in ["spider", "bird"]:
        run, _ = load_run(ds)
        cases[f"text2sql/{ds}"] = dict(
            obs=run.true_class, labeled=src[ds], group=run.group, prior=run.prior,
            prior_sigma=run.prior_sigma, true_acc=run.true_acc)
    for key in ["vision_mnist_usps", "vision_mnist_svhn", "graph_AC", "graph_CD"]:
        path = os.path.join(POOL_CACHE, key + ".npz")
        if not os.path.exists(path):
            continue
        z = np.load(path, allow_pickle=True)
        pred, gold, group, prior = z["pred"], z["gold"], z["group"], z["prior"]
        pred_s = (z["pred_s"] if "pred_s" in z.files else z["prob_s"].argmax(-1))
        ti = _subsample(pred.shape[1], args.n_target, args.seed)
        si = _subsample(pred_s.shape[1], args.n_labeled, args.seed)
        obs = encode_classes(pred[:, ti], gold[ti])
        cases[("vision/" if key.startswith("vision") else "graph/") + key.split("_", 1)[1]] = dict(
            obs=obs, labeled=encode_classes(pred_s[:, si], z["src_gold_val"][si]),
            group=group, prior=prior,
            prior_sigma=np.full(len(prior), args.prior_noise),
            true_acc=(obs == 0).mean(axis=1))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-target", type=int, default=600)
    ap.add_argument("--n-labeled", type=int, default=3000)
    ap.add_argument("--prior-noise", type=float, default=0.07)
    ap.add_argument("--ig-candidates", type=int, default=25)
    ap.add_argument("--ig-iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "refresh_rule.json"))
    args = ap.parse_args()

    payload = {}
    for name, c in load_cases(args).items():
        obs, lab = c["obs"], c["labeled"]
        N = obs.shape[1]
        s = _strength(c["prior"], c["prior_sigma"], cap=N)
        fresh = lambda **kw: build_stats(lab, c["group"], c["prior"], **kw)
        base = validated_em(obs, fresh(), c["prior"], s, max_iters=200)
        truth_beta = float((base["pseudo_label"] == 0).mean())
        print(f"\n[{name}]  N={N}  anchor s={s:.0f}  "
              f"true pseudo-label accuracy {truth_beta:.3f}  "
              f"source-split beta {fresh().beta:.3f}")
        print(f"  {'selection':20s}" + "".join(f"{r[0]:>32s}" for r in RULES))
        print(f"  {'':20s}" + "".join(f"{'beta':>14s}{'MAE':>10s}{'':>8s}" for _ in RULES))
        rows = {}
        for sel in SELECTORS:
            cells = []
            for rule_name, skw in RULES:
                betas, maes = [], []
                for rep in range(args.seeds):
                    st = fresh(**skw)
                    out = run_validation(obs, st, c["prior"], s,
                                         expert=OracleExpert(obs, allow_none=True),
                                         budget=args.budget, select=sel,
                                         ig_candidates=args.ig_candidates,
                                         ig_iters=args.ig_iters,
                                         seed=args.seed + 1000 * rep)
                    betas.append(st.beta)
                    maes.append(mae(out["acc"], c["true_acc"]))
                cells.append((float(np.mean(betas)), float(np.mean(maes))))
            rows[sel] = {RULES[k][0]: dict(beta=cells[k][0], mae=cells[k][1])
                         for k in range(len(RULES))}
            print(f"  {sel:20s}" + "".join(f"{b:14.4f}{m:10.2f}{'':>8s}"
                                           for b, m in cells), flush=True)
        payload[name] = dict(true_beta=truth_beta, source_beta=fresh().beta,
                             N=N, strength=s, rows=rows)
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as h:
        json.dump(dict(payload, metadata=vars(args)), h, indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
