"""Which `gamma_mode` is right *for the headline configuration* -- the judge loop with A_mu?

The global tally across every method favours `model_wrong` (41 better, 19 worse). But the
losses are not spread evenly: deterministic `A_mu` acquisition prefers `both_wrong` on all
four blocks, and it is the only method that does so consistently. If the paper's headline
result is the judge loop driven by A_mu, the global tally is the wrong statistic to decide
on and this comparison is the right one.

The plausible mechanism, worth stating so the result can falsify it: A_mu scores an item by
how far the truth would MOVE the model. A model carrying a biased gamma has more to be
moved by, so A_mu has more leverage to find. Removing the bias would then remove some of
A_mu's advantage -- meaning `both_wrong` would look better under A_mu for a reason that is
about A_mu, not about gamma being correct. If that is what is happening, the effect should
(a) hold for the deterministic argmax but not the sampled variant, and (b) shrink as the
budget grows and the judge labels dominate the model's beliefs.

  python experiments/run_gamma_mode_headline.py --seeds 5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval.domains.adapter import encode_classes                   # noqa: E402
from pooleval.validated_em import OracleExpert, run_validation        # noqa: E402
from experiments.run_validated_em import (_strength, _subsample,      # noqa: E402
                                          build_stats, mae)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")
MODES = ["both_wrong", "model_wrong"]
SELECTORS = ["mean_gain", "mean_gain_sampled"]


def load_cases(args):
    """`--part a` is the leak-free half (protocol A + a graph sample); `--part b` is the
    complement (target-holdout protocol + the remaining graph shifts). Together they cover
    every case available, so the two runs can be reported as one table."""
    cases = {}
    from zoo.new_formulation_real import load_run, subset_run
    src = np.load(os.path.join(ROOT, "zoo_artifacts", "source_true_class.npz"))
    if args.part == "a":
        for ds in ["spider", "bird"]:
            run, _ = load_run(ds)
            cases[f"text2sql/{ds}/A"] = dict(obs=run.true_class, labeled=src[ds],
                                             group=run.group, prior=run.prior,
                                             prior_sigma=run.prior_sigma,
                                             true_acc=run.true_acc)
        keys = ["vision_mnist_usps", "vision_mnist_svhn",
                "graph_AC", "graph_CD", "graph_DA"]
    else:
        from experiments.run_validated_em import split_indices
        for ds in ["spider", "bird", "sqlflow", "bird_minidev", "spider2local"]:
            run, _ = load_run(ds)
            cal, ev = split_indices(run.N, 0.4, seed=args.seed)
            sub = subset_run(run, ev)
            cases[f"text2sql/{ds}/B"] = dict(obs=sub.true_class,
                                             labeled=run.true_class[:, cal],
                                             group=run.group, prior=run.prior,
                                             prior_sigma=run.prior_sigma,
                                             true_acc=sub.true_acc)
        keys = ["graph_AD", "graph_CA", "graph_DC"]
    for key in keys:
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
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 20, 40])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-target", type=int, default=600)
    ap.add_argument("--n-labeled", type=int, default=3000)
    ap.add_argument("--prior-noise", type=float, default=0.07)
    ap.add_argument("--ig-candidates", type=int, default=25)
    ap.add_argument("--ig-iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", choices=["a", "b"], default="a")
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "gamma_mode_headline.json"))
    args = ap.parse_args()

    payload = {}
    for name, c in load_cases(args).items():
        obs, lab = c["obs"], c["labeled"]
        s = _strength(c["prior"], c["prior_sigma"], cap=obs.shape[1])
        print(f"\n[{name}]  N={obs.shape[1]}  anchor s={s:.0f}", flush=True)
        print(f"  {'selector':20s}{'budget':>8s}" +
              "".join(f"{m:>16s}" for m in MODES) + f"{'delta':>10s}")
        payload[name] = {}
        for sel in SELECTORS:
            for b in args.budgets:
                got = {}
                for mode in MODES:
                    vals = []
                    for rep in range(args.seeds):
                        st = build_stats(lab, c["group"], c["prior"], gamma_mode=mode)
                        out = run_validation(
                            obs, st, c["prior"], s,
                            expert=OracleExpert(obs, allow_none=True), budget=b,
                            select=sel, ig_candidates=args.ig_candidates,
                            ig_iters=args.ig_iters, seed=args.seed + 1000 * rep)
                        vals.append(mae(out["acc"], c["true_acc"]))
                    got[mode] = float(np.mean(vals))
                d = got["model_wrong"] - got["both_wrong"]
                payload[name][f"{sel}/b{b}"] = got
                print(f"  {sel:20s}{b:8d}" +
                      "".join(f"{got[m]:16.2f}" for m in MODES) + f"{d:+10.2f}",
                      flush=True)
    with open(args.out, "w") as h:
        json.dump(dict(payload, metadata=vars(args)), h, indent=2, default=float)
    summarize(payload, args)
    print(f"\n[saved] {args.out}")


def summarize(payload, args):
    print(f"\n{'=' * 78}\n  model_wrong minus both_wrong  (negative = model_wrong better)"
          f"\n{'=' * 78}")
    print(f"  {'selector':22s}" + "".join(f"{'b=' + str(b):>12s}" for b in args.budgets))
    for sel in SELECTORS:
        row = []
        for b in args.budgets:
            d = [v[f"{sel}/b{b}"]["model_wrong"] - v[f"{sel}/b{b}"]["both_wrong"]
                 for v in payload.values() if f"{sel}/b{b}" in v]
            row.append(float(np.mean(d)))
        print(f"  {sel:22s}" + "".join(f"{x:+12.2f}" for x in row))


if __name__ == "__main__":
    main()
