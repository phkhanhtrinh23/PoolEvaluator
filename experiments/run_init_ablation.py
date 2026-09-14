"""Where do the starting values come from, and does the pipeline need a labeled corpus?

The pipeline currently draws THREE things from a pre-built labeled split: the per-model
accuracy prior pi_j (the Beta anchor centre), the correlated-error matrix e, and the
gamma vector. An earlier anchor sweep zeroed only the WEIGHT on pi_j while e and gamma
still came from labeled data, so it never tested the interesting question: can the expert
budget alone carry the method?

Two axes are crossed here.

  init      where the per-model accuracy starts
            prior      the labeled split's source accuracy  (current default)
            uniform    0.5 for every model, anchor strength 0
            random     drawn from Uniform(0.15, 0.95) per model, anchor strength 0
            agreement  estimated from unlabeled agreement with the pool consensus.
                       Realistic but CIRCULAR -- a model that agrees with a wrong
                       consensus scores as accurate -- so it is expected to fail exactly
                       where the consensus is broken, and that is worth seeing rather
                       than assuming.

  stats     where e and gamma start
            labeled    measured on the pre-built labeled split  (current default)
            none       e = 0 (no discount) and gamma = 0.5 (uninformative); both are then
                       learned only from the items the expert validates

`init=random` is averaged over several draws, since a single draw says nothing.

  python experiments/run_init_ablation.py --seeds 5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval.domains.adapter import encode_classes                   # noqa: E402
from pooleval.validated_em import (LabeledStatistics, LatentPlan,     # noqa: E402
                                   OracleExpert, hard_labels,
                                   run_validation, validated_em)
from experiments.run_validated_em import (_strength, _subsample,      # noqa: E402
                                          build_stats, mae)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")


def uninformed_stats(M, group, gamma_mode="model_wrong"):
    """LabeledStatistics with NO labeled split: no discount, uninformative gamma.

    Built from a one-item placeholder so the counters exist, then overwritten. Everything
    it will ever know comes from `add_validated`.
    """
    placeholder = np.zeros((M, 1), dtype=np.int64)
    st = LabeledStatistics(placeholder, np.zeros(1, dtype=np.int64), group=group,
                           gamma_mode=gamma_mode)
    st._e = np.zeros((M, M))
    st._gamma = np.full(M, 0.5)
    st._pseudo_acc = 0.5
    if getattr(st, "pair_same", None) is not None:
        st.pair_same = np.zeros((M, M))
        st.pair_wrong = np.zeros(M)
    return st


def make_prior(kind, c, rng):
    """-> (pi, anchor strength). Only `prior` uses the labeled split."""
    M = c["obs"].shape[0]
    if kind == "prior":
        return c["prior"], _strength(c["prior"], c["prior_sigma"], cap=c["obs"].shape[1])
    if kind == "uniform":
        return np.full(M, 0.5), 0.0
    if kind == "random":
        return rng.uniform(0.15, 0.95, size=M), 0.0
    if kind == "agreement":
        # circular by construction: score each model against the pool's own consensus
        obs = c["obs"]
        yhat = hard_labels(LatentPlan(obs, np.zeros((M, M))).posterior(np.full(M, 0.5)))
        return np.clip((obs == yhat[None, :]).mean(axis=1), 0.02, 0.98), 0.0
    raise ValueError(kind)


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


ARMS = [("prior", "labeled"), ("uniform", "labeled"), ("random", "labeled"),
        ("agreement", "labeled"), ("uniform", "none"), ("random", "none")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 40])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-target", type=int, default=600)
    ap.add_argument("--n-labeled", type=int, default=3000)
    ap.add_argument("--prior-noise", type=float, default=0.07)
    ap.add_argument("--ig-candidates", type=int, default=25)
    ap.add_argument("--ig-iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "init_ablation.json"))
    args = ap.parse_args()

    payload = {}
    cols = ["b0"] + [f"b{b}" for b in args.budgets]
    for name, c in load_cases(args).items():
        obs, M = c["obs"], c["obs"].shape[0]
        print(f"\n  [{name}]  N={obs.shape[1]}  M={M}")
        print(f"    {'init':11s}{'stats':9s}" + "".join(f"{k:>9s}" for k in cols))
        payload[name] = {}
        for init, stats_src in ARMS:
            got = {k: [] for k in cols}
            for rep in range(args.seeds):
                rng = np.random.default_rng(args.seed + 1000 * rep)
                pi, s = make_prior(init, c, rng)

                def fresh():
                    return (build_stats(c["labeled"], c["group"], pi)
                            if stats_src == "labeled"
                            else uninformed_stats(M, c["group"]))

                got["b0"].append(mae(validated_em(obs, fresh(), pi, s,
                                                  max_iters=200)["acc"], c["true_acc"]))
                for b in args.budgets:
                    o = run_validation(obs, fresh(), pi, s,
                                       expert=OracleExpert(obs, allow_none=True),
                                       budget=b, select="mean_gain",
                                       ig_candidates=args.ig_candidates,
                                       ig_iters=args.ig_iters,
                                       seed=args.seed + 1000 * rep)
                    got[f"b{b}"].append(mae(o["acc"], c["true_acc"]))
            row = {k: float(np.mean(v)) for k, v in got.items()}
            payload[name][f"{init}/{stats_src}"] = row
            print(f"    {init:11s}{stats_src:9s}" +
                  "".join(f"{row[k]:9.2f}" for k in cols), flush=True)

    print("\n" + "=" * 70 + f"\n  MEANS over {len(payload)} cases\n" + "=" * 70)
    print(f"  {'init':11s}{'stats':9s}" + "".join(f"{k:>9s}" for k in cols))
    for init, stats_src in ARMS:
        key = f"{init}/{stats_src}"
        print(f"  {init:11s}{stats_src:9s}" +
              "".join(f"{np.mean([payload[n][key][k] for n in payload]):9.2f}"
                      for k in cols))
    with open(args.out, "w") as h:
        json.dump(dict(table=payload, metadata=vars(args)), h, indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
