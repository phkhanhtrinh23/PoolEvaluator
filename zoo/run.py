"""End-to-end real zoo run on Spider.

  generate (dev + source) -> execute -> PoolRun -> PoolEval + baselines vs true EX.

    python -m zoo.run --n-target 150 --n-source 120

Writes zoo_artifacts/poolrun_dev.npz and results/zoo_spider_main.json.
All API generations are cached, so re-runs are cheap and resumable.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import Config, PoolEval, metrics          # noqa: E402
from baselines import Independent, Majority, DawidSkene, AgreementLine  # noqa: E402
from zoo.config import ZooConfig, ARTIFACT_ROOT         # noqa: E402
from zoo.data_spider import load_spider_split           # noqa: E402
from zoo.generate import generate_predictions           # noqa: E402
from zoo.build import build_poolrun                      # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(ARTIFACT_ROOT, exist_ok=True)


def main(n_target=150, n_source=120, seed=0):
    zcfg = ZooConfig(n_target=n_target, n_source=n_source, seed=seed)
    members = zcfg.manifest
    names = [m.name for m in members]
    groups = zcfg.group_ids()

    print(f"[zoo] {len(members)} members / {len(zcfg.groups)} provenance groups")
    print("      groups:", dict(zip(names, [m.group for m in members])))

    dev = load_spider_split("dev", zcfg.n_target, seed=seed)
    src = load_spider_split("train", zcfg.n_source, seed=seed)
    print(f"[zoo] dev items={len(dev)}  source items={len(src)}")

    print("[zoo] generating dev predictions ...")
    preds_dev = generate_predictions(members, dev, zcfg, tag="dev")
    print("[zoo] generating source predictions (for the seen prior) ...")
    preds_src = generate_predictions(members, src, zcfg, tag="src")

    print("[zoo] executing + building PoolRun ...")
    run, keys = build_poolrun(dev, names, groups, preds_dev, preds_src, src,
                              timeout=zcfg.exec_timeout)

    # persist the PoolRun so downstream analysis needs no re-execution
    np.savez(os.path.join(ARTIFACT_ROOT, "poolrun_dev.npz"),
             true_class=run.true_class, true_acc=run.true_acc, group=run.group,
             prior=run.prior, prior_sigma=run.prior_sigma,
             verifier_guess=run.verifier_guess, names=np.array(names))

    print("\n[zoo] real dev EX per member:")
    for m, name in enumerate(names):
        print(f"    {name:20s} EX={run.true_acc[m]:.3f}  prior(src)={run.prior[m]:.3f}"
              f"  group={zcfg.groups[run.group[m]]}")
    print(f"    verifier real accuracy (points at correct): "
          f"{run.verifier_correct.mean():.3f}")

    # --- estimators on the REAL PoolRun (kernel = identity) ---
    cfg = Config(real_data=True, use_prior=True, use_verifier=True,
                 use_correlation=True, fusion="precision")
    methods = {"B1 Independent": Independent(), "B2 Majority/self-cons.": Majority(),
               "B3 Dawid--Skene": DawidSkene(), "B4 Agreement-on-line": AgreementLine()}
    rows = {}
    for name, b in methods.items():
        rows[name] = metrics.all_metrics(b.evaluate(run, cfg), run.true_acc)
    out = PoolEval(cfg).evaluate(run)
    rows["PoolEval-SQL (ours)"] = metrics.all_metrics(out["acc"], run.true_acc)

    print("\n=== Spider real-zoo ranking (M={}, N={}) ===".format(run.M, run.N))
    keys_order = ["MAE", "Flip", "Kendall", "Top1", "Top3"]
    print(f"{'Method':24s}" + "".join(f"{k:>10s}" for k in keys_order))
    for name in list(methods) + ["PoolEval-SQL (ours)"]:
        print(f"{name:24s}" + "".join(f"{rows[name][k]:>10.3f}" for k in keys_order))
    print(f"\nM_eff={out['Meff']:.1f} of {run.M}   collusion_flag={out['collusion']}")

    payload = {"dataset": "spider-dev", "M": run.M, "N": run.N,
               "groups": zcfg.groups, "members": names,
               "true_acc": run.true_acc.tolist(), "prior": run.prior.tolist(),
               "verifier_acc": float(run.verifier_correct.mean()),
               "Meff": float(out["Meff"]), "metrics": rows}
    with open(os.path.join(RESULTS, "zoo_spider_main.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"[saved] {os.path.join(RESULTS, 'zoo_spider_main.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=150)
    ap.add_argument("--n-source", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(n_target=a.n_target, n_source=a.n_source, seed=a.seed)
