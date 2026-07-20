"""Active PoolEval-SQL on the REAL Spider zoo, with a gpt-5-mini judge.

Loads the pre-built real PoolRun (zoo_artifacts/poolrun_dev.npz + cached preds), runs
label-free PoolEval-SQL, then spends a small gpt-5-mini judge budget on the items the
submodular selector deems most ambiguous (near-clone-dominated consensus), and reports
how the deployment ranking moves toward the true dev execution accuracy.

    python -m zoo.run_active --budget 12                 # real gpt-5-mini judge
    python -m zoo.run_active --budget 12 --mock          # oracle judge, no API cost
    python -m zoo.run_active --budget 12 --strategy disagreement

Requires zoo_artifacts/poolrun_dev.npz (build it with `python -m zoo.run`). Real runs
call the OpenAI API for `--budget` items only (all cached and resumable).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import Config, PoolEval, ActivePoolEval, ActiveConfig, metrics  # noqa: E402
from pooleval import SimulatedJudge                                           # noqa: E402
from pooleval.data.simulator import PoolRun                                   # noqa: E402
from zoo.config import ZooConfig, ARTIFACT_ROOT                              # noqa: E402
from zoo.data_spider import load_spider_split                                # noqa: E402
from zoo.judge import RealJudge                                              # noqa: E402
from pooleval.active import abstention                                       # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


def load_poolrun():
    d = np.load(os.path.join(ARTIFACT_ROOT, "poolrun_dev.npz"), allow_pickle=True)
    tc = d["true_class"]
    M, N = tc.shape
    group = d["group"]
    vg = d["verifier_guess"]
    run = PoolRun(true_class=tc, true_acc=d["true_acc"], group=group,
                  prior=d["prior"], prior_sigma=d["prior_sigma"],
                  b=np.zeros(N), phi=np.ones(N),           # unused by the estimator
                  verifier_guess=vg, verifier_correct=(vg == 0),
                  M=M, N=N, n_groups=int(group.max()) + 1)
    return run, [str(x) for x in d["names"]]


def main(budget=12, strategy="hybrid_submodular", n_target=150, seed=0, mock=False,
         model="gpt-5-mini", rounds=4):
    run, names = load_poolrun()
    cfg = Config(real_data=True)
    print(f"[zoo-active] real PoolRun: M={run.M} N={run.N}  members={names}")

    base = PoolEval(cfg).evaluate(run)
    base_m = metrics.all_metrics(base["acc"], run.true_acc)
    base_ab = abstention(base, run, base["obs"])
    print(f"[zoo-active] pure PoolEval-SQL: Top1={base_m['Top1']:.0f} "
          f"Kendall={base_m['Kendall']:.2f} MAE={base_m['MAE']:.2f} "
          f"Meff={base['Meff']:.1f}  flagged={base_ab['n_flagged']} items")

    if mock:
        judge = SimulatedJudge()
        print("[zoo-active] MOCK judge (oracle, no API calls)")
    else:
        dev = load_spider_split("dev", n_target, seed=seed)   # aligns with build order
        judge = RealJudge(dev, names, _load_preds(), run.true_class, model=model,
                          verbose=True)
        print(f"[zoo-active] real judge = {model} on the {budget} most ambiguous items")

    out = ActivePoolEval(cfg, ActiveConfig(budget=budget, rounds=rounds,
                         strategy=strategy), judge=judge).run(run)
    act_m = metrics.all_metrics(out["acc"], run.true_acc)
    print(f"[zoo-active] active ({strategy}, {out['judge_calls']} judge calls): "
          f"Top1={act_m['Top1']:.0f} Kendall={act_m['Kendall']:.2f} "
          f"MAE={act_m['MAE']:.2f}")

    order_true = np.argsort(-run.true_acc)
    print("\n rank | true-EX order        | pure order           | active order")
    pure_o, act_o = np.argsort(-base["acc"]), np.argsort(-out["acc"])
    for r in range(run.M):
        print(f"  {r:2d}  | {names[order_true[r]]:20s} | {names[pure_o[r]]:20s} "
              f"| {names[act_o[r]]:20s}")

    payload = dict(dataset="spider-dev", M=int(run.M), N=int(run.N), budget=int(budget),
                   strategy=strategy, mock=bool(mock), model=(None if mock else model),
                   judge_calls=int(out["judge_calls"]),
                   validated_items=[int(i) for i in out["constraints"].keys()],
                   pure=base_m, active=act_m,
                   true_acc=run.true_acc.tolist(),
                   pure_acc=base["acc"].tolist(), active_acc=out["acc"].tolist(),
                   judge_log=getattr(judge, "log", []))
    os.makedirs(RESULTS, exist_ok=True)
    tag = "mock" if mock else strategy
    path = os.path.join(RESULTS, f"zoo_active_{tag}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"[saved] {path}")


def _load_preds():
    with open(os.path.join(ARTIFACT_ROOT, "preds_dev.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--strategy", default="hybrid_submodular")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-target", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    main(budget=a.budget, strategy=a.strategy, rounds=a.rounds, n_target=a.n_target,
         seed=a.seed, mock=a.mock, model=a.model)
