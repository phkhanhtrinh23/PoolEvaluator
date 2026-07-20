"""Real zoo + Active PoolEval-SQL across MULTIPLE datasets.

For each dataset: generate the model-pool predictions (cached/resumable), execute on
the live SQLite DBs to build the real PoolRun, then run label-free PoolEval-SQL and
Active PoolEval-SQL (gpt-5-mini judge on the most ambiguous items) and report how the
ranking moves toward the true execution accuracy.

    python -m zoo.run_multi --datasets bird bird_minidev sqlflow --budget 12
    python -m zoo.run_multi --datasets bird --budget 12 --mock     # oracle judge, no API
    python -m zoo.run_multi --datasets bird --n-target 6 --n-source 6 --mock  # smoke

Per-dataset artifacts: poolrun_<ds>.npz, preds_<ds>_dev/src.json; results in
results/zoo_multi_<ds>.json and a combined results/zoo_multi_summary.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import (Config, PoolEval, ActivePoolEval, ActiveConfig, metrics,  # noqa: E402
                      SimulatedJudge)
from pooleval.active import abstention                                          # noqa: E402
from zoo.config import ZooConfig, ARTIFACT_ROOT                                # noqa: E402
from zoo.datasets import load_split                                            # noqa: E402
from zoo.generate import generate_predictions                                  # noqa: E402
from zoo.build import build_poolrun                                            # noqa: E402
from zoo.judge import RealJudge                                                # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


def build_or_load(ds, zcfg, seed, force=False):
    """Generate predictions + build the PoolRun for one dataset (cached per dataset)."""
    npz = os.path.join(ARTIFACT_ROOT, f"poolrun_{ds}.npz")
    members = zcfg.manifest
    names = [m.name for m in members]
    dev = load_split(ds, "dev", zcfg.n_target, seed=seed)
    src = load_split(ds, "source", zcfg.n_source, seed=seed)
    print(f"[{ds}] dev={len(dev)} source={len(src)} over "
          f"{len(set(x['db_id'] for x in dev))} databases")
    preds_dev = generate_predictions(members, dev, zcfg, tag=f"{ds}_dev")
    preds_src = generate_predictions(members, src, zcfg, tag=f"{ds}_src")
    run, _ = build_poolrun(dev, names, zcfg.group_ids(), preds_dev, preds_src, src,
                           timeout=zcfg.exec_timeout)
    np.savez(npz, true_class=run.true_class, true_acc=run.true_acc, group=run.group,
             prior=run.prior, prior_sigma=run.prior_sigma,
             verifier_guess=run.verifier_guess, names=np.array(names))
    return run, names, dev, preds_dev


def run_dataset(ds, budget, strategy, seed, mock, model, rounds, zcfg,
                synthesize=False):
    run, names, dev, preds_dev = build_or_load(ds, zcfg, seed)
    cfg = Config(real_data=True)
    base = PoolEval(cfg).evaluate(run)
    base_m = metrics.all_metrics(base["acc"], run.true_acc)
    base_ab = abstention(base, run, base["obs"])
    print(f"[{ds}] true dev EX: mean={run.true_acc.mean():.3f} "
          f"range=[{run.true_acc.min():.3f},{run.true_acc.max():.3f}]  "
          f"verifier_acc={run.verifier_correct.mean():.3f}")
    print(f"[{ds}] pure PoolEval-SQL: MAE={base_m['MAE']:.2f} Kendall={base_m['Kendall']:.2f} "
          f"Top1={base_m['Top1']:.0f}  Meff={base['Meff']:.1f}  flagged={base_ab['n_flagged']}")

    judge = SimulatedJudge() if mock else RealJudge(
        dev, names, preds_dev, run.true_class, model=model, synthesize=synthesize)
    out = ActivePoolEval(cfg, ActiveConfig(budget=budget, rounds=rounds,
                         strategy=strategy), judge=judge).run(run)
    act_m = metrics.all_metrics(out["acc"], run.true_acc)
    log = getattr(judge, "log", [])
    # mock SimulatedJudge has no 'verdict'; RealJudge logs verdict in {choice,sql,none}
    nnone = sum(1 for e in log if e.get("verdict") == "none")
    nsql = sum(1 for e in log if e.get("verdict") == "sql")
    mode = "synth" if synthesize else ("mock" if mock else model)
    print(f"[{ds}] active ({mode}, {out['judge_calls']} calls, {nnone} 'none'"
          f"{f', {nsql} synth-SQL' if synthesize else ''}): MAE={act_m['MAE']:.2f} "
          f"Kendall={act_m['Kendall']:.2f} Top1={act_m['Top1']:.0f}")

    res = dict(dataset=ds, M=int(run.M), N=int(run.N),
               true_acc_mean=float(run.true_acc.mean()),
               verifier_acc=float(run.verifier_correct.mean()),
               Meff=float(base["Meff"]), abst_flagged=int(base_ab["n_flagged"]),
               budget=int(budget), strategy=strategy, mock=bool(mock),
               model=(None if mock else model), judge_calls=int(out["judge_calls"]),
               judge_none=int(nnone), judge_synth=int(nsql), synthesize=bool(synthesize),
               pure=base_m, active=act_m, judge_log=log)
    os.makedirs(RESULTS, exist_ok=True)
    suffix = "_synth" if synthesize else ""
    with open(os.path.join(RESULTS, f"zoo_multi_{ds}{suffix}.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    return res


def main(datasets, budget=12, strategy="hybrid_submodular", seed=0, mock=False,
         model="gpt-5-mini", rounds=4, n_target=150, n_source=120, synthesize=False):
    zcfg = ZooConfig(n_target=n_target, n_source=n_source, seed=seed)
    summary = []
    for ds in datasets:
        print("\n" + "=" * 70 + f"\n  DATASET: {ds}\n" + "=" * 70)
        try:
            summary.append(run_dataset(ds, budget, strategy, seed, mock, model,
                                       rounds, zcfg, synthesize=synthesize))
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
            print(f"[{ds}] FAILED: {repr(e)[:120]}")

    print("\n" + "=" * 78)
    print(f"{'dataset':14s}{'N':>5s}{'trueEX':>8s}{'pure MAE':>10s}{'act MAE':>9s}"
          f"{'pure Ken':>10s}{'act Ken':>9s}{'none':>6s}")
    for r in summary:
        print(f"{r['dataset']:14s}{r['N']:>5d}{r['true_acc_mean']:>8.2f}"
              f"{r['pure']['MAE']:>10.2f}{r['active']['MAE']:>9.2f}"
              f"{r['pure']['Kendall']:>10.2f}{r['active']['Kendall']:>9.2f}"
              f"{r.get('judge_none',0):>6d}")
    with open(os.path.join(RESULTS, "zoo_multi_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[saved] {os.path.join(RESULTS, 'zoo_multi_summary.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bird", "bird_minidev", "sqlflow"])
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--strategy", default="hybrid_submodular")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-target", type=int, default=150)
    ap.add_argument("--n-source", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--synthesize", action="store_true",
                    help="judge may WRITE a correct SQL when no candidate is right")
    a = ap.parse_args()
    main(a.datasets, budget=a.budget, strategy=a.strategy, seed=a.seed, mock=a.mock,
         model=a.model, rounds=a.rounds, n_target=a.n_target, n_source=a.n_source,
         synthesize=a.synthesize)
