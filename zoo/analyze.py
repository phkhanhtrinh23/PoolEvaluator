"""Post-hoc analyses on the saved real-zoo PoolRuns (no regeneration).

Three studies, all reusing the REAL gpt-5-mini judge decisions logged in
results/zoo_multi_<ds>.json (constraints = {item: judged class}):

  prior     no-prior vs with-prior, for pure and active
  judge     gpt-5-mini's reliability as a verifier, scored against withheld gold
  bootstrap item-level bootstrap CIs on MAE(pure), MAE(active) and their difference
            (is the active improvement statistically real?)

    python -m zoo.analyze --datasets spider sqlflow bird bird_minidev spider2local
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import Config, PoolEval, metrics                     # noqa: E402
from pooleval.data.simulator import PoolRun                        # noqa: E402
from zoo.config import ARTIFACT_ROOT                               # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


def load_run(ds):
    d = np.load(os.path.join(ARTIFACT_ROOT, f"poolrun_{ds}.npz"), allow_pickle=True)
    tc = d["true_class"]; M, N = tc.shape; group = d["group"]; vg = d["verifier_guess"]
    run = PoolRun(true_class=tc, true_acc=d["true_acc"], group=group, prior=d["prior"],
                  prior_sigma=d["prior_sigma"], b=np.zeros(N), phi=np.ones(N),
                  verifier_guess=vg, verifier_correct=(vg == 0), M=M, N=N,
                  n_groups=int(group.max()) + 1)
    return run


def load_constraints(ds):
    """The real judge's decisions: {item_index: judged class id}."""
    d = json.load(open(os.path.join(RESULTS, f"zoo_multi_{ds}.json")))
    return {int(e["item"]): int(e["class_id"]) for e in d.get("judge_log", [])}, d


def _subset(run, idx):
    tc = run.true_class[:, idx]
    return PoolRun(true_class=tc, true_acc=(tc == 0).mean(axis=1), group=run.group,
                  prior=run.prior, prior_sigma=run.prior_sigma, b=np.zeros(len(idx)),
                  phi=np.ones(len(idx)), verifier_guess=run.verifier_guess[idx],
                  verifier_correct=run.verifier_correct[idx], M=run.M, N=len(idx),
                  n_groups=run.n_groups)


# --------------------------------------------------------------------------- #
#  1. Prior ablation                                                           #
# --------------------------------------------------------------------------- #
def prior_ablation(ds):
    run = load_run(ds); cons, _ = load_constraints(ds)
    obs = run.true_class
    out = {}
    for use_prior in (True, False):
        cfg = Config(real_data=True, use_prior=use_prior)
        pe = PoolEval(cfg)
        pure = pe.evaluate(run, obs=obs)["acc"]
        act = pe.evaluate(run, obs=obs, constraints=cons)["acc"]
        tag = "with_prior" if use_prior else "no_prior"
        out[tag] = dict(pure=metrics.all_metrics(pure, run.true_acc),
                        active=metrics.all_metrics(act, run.true_acc))
    return out


# --------------------------------------------------------------------------- #
#  2. Judge reliability vs withheld gold                                       #
# --------------------------------------------------------------------------- #
def judge_reliability(ds):
    run = load_run(ds)
    _, d = load_constraints(ds)
    log = d.get("judge_log", [])
    tc = run.true_class
    n = correct = none_true = none_calls = pick_correct = pick_calls = errors = 0
    gap_items = 0
    for e in log:
        i = int(e["item"])
        if "error" in e:
            errors += 1
            continue
        correct_exists = bool((tc[:, i] == 0).any())   # a pool model is truly right
        gap_items += (not correct_exists)
        # normalize both log formats: old selection logs use 'choice', synth logs
        # use 'verdict' in {choice, sql, none}
        verdict = e.get("verdict")
        if verdict is None:
            verdict = "none" if e.get("choice") is None else "choice"
        n += 1
        if verdict == "none":                           # judge said "none"
            none_calls += 1
            ok = (not correct_exists)                   # right iff truly no model correct
            none_true += ok
            correct += ok
        else:                                           # judge picked / wrote a query
            pick_calls += 1
            ok = (int(e["class_id"]) == 0)              # resolved to the correct class
            pick_correct += ok
            correct += ok
    return dict(n_judged=n, errors=errors,
                judge_accuracy=(correct / n if n else 0.0),
                none_calls=none_calls,
                none_precision=(none_true / none_calls if none_calls else None),
                pick_calls=pick_calls,
                pick_accuracy=(pick_correct / pick_calls if pick_calls else None),
                true_gap_rate=(gap_items / n if n else 0.0))


# --------------------------------------------------------------------------- #
#  3. Item-level bootstrap CIs                                                 #
# --------------------------------------------------------------------------- #
def bootstrap(ds, B=400, seed=0):
    run = load_run(ds); cons, _ = load_constraints(ds)
    cfg = Config(real_data=True)
    pe = PoolEval(cfg)
    rng = np.random.default_rng(seed)
    N = run.N
    mp, ma, dl = [], [], []
    for _ in range(B):
        idx = rng.integers(0, N, N)
        rb = _subset(run, idx); ob = rb.true_class
        cb = {j: cons[idx[j]] for j in range(N) if idx[j] in cons}
        ap = pe.evaluate(rb, obs=ob)["acc"]
        aa = pe.evaluate(rb, obs=ob, constraints=cb)["acc"]
        p, a = metrics.mae(ap, rb.true_acc), metrics.mae(aa, rb.true_acc)
        mp.append(p); ma.append(a); dl.append(a - p)

    def ci(x):
        x = np.array(x)
        return dict(mean=float(x.mean()), lo=float(np.percentile(x, 2.5)),
                    hi=float(np.percentile(x, 97.5)))
    d = np.array(dl)
    return dict(pure_MAE=ci(mp), active_MAE=ci(ma), delta_MAE=ci(dl),
                p_improve=float((d < 0).mean()))   # fraction of resamples where active wins


# --------------------------------------------------------------------------- #
def main(datasets):
    prior_rows, judge_rows, boot_rows = {}, {}, {}
    for ds in datasets:
        print(f"\n=== {ds} ===")
        pr = prior_ablation(ds); prior_rows[ds] = pr
        jr = judge_reliability(ds); judge_rows[ds] = jr
        bs = bootstrap(ds); boot_rows[ds] = bs
        print(f"  prior:  with-prior pure MAE={pr['with_prior']['pure']['MAE']:.2f} "
              f"active MAE={pr['with_prior']['active']['MAE']:.2f} | "
              f"no-prior pure MAE={pr['no_prior']['pure']['MAE']:.2f} "
              f"active MAE={pr['no_prior']['active']['MAE']:.2f}")
        np_ = "n/a" if jr['none_precision'] is None else f"{jr['none_precision']:.2f}"
        pa_ = "n/a" if jr['pick_accuracy'] is None else f"{jr['pick_accuracy']:.2f}"
        print(f"  judge:  accuracy={jr['judge_accuracy']:.2f} over {jr['n_judged']} "
              f"({jr['errors']} API-err) | none-precision={np_} pick-acc={pa_} "
              f"true-gap-rate={jr['true_gap_rate']:.2f}")
        print(f"  boot:   ΔMAE mean={bs['delta_MAE']['mean']:+.2f} "
              f"CI[{bs['delta_MAE']['lo']:+.2f},{bs['delta_MAE']['hi']:+.2f}]  "
              f"P(active better)={bs['p_improve']:.2f}")

    out = dict(prior=prior_rows, judge=judge_rows, bootstrap=boot_rows)
    with open(os.path.join(RESULTS, "zoo_analysis.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[saved] {os.path.join(RESULTS, 'zoo_analysis.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["spider", "sqlflow", "bird", "bird_minidev", "spider2local"])
    main(ap.parse_args().datasets)
