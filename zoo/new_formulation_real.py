"""Evaluate the closed-form formulation on saved real Text-to-SQL zoo artifacts.

No model generation or API access is needed. Source-derived priors remain fixed while
held-out target items are evaluated and bootstrapped.

  python -m zoo.new_formulation_real --bootstrap 500
"""
import argparse
import json
import os

import numpy as np

from pooleval import Config, NewFormulationPoolEval, PoolEval, metrics
from pooleval.data.simulator import PoolRun
from zoo.config import ARTIFACT_ROOT


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DEFAULT_DATASETS = ["spider", "sqlflow", "bird", "bird_minidev", "spider2local"]


def load_run(dataset):
    path = os.path.join(ARTIFACT_ROOT, f"poolrun_{dataset}.npz")
    data = np.load(path, allow_pickle=True)
    tc = data["true_class"]
    M, N = tc.shape
    group = data["group"]
    verifier = data["verifier_guess"]
    run = PoolRun(
        true_class=tc,
        true_acc=(tc == 0).mean(axis=1),
        group=group,
        prior=data["prior"],
        prior_sigma=data["prior_sigma"],
        b=np.zeros(N),
        phi=np.zeros(N),
        verifier_guess=verifier,
        verifier_correct=(verifier == 0),
        M=M,
        N=N,
        n_groups=int(group.max()) + 1,
    )
    names = data["names"].astype(str) if "names" in data else np.arange(M).astype(str)
    return run, names


def subset_run(run, indices):
    indices = np.asarray(indices, dtype=int)
    tc = run.true_class[:, indices]
    return PoolRun(
        true_class=tc,
        true_acc=(tc == 0).mean(axis=1),
        group=run.group,
        prior=run.prior,
        prior_sigma=run.prior_sigma,
        b=np.zeros(len(indices)),
        phi=np.zeros(len(indices)),
        verifier_guess=run.verifier_guess[indices],
        verifier_correct=run.verifier_correct[indices],
        M=run.M,
        N=len(indices),
        n_groups=run.n_groups,
    )


def evaluate(run):
    cfg = Config(real_data=True)
    old = PoolEval(cfg).evaluate(run, obs=run.true_class)
    new = NewFormulationPoolEval(cfg).evaluate(
        run, obs=run.true_class, pseudo_out=old
    )
    pseudo_correct = new["pseudo_label"] == 0
    return dict(
        prior=metrics.all_metrics(run.prior, run.true_acc),
        old=metrics.all_metrics(old["acc"], run.true_acc),
        new=metrics.all_metrics(new["acc"], run.true_acc),
        prior_acc=run.prior.copy(),
        old_acc=old["acc"].copy(),
        new_acc=new["acc"].copy(),
        pseudo_accuracy=float(pseudo_correct.mean()),
        beta=float(new["beta"]),
        beta_error=float(abs(new["beta"] - pseudo_correct.mean())),
        iterations=int(new["n_iters"]),
    )


def interval(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), lo=float(np.percentile(values, 2.5)),
                hi=float(np.percentile(values, 97.5)))


def bootstrap(run, count, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        rows.append(evaluate(subset_run(run, rng.integers(0, run.N, run.N))))
    out = {}
    for method in ["prior", "old", "new"]:
        out[method] = {
            key: interval([row[method][key] for row in rows])
            for key in ["MAE", "Flip", "Kendall", "Top1", "Top3"]
        }
    out["paired_delta"] = {
        key: interval([row["new"][key] - row["old"][key] for row in rows])
        for key in ["MAE", "Flip", "Kendall", "Top1", "Top3"]
    }
    out["diagnostics"] = {
        key: interval([row[key] for row in rows])
        for key in ["pseudo_accuracy", "beta", "beta_error", "iterations"]
    }
    return out


def source_provenance(dataset):
    if dataset == "spider":
        return "Spider train (120) -> Spider dev target (150); database-disjoint"
    if dataset == "bird":
        return "BIRD train (120) -> BIRD dev target (150)"
    if dataset in {"sqlflow", "bird_minidev"}:
        return "deterministic disjoint labeled 30% source slice (120) -> held-out target (150)"
    return "OVERLAP WARNING: tiny-set fallback source overlaps the 24-item target"


def main(datasets, bootstrap_count=500, seed=20260810):
    payload = {}
    print(f"{'dataset':14s}{'N':>5s}{'trueEX':>8s}{'priorMAE':>10s}"
          f"{'oldMAE':>9s}{'newMAE':>9s}{'oldKen':>9s}{'newKen':>9s}"
          f"{'pseudo':>9s}{'beta':>8s}")
    for offset, dataset in enumerate(datasets):
        run, names = load_run(dataset)
        point = evaluate(run)
        boot = bootstrap(run, bootstrap_count, seed + offset)
        contaminated = dataset == "spider2local"
        payload[dataset] = dict(
            M=run.M, N=run.N, names=names.tolist(),
            source_provenance=source_provenance(dataset),
            source_target_separated=(not contaminated),
            true_acc=run.true_acc.tolist(), prior=run.prior.tolist(),
            prior_acc=point.pop("prior_acc").tolist(),
            old_acc=point.pop("old_acc").tolist(),
            new_acc=point.pop("new_acc").tolist(),
            point=point, bootstrap=boot,
        )
        print(f"{dataset:14s}{run.N:5d}{run.true_acc.mean():8.3f}"
              f"{point['prior']['MAE']:10.2f}{point['old']['MAE']:9.2f}"
              f"{point['new']['MAE']:9.2f}{point['old']['Kendall']:9.2f}"
              f"{point['new']['Kendall']:9.2f}{point['pseudo_accuracy']:9.3f}"
              f"{point['beta']:8.3f}{' *' if contaminated else ''}")
    payload["metadata"] = dict(
        bootstrap_count=bootstrap_count,
        bootstrap_seed=seed,
        note="Spider2local is excluded from clean source/target conclusions due overlap.",
    )
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "new_formulation_real.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\n* source/target overlap warning\n[saved] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--bootstrap", dest="bootstrap_count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260810)
    main(**vars(parser.parse_args()))
