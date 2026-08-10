"""Replay saved real LLM-judge verdicts through collision-aware binary EM.

The saved verdicts were acquired by the old active selector. This runner applies
them only after an initial collision-aware fit, replaces the judged pseudo-labels,
rebuilds C, and re-fits the case-3 estimator at increasing budgets.
"""
import argparse
import json
import os

import numpy as np

from pooleval import Config, PoolEval, collision_agreement_em, metrics
from zoo.collision_formulation_real import (CLEAN_DATASETS, alpha_effective_size,
                                            collision_statistics, old_pseudo)
from zoo.new_formulation_real import load_run, subset_run, interval


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
BUDGETS = [0, 3, 6, 9, 12]


def load_judge_log(dataset):
    path = os.path.join(RESULTS, f"zoo_multi_{dataset}.json")
    with open(path) as handle:
        return json.load(handle).get("judge_log", [])


def fit_from_labels(run, labels, stats):
    C = (run.true_class == labels[None, :]).astype(float)
    out = collision_agreement_em(
        C, run.group, stats["gamma"], run.prior, alpha_effective_size(run),
        beta_init=stats["beta"], beta_strength=120.0,
    )
    return out["alpha"], out


def labels_at_budget(base_labels, log, budget):
    labels = np.asarray(base_labels).copy()
    applied = []
    for entry in log[:budget]:
        i = int(entry["item"])
        c = int(entry["class_id"])
        before = int(labels[i])
        labels[i] = c
        applied.append(dict(item=i, before=before, after=c,
                            changed=(before != c),
                            rejected_all=(c not in set(entry.get("candidate_classes", []))
                                          and entry.get("choice") is None
                                          and entry.get("verdict") != "choice")))
    return labels, applied


def judge_diagnostics(run, log):
    correct = changed = rejected = rejected_correct = 0
    for entry in log:
        i = int(entry["item"])
        c = int(entry["class_id"])
        is_none = entry.get("choice") is None and c not in set(run.true_class[:, i])
        ok = (not (run.true_class[:, i] == 0).any()) if is_none else c == 0
        correct += int(ok)
        rejected += int(is_none)
        rejected_correct += int(is_none and ok)
    return dict(n=len(log), correct=correct,
                accuracy=(correct / len(log) if log else 0.0),
                rejected_all=rejected, rejected_all_correct=rejected_correct)


def point_evaluate(run, source_stats, log):
    old, base_labels = old_pseudo(run)
    rows = {}
    changes = {}
    for budget in BUDGETS:
        labels, applied = labels_at_budget(base_labels, log, budget)
        acc, out = fit_from_labels(run, labels, source_stats)
        rows[str(budget)] = dict(metrics=metrics.all_metrics(acc, run.true_acc),
                                 acc=acc, beta=float(out["beta"]),
                                 iterations=int(out["n_iters"]))
        changes[str(budget)] = dict(
            applied=len(applied), changed=sum(x["changed"] for x in applied)
        )
    return rows, changes, judge_diagnostics(run, log)


def bootstrap(run, source_stats, log, count, seed):
    """Item bootstrap conditional on the fixed saved judge verdicts."""
    _, base_labels = old_pseudo(run)
    labels_by_budget = {b: labels_at_budget(base_labels, log, b)[0] for b in BUDGETS}
    rng = np.random.default_rng(seed)
    values = {b: [] for b in BUDGETS}
    for _ in range(count):
        idx = rng.integers(0, run.N, run.N)
        rb = subset_run(run, idx)
        for b in BUDGETS:
            acc, _ = fit_from_labels(rb, labels_by_budget[b][idx], source_stats)
            values[b].append(metrics.all_metrics(acc, rb.true_acc))
    out = {}
    for b in BUDGETS:
        out[str(b)] = {key: interval([row[key] for row in values[b]])
                       for key in ["MAE", "Flip", "Kendall", "Top1", "Top3"]}
    out["delta_12_minus_0"] = {
        key: interval([values[12][k][key] - values[0][k][key]
                       for k in range(count)])
        for key in ["MAE", "Flip", "Kendall", "Top1", "Top3"]
    }
    return out


def serialize(rows):
    for row in rows.values():
        if isinstance(row, dict) and isinstance(row.get("acc"), np.ndarray):
            row["acc"] = row["acc"].tolist()
    return rows


def main(bootstrap_count=400, seed=20260810):
    runs = {ds: load_run(ds)[0] for ds in CLEAN_DATASETS}
    payload = {}
    print(f"{'dataset':14s}{'b0 MAE':>9s}{'b12 MAE':>10s}{'b0 Ken':>9s}"
          f"{'b12 Ken':>10s}{'changed':>9s}{'none':>7s}{'judgeAcc':>10s}")
    for offset, dataset in enumerate(CLEAN_DATASETS):
        source_stats = collision_statistics(
            [run for ds, run in runs.items() if ds != dataset]
        )
        log = load_judge_log(dataset)
        rows, changes, judge = point_evaluate(runs[dataset], source_stats, log)
        boot = bootstrap(runs[dataset], source_stats, log, bootstrap_count,
                         seed + offset)
        payload[dataset] = dict(
            budgets=serialize(rows), changes=changes,
            judge=judge, judge_log=log, bootstrap=boot,
            source_datasets=[ds for ds in CLEAN_DATASETS if ds != dataset],
        )
        b0, b12 = rows["0"]["metrics"], rows["12"]["metrics"]
        print(f"{dataset:14s}{b0['MAE']:9.2f}{b12['MAE']:10.2f}"
              f"{b0['Kendall']:9.2f}{b12['Kendall']:10.2f}"
              f"{changes['12']['changed']:9d}{judge['rejected_all']:7d}"
              f"{judge['accuracy']:10.2f}")
    payload["metadata"] = dict(
        protocol="saved real LLM verdict replay after collision-aware EM",
        selection="original old-method active ambiguity order",
        bootstrap_count=bootstrap_count, bootstrap_seed=seed,
        bootstrap_conditioning="fixed judged item set and verdicts",
    )
    path = os.path.join(RESULTS, "collision_active_real.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"[saved] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", dest="bootstrap_count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260810)
    main(**vars(parser.parse_args()))
