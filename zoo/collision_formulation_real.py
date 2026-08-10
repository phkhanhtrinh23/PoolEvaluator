"""Real-data experiment for the case-3 collision-aware binary formulation.

Collision rates are estimated without target leakage by pooling the other labeled
real-zoo datasets (leave-one-dataset-out). Target train priors remain explicit Beta
anchors. An oracle target-gamma result is reported only as an upper-bound diagnostic.
"""
import argparse
import json
import os

import numpy as np

from pooleval import (CollisionAwareNewFormulationPoolEval, Config,
                      NewFormulationPoolEval, PoolEval, metrics)
from zoo.new_formulation_real import load_run, subset_run, interval


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CLEAN_DATASETS = ["spider", "sqlflow", "bird", "bird_minidev"]


def old_pseudo(run):
    cfg = Config(real_data=True)
    old = PoolEval(cfg).evaluate(run, obs=run.true_class)
    labels = np.asarray([max(p, key=p.get) for p in old["latent_post"]])
    return old, labels


def collision_statistics(runs, smoothing=1.0):
    """Source-only group collision rates and pseudo-label accuracy."""
    G = max(run.n_groups for run in runs)
    hits = np.zeros(G)
    eligible = np.zeros(G)
    pseudo_correct = total_items = 0
    for run in runs:
        _, labels = old_pseudo(run)
        pseudo_wrong = labels != 0
        pseudo_correct += int((labels == 0).sum())
        total_items += run.N
        for m in range(run.M):
            g = run.group[m]
            valid = (run.true_class[m] != 0) & pseudo_wrong
            eligible[g] += valid.sum()
            hits[g] += (valid & (run.true_class[m] == labels)).sum()
    gamma = (hits + smoothing) / (eligible + 2.0 * smoothing)
    beta = (pseudo_correct + smoothing) / (total_items + 2.0 * smoothing)
    return dict(gamma=gamma, beta=float(beta), hits=hits, eligible=eligible,
                items=int(total_items))


def alpha_effective_size(run):
    variance = np.asarray(run.prior_sigma) ** 2
    numer = np.asarray(run.prior) * (1.0 - np.asarray(run.prior))
    valid = (variance > 0) & (numer > 0)
    sizes = numer[valid] / variance[valid]
    return float(np.median(sizes)) if sizes.size else 120.0


def evaluate(run, source_stats, oracle_stats=None):
    cfg = Config(real_data=True)
    old, labels = old_pseudo(run)
    plain = NewFormulationPoolEval(cfg).evaluate(
        run, obs=run.true_class, pseudo_out=old
    )
    strength = alpha_effective_size(run)
    unanchored = CollisionAwareNewFormulationPoolEval(
        cfg, source_stats["gamma"], strength,
        beta_init=source_stats["beta"],
    ).evaluate(run, obs=run.true_class, pseudo_out=old)
    corrected = CollisionAwareNewFormulationPoolEval(
        cfg, source_stats["gamma"], strength,
        beta_init=source_stats["beta"], beta_strength=120.0,
    ).evaluate(run, obs=run.true_class, pseudo_out=old)
    result = dict(
        prior=metrics.all_metrics(run.prior, run.true_acc),
        old=metrics.all_metrics(old["acc"], run.true_acc),
        plain=metrics.all_metrics(plain["acc"], run.true_acc),
        unanchored=metrics.all_metrics(unanchored["acc"], run.true_acc),
        corrected=metrics.all_metrics(corrected["acc"], run.true_acc),
        pseudo_accuracy=float((labels == 0).mean()),
        plain_beta=float(plain["beta"]),
        unanchored_beta=float(unanchored["beta"]),
        corrected_beta=float(corrected["beta"]),
        corrected_iterations=int(corrected["n_iters"]),
        alpha_strength=strength,
        true_acc=run.true_acc.copy(), prior_acc=run.prior.copy(),
        old_acc=old["acc"].copy(), plain_acc=plain["acc"].copy(),
        unanchored_acc=unanchored["acc"].copy(),
        corrected_acc=corrected["acc"].copy(),
    )
    if oracle_stats is not None:
        oracle = CollisionAwareNewFormulationPoolEval(
            cfg, oracle_stats["gamma"], strength,
            beta_init=oracle_stats["beta"], beta_strength=120.0,
        ).evaluate(run, obs=run.true_class, pseudo_out=old)
        result["oracle_gamma"] = metrics.all_metrics(oracle["acc"], run.true_acc)
        result["oracle_acc"] = oracle["acc"].copy()
        result["oracle_beta"] = float(oracle["beta"])
    return result


def bootstrap(run, source_stats, count, seed):
    rng = np.random.default_rng(seed)
    rows = [evaluate(subset_run(run, rng.integers(0, run.N, run.N)), source_stats)
            for _ in range(count)]
    out = {}
    for method in ["prior", "old", "plain", "unanchored", "corrected"]:
        out[method] = {key: interval([r[method][key] for r in rows])
                       for key in ["MAE", "Flip", "Kendall", "Top1", "Top3"]}
    out["corrected_minus_old"] = {
        key: interval([r["corrected"][key] - r["old"][key] for r in rows])
        for key in ["MAE", "Flip", "Kendall", "Top1", "Top3"]
    }
    return out


def serializable_point(point):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in point.items()}


def main(bootstrap_count=500, seed=20260810):
    runs = {ds: load_run(ds)[0] for ds in CLEAN_DATASETS}
    payload = {}
    print(f"{'dataset':14s}{'prior':>9s}{'old':>9s}{'plain':>9s}"
          f"{'case3':>9s}{'oldKen':>9s}{'case3Ken':>10s}{'beta':>8s}")
    for offset, dataset in enumerate(CLEAN_DATASETS):
        run = runs[dataset]
        sources = [other for ds, other in runs.items() if ds != dataset]
        source_stats = collision_statistics(sources)
        oracle_stats = collision_statistics([run])
        point = evaluate(run, source_stats, oracle_stats)
        boot = bootstrap(run, source_stats, bootstrap_count, seed + offset)
        payload[dataset] = dict(
            source_datasets=[ds for ds in CLEAN_DATASETS if ds != dataset],
            source_gamma=source_stats["gamma"].tolist(),
            source_gamma_hits=source_stats["hits"].tolist(),
            source_gamma_eligible=source_stats["eligible"].tolist(),
            source_beta=source_stats["beta"],
            oracle_target_gamma=oracle_stats["gamma"].tolist(),
            oracle_target_beta=oracle_stats["beta"],
            point=serializable_point(point), bootstrap=boot,
        )
        print(f"{dataset:14s}{point['prior']['MAE']:9.2f}{point['old']['MAE']:9.2f}"
              f"{point['plain']['MAE']:9.2f}{point['corrected']['MAE']:9.2f}"
              f"{point['old']['Kendall']:9.2f}{point['corrected']['Kendall']:10.2f}"
              f"{point['corrected_beta']:8.3f}")
    payload["metadata"] = dict(
        protocol="leave-one-dataset-out gamma; target train prior anchors alpha",
        bootstrap_count=bootstrap_count, bootstrap_seed=seed,
        oracle_gamma_is_test_leaking_diagnostic=True,
    )
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "collision_formulation_real.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"[saved] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", dest="bootstrap_count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260810)
    main(**vars(parser.parse_args()))
