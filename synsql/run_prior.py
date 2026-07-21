"""RETRIEVAL-ALIGNED PRIORS from SynSQL-2.5M.

    Split SynSQL-2.5M into ~1K subsets -> for an UNLABELED target set, retrieve the
    top-k most aligned subsets -> measure the pool on those subsets -> use that as the
    seen prior.

Why this matters for the paper: PoolEval-SQL needs a prior to fix the gauge, and so
far that prior was measured on a labeled split OF THE TARGET BENCHMARK -- which a
deployed operator does not have. A retrieval-aligned SynSQL prior needs zero target
labels, and the probes are amortized across every target.

    python -m synsql.run_prior --dry                     # retrieval analysis, no API
    python -m synsql.run_prior --partition db --probe    # spend the generation budget
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import Config, PoolEval, metrics                       # noqa: E402
from pooleval.data.simulator import PoolRun                          # noqa: E402
from zoo.config import ZooConfig, ARTIFACT_ROOT                      # noqa: E402
from zoo.datasets import load_split                                  # noqa: E402
from synsql.config import (N_CANDIDATES, N_PROBE, TOP_K, SEED, RESULTS)  # noqa: E402
from synsql.subsets import load_built                                # noqa: E402
from synsql.retrieve import SubsetRetriever                          # noqa: E402
from synsql.prior import (probe_subsets, prior_from_probes,           # noqa: E402
                          prior_soft)

TARGETS = ["spider", "bird", "bird_minidev", "sqlflow", "spider2local"]


# --------------------------------------------------------------------------- #
#  candidate selection: a DIVERSE cover of the corpus                          #
# --------------------------------------------------------------------------- #
def pick_candidates(mu, n, seed=SEED, mode="medoid"):
    """Choose which subsets to spend the probing budget on.

    IMPORTANT: candidates are chosen from the CORPUS ONLY -- the targets are never
    consulted. This is the honest deployment protocol: probe a diverse cover once,
    offline, then serve any future unlabeled target by retrieval alone.

    medoid : k-means over subset centroids, take each cluster's medoid -> a
             representative cover (what we use).
    kcenter: farthest-point traversal -> maximal spread, but biased to outliers.
    """
    if mode == "kcenter":
        rng = np.random.default_rng(seed)
        picked = [int(rng.integers(len(mu)))]
        d = 1.0 - mu @ mu[picked[0]]
        for _ in range(n - 1):
            picked.append(int(np.argmax(d)))
            d = np.minimum(d, 1.0 - mu @ mu[picked[-1]])
        return picked
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=n, random_state=seed, n_init=5).fit(mu)
    picked = []
    for c in range(n):
        idx = np.flatnonzero(lab.labels_ == c)
        if len(idx) == 0:
            continue
        picked.append(int(idx[np.argmax(mu[idx] @ lab.cluster_centers_[c])]))
    return sorted(set(picked))


# --------------------------------------------------------------------------- #
#  downstream: plug a prior into the real PoolRun and re-run PoolEval-SQL      #
# --------------------------------------------------------------------------- #
def load_run(ds):
    p = os.path.join(ARTIFACT_ROOT, f"poolrun_{ds}.npz")
    d = np.load(p, allow_pickle=True)
    N = d["true_class"].shape[1]
    return PoolRun(d["true_class"], d["true_acc"], d["group"], d["prior"],
                   d["prior_sigma"], np.zeros(N), np.zeros(N), d["verifier_guess"],
                   d["verifier_guess"] == 0, d["true_class"].shape[0], N,
                   int(np.asarray(d["group"]).max()) + 1), list(d["names"])


def with_prior(run, prior, sigma):
    return PoolRun(run.true_class, run.true_acc, run.group, np.asarray(prior),
                   np.asarray(sigma), run.b, run.phi, run.verifier_guess,
                   run.verifier_correct, run.M, run.N, run.n_groups)


def kendall(a, b):
    from scipy.stats import kendalltau
    t = kendalltau(a, b).correlation
    return 0.0 if np.isnan(t) else float(t)


def score_prior(run, prior, sigma):
    """Quality of the prior itself + of PoolEval-SQL run with it.

    Prior error is decomposed into LEVEL and SHAPE, because they have different
    causes and different fixes:
      prior_bias    signed mean offset -- a corpus-difficulty gap (SynSQL is harder
                    than Spider), i.e. the gauge. No label-free prior can close it.
      prior_mae_c   MAE after removing that offset = how well the prior captures the
                    RELATIVE ability of the pool, which is what retrieval can fix.
      prior_kendall rank agreement with true EX (level-invariant).
    """
    p = np.asarray(prior, dtype=float)
    bias = float(np.mean(p - run.true_acc))
    pr = dict(prior_mae=float(np.mean(np.abs(p - run.true_acc)) * 100),
              prior_bias=bias * 100,
              prior_mae_c=float(np.mean(np.abs(p - bias - run.true_acc)) * 100),
              prior_kendall=kendall(prior, run.true_acc))
    out = PoolEval(Config(real_data=True)).evaluate(with_prior(run, prior, sigma))
    m = metrics.all_metrics(out["acc"], run.true_acc)
    pr.update({f"pe_{k}": float(v) for k, v in m.items()})
    pr["Meff"] = float(out["Meff"])
    return pr


# --------------------------------------------------------------------------- #
def main(partition="db", probe=False, k=TOP_K, n_cand=N_CANDIDATES,
         n_probe=N_PROBE, n_target=150, seed=SEED, targets=None, workers=12):
    targets = targets or TARGETS
    recs, parts = load_built()
    print(f"[data] {len(recs):,} SynSQL records; partitions: "
          + ", ".join(f"{k2}={len(v)}" for k2, v in parts.items()))

    # ---------- 1. load targets ----------
    tgt_items = {}
    for ds in targets:
        try:
            tgt_items[ds] = load_split(ds, "dev", n_target, seed=seed)
        except Exception as e:                                   # noqa: BLE001
            print(f"[skip] {ds}: {repr(e)[:90]}")
    print(f"[data] targets loaded: {list(tgt_items)}")

    # ---------- 2. FREE analysis: is any partitioning retrievable? ----------
    spread = {}
    retrievers = {}
    shared = {}   # the TF-IDF space is the same for all partitionings -> fit once
    for pname, subs in parts.items():
        r = SubsetRetriever(recs, subs, **shared)
        shared = dict(vec=r.vec, X=r.X)
        retrievers[pname] = r
        spread[pname] = {}
        for ds, items in tgt_items.items():
            cos, mmd = r.distances(items)
            spread[pname][ds] = dict(
                min=float(cos.min()), max=float(cos.max()), mean=float(cos.mean()),
                std=float(cos.std()),
                # separability: how many std devs the best subset is from the mean
                z_best=float((cos.mean() - cos.min()) / (cos.std() + 1e-12)))
        print(f"[spread] {pname:7s} " + "  ".join(
            f"{ds}: d=[{spread[pname][ds]['min']:.3f},{spread[pname][ds]['max']:.3f}]"
            f" sd={spread[pname][ds]['std']:.4f}" for ds in tgt_items))

    ret = retrievers[partition]
    subs = parts[partition]
    cands = pick_candidates(ret.mu, n_cand, seed=seed)
    print(f"[cand] {n_cand} diverse candidate subsets from '{partition}': {cands}")

    out = dict(partition=partition, k=k, n_cand=n_cand, n_probe=n_probe,
               candidates=cands, spread=spread, targets={})
    if not probe:
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, "synsql_retrieval_dry.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {RESULTS}/synsql_retrieval_dry.json  (dry run, no API spend)")
        return out

    # ---------- 3. pay once: probe the candidate subsets ----------
    zcfg = ZooConfig(seed=seed)
    store = probe_subsets(recs, subs, cands, zcfg, n_probe=n_probe, seed=seed,
                          workers=workers)
    names = [m.name for m in zcfg.manifest]

    # ---------- 4. per target: retrieve, build priors, score ----------
    rng = np.random.default_rng(seed)
    for ds, items in tgt_items.items():
        run, run_names = load_run(ds)
        if [str(x) for x in run_names] != names:
            print(f"[warn] {ds}: member mismatch, skipping")
            continue
        cos, _ = ret.distances(items)
        d_c = cos[cands]
        order = np.argsort(d_c)
        top = [cands[i] for i in order[:k]]
        far = [cands[i] for i in order[-k:]]
        rnd = [cands[i] for i in rng.permutation(len(cands))[:k]]

        # the scientific core: does DISTANCE predict PRIOR ERROR across candidates?
        errs, ds_list = [], []
        for s in cands:
            a, sg, n = prior_from_probes(store, [s])
            errs.append(float(np.mean(np.abs(a - run.true_acc)) * 100))
            ds_list.append(float(cos[s]))
        rho = float(np.corrcoef(ds_list, errs)[0, 1])
        # ORACLE: the k subsets a perfect retriever would pick (chosen by TRUE prior
        # error, which needs target labels). Upper bound on what retrieval can buy.
        orc = [cands[i] for i in np.argsort(errs)[:k]]

        conds = {"insplit (target-labeled)": (run.prior, run.prior_sigma),
                 "synsql-topk": prior_from_probes(store, top)[:2],
                 "synsql-soft": prior_soft(store, cands, d_c)[:2],
                 "synsql-random": prior_from_probes(store, rnd)[:2],
                 "synsql-far": prior_from_probes(store, far)[:2],
                 "synsql-all": prior_from_probes(store, cands)[:2],
                 "synsql-oracle (bound)": prior_from_probes(store, orc)[:2]}
        # level-corrected oracle: retrieved SHAPE, gauge handed to it for free.
        # Isolates "is the level offset the whole problem?" from "is the shape wrong?"
        pt, st = conds["synsql-topk"]
        conds["synsql-topk +true level"] = (
            np.asarray(pt) - np.mean(np.asarray(pt) - run.true_acc), st)
        res = {c: score_prior(run, p, s) for c, (p, s) in conds.items()}
        out["targets"][ds] = dict(
            true_acc=run.true_acc.tolist(), M=int(run.M), N=int(run.N),
            top=top, far=far, rnd=rnd, oracle=orc,
            dist_top=[float(cos[s]) for s in top],
            dist_far=[float(cos[s]) for s in far],
            per_cand=dict(dist=ds_list, prior_mae=errs), rho_dist_vs_err=rho,
            conditions=res)
        print(f"\n=== {ds} (true EX mean {run.true_acc.mean():.3f}) ===")
        print(f"  corr(distance, prior MAE) over {len(cands)} subsets = {rho:+.3f}")
        print(f"  {'condition':26s}{'priorMAE':>9s}{'bias':>8s}{'MAE_c':>8s}"
              f"{'priorKen':>9s}{'PE MAE':>9s}{'PE Ken':>8s}{'PE Top1':>8s}")
        for c, r in res.items():
            print(f"  {c:26s}{r['prior_mae']:>9.2f}{r['prior_bias']:>+8.2f}"
                  f"{r['prior_mae_c']:>8.2f}{r['prior_kendall']:>9.2f}"
                  f"{r['pe_MAE']:>9.2f}{r['pe_Kendall']:>8.2f}{r['pe_Top1']:>8.2f}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "synsql_prior.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[saved] {RESULTS}/synsql_prior.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="db", choices=["random", "db", "kmeans"])
    ap.add_argument("--probe", action="store_true", help="spend the generation budget")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--n-cand", type=int, default=N_CANDIDATES)
    ap.add_argument("--n-probe", type=int, default=N_PROBE)
    ap.add_argument("--n-target", type=int, default=150)
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    main(partition=a.partition, probe=(a.probe and not a.dry), k=a.k,
         n_cand=a.n_cand, n_probe=a.n_probe, n_target=a.n_target, targets=a.targets,
         workers=a.workers)
