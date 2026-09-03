"""RQ-D9: the LARGE-K port -- knowledge-graph completion.

FB15k-237 (14,541 entities) and WN18RR (40,943). Pool = {TransE, DistMult,
ComplEx, RotatE} x 3 seeds, predicting the tail of (head, relation, ?) under the
filtered protocol. Accuracy is filtered Hits@1; the prior is Hits@1 on the
labelled validation split.

This is the regime section 7 of docs/multiclass_ds.md could only SIMULATE. The
simulation predicted full Dawid--Skene collapses once it falls below roughly one
observation per free parameter, somewhere between K=50 and K=100. Here K is 14541
and full DS needs ~20 GB of confusion matrices, so it is not merely worse -- it
cannot be constructed, and `multiclass_ds.full_ds` says so with the arithmetic.
One-coin DS survives on a sparse E-step because its parameter count is
independent of K.
"""
import argparse, json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._domain_eval import (score_pool, print_rows,   # noqa: E402
                                      summarize, print_gamma)
from pooleval.domains.kgc import build_pool                     # noqa: E402
from pooleval.domains.multiclass_ds import dense_cost           # noqa: E402

DATASETS = ["fb15k237", "wn18rr"]


def get_pool(name, seeds, epochs, n_target, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"kgc_{name}_{n_target}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    pool = build_pool(name, seeds=seeds, epochs=epochs, n_target=n_target)
    np.savez(cache, **{k: v for k, v in pool.items() if k != "dataset"})
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-target", type=int, default=3000)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/pooleval_domains/pools"))
    ap.add_argument("--out", default="results/domain_kgc.json")
    a = ap.parse_args()

    allrows, per_ds, gammas, feas = {}, {}, {}, {}
    for name in DATASETS:
        print(f"\n=== {name} ===", flush=True)
        pool = get_pool(name, tuple(range(a.seeds)), a.epochs, a.n_target, a.cache)
        K, M = int(np.asarray(pool["n_classes"])), pool["pred"].shape[0]
        gb = dense_cost(M, K) / 1024 ** 3
        feas[name] = dict(K=K, M=M, full_ds_gb=gb,
                          full_ds_params_per_model=K * (K - 1))
        print(f"  K = {K} entities, M = {M} models")
        print(f"  full-DS confusion matrices would need {gb:.1f} GB "
              f"({K*(K-1):,} free parameters per model) -> not constructible")

        rows, truth, diag = score_pool(pool, extra_variants={
            "PoolEval (learned verif.)": dict(verifier_mode="learned"),
            "PoolEval (no anchors)": dict(use_verifier=False, use_prior=False)})
        per_ds[name] = rows
        gammas[name] = diag
        print(f"  true test H@1: {truth.min():.3f}-{truth.max():.3f} "
              f"(mean {truth.mean():.3f}), N={len(pool['gold'])}")
        print_gamma(diag)
        print_rows(rows)
        for k, v in rows.items():
            allrows.setdefault(k, []).append(v)

    summary = summarize(allrows, f"MEAN OVER {len(DATASETS)} KG-COMPLETION DATASETS")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(per_dataset=per_ds, summary=summary, gamma=gammas,
                   feasibility=feas), open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
