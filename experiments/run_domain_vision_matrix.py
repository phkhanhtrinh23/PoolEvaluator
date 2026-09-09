"""RQ-D6: the image port with the SOURCE varied, not just the target.

The original image port trained on MNIST only, so every conclusion was
conditional on one training distribution -- and MNIST -> SVHN turned out to be
degenerate (models collapse to one class, accuracy below chance), which then
dominated the two-shift mean.

This runs the full directed transfer matrix over {MNIST, USPS, SVHN}: six
transfers, the exact analogue of the six citation-graph transfers in
`run_domain_graph.py`. Models are trained ONCE per source and reused across that
source's two targets, so the pool is literally identical across targets.
"""
import argparse, json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._domain_eval import (score_pool, print_rows,   # noqa: E402
                                      summarize, print_gamma)
from pooleval.domains.vision import build_pools_from_source     # noqa: E402

DATASETS = ["mnist", "usps", "svhn"]
TRANSFERS = [(s, d) for s in DATASETS for d in DATASETS if s != d]


def get_pools(src, dsts, seeds, epochs, cache_dir, n_target):
    """-> {dst: pool}, training once per source and caching per (src, dst)."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = {d: os.path.join(cache_dir, f"vmat_{src}_{d}.npz") for d in dsts}
    if all(os.path.exists(p) for p in paths.values()):
        return {d: {k: z[k] for k in z.files}
                for d, z in ((d, np.load(p, allow_pickle=True)) for d, p in paths.items())}
    pools = build_pools_from_source(src, dsts, seeds=seeds, epochs=epochs,
                                    n_target=n_target)
    for d, pool in pools.items():
        np.savez(paths[d], **{k: v for k, v in pool.items() if k not in ("src", "dst")})
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--n-target", type=int, default=5000)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/pooleval_domains/pools"))
    ap.add_argument("--out", default="results/domain_vision_matrix.json")
    a = ap.parse_args()

    allrows, healthy_rows, per_transfer, gammas, health = {}, {}, {}, {}, {}
    for src in DATASETS:
        dsts = [d for d in DATASETS if d != src]
        pools = get_pools(src, dsts, tuple(range(a.seeds)), a.epochs, a.cache, a.n_target)
        for dst in dsts:
            pool = pools[dst]
            print(f"\n=== {src} -> {dst} ===", flush=True)
            rows, truth, diag = score_pool(pool, extra_variants={
                "PoolEval (learned verif.)": dict(verifier_mode="learned"),
                "PoolEval (no anchors)": dict(use_verifier=False, use_prior=False)})
            # A transfer is only usable if the pool is not collapsed: report the
            # share of each model's predictions on its single most-used class.
            pred, K = np.asarray(pool["pred"]), int(np.asarray(pool["n_classes"]))
            share = np.array([(np.bincount(p, minlength=K) / len(p)).max() for p in pred])
            ok = bool(truth.mean() > 1.5 / K and (share > 0.5).mean() < 0.5)
            health[f"{src}->{dst}"] = dict(mean_acc=float(truth.mean()),
                                           acc_min=float(truth.min()), acc_max=float(truth.max()),
                                           mean_top_class_share=float(share.mean()),
                                           n_collapsed=int((share > 0.5).sum()),
                                           M=int(pred.shape[0]), usable=ok)
            per_transfer[f"{src}->{dst}"] = rows
            gammas[f"{src}->{dst}"] = diag
            print(f"  true target acc: {truth.min():.3f}-{truth.max():.3f} "
                  f"(mean {truth.mean():.3f}), N={len(pool['gold'])}  |  collapse: "
                  f"{int((share>0.5).sum())}/{len(share)} models >50% on one class"
                  f"  -> {'USABLE' if ok else 'DEGENERATE'}")
            print_gamma(diag)
            print_rows(rows)
            for k, v in rows.items():
                allrows.setdefault(k, []).append(v)
                if ok:
                    healthy_rows.setdefault(k, []).append(v)

    summary = summarize(allrows, f"MEAN OVER ALL {len(per_transfer)} TRANSFERS")
    n_ok = sum(h["usable"] for h in health.values())
    summary_ok = summarize(healthy_rows, f"MEAN OVER {n_ok} NON-DEGENERATE TRANSFERS")
    print("\n=== pool health ===")
    for k, h in health.items():
        print(f"  {k:14s} acc {h['acc_min']:.3f}-{h['acc_max']:.3f} "
              f"(mean {h['mean_acc']:.3f})  collapsed {h['n_collapsed']}/{h['M']}  "
              f"{'usable' if h['usable'] else 'DEGENERATE'}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(per_transfer=per_transfer, summary=summary,
                   summary_non_degenerate=summary_ok, gamma=gammas, health=health),
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
