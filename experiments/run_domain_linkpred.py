"""RQ-D8: does the pool machinery transfer to LINK PREDICTION (K=2)?

Six cross-graph transfers over ACMv9 / Citationv1 / DBLPv7, pool = {GCN, SAGE,
GAT, GIN, MLP} x 3 seeds with a dot-product decoder, no target supervision.

This task is the K=2 boundary case of the collision argument. When there are only
two possible answers, a wrong model and a wrong pseudo-label MUST coincide, so
gamma = 1 is exact rather than approximate and the collision correction should
buy nothing at all. Everywhere else in this repo gamma = 1 was wrong; here it is
right, and that is the prediction being tested.
"""
import argparse, json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._domain_eval import (score_pool, print_rows,   # noqa: E402
                                      summarize, print_gamma)
from pooleval.domains.linkpred import build_pool                # noqa: E402

TRANSFERS = [("A", "C"), ("A", "D"), ("C", "A"), ("C", "D"), ("D", "A"), ("D", "C")]


def get_pool(src, dst, seeds, epochs, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"link_{src}{dst}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    pool = build_pool(src, dst, seeds=seeds, epochs=epochs)
    np.savez(cache, **{k: v for k, v in pool.items() if k not in ("src", "dst")})
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/pooleval_domains/pools"))
    ap.add_argument("--out", default="results/domain_linkpred.json")
    a = ap.parse_args()

    allrows, per_transfer, gammas = {}, {}, {}
    for src, dst in TRANSFERS:
        print(f"\n=== {src} -> {dst} ===", flush=True)
        pool = get_pool(src, dst, tuple(range(a.seeds)), a.epochs, a.cache)
        rows, truth, diag = score_pool(pool, extra_variants={
            "PoolEval (learned verif.)": dict(verifier_mode="learned"),
            "PoolEval (no anchors)": dict(use_verifier=False, use_prior=False)})
        per_transfer[f"{src}->{dst}"] = rows
        gammas[f"{src}->{dst}"] = diag
        print(f"  true target acc: {truth.min():.3f}-{truth.max():.3f} "
              f"(mean {truth.mean():.3f}), N={len(pool['gold'])}, K=2")
        print_gamma(diag)
        print_rows(rows)
        for k, v in rows.items():
            allrows.setdefault(k, []).append(v)

    summary = summarize(allrows, "MEAN OVER 6 LINK-PREDICTION TRANSFERS")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(per_transfer=per_transfer, summary=summary, gamma=gammas),
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
