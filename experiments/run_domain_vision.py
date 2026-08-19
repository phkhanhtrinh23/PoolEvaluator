"""RQ-D2: does PoolEval transfer to IMAGE CLASSIFICATION?

MNIST -> {USPS, SVHN} (MetaEvaluator's digit shifts), pool = five architecture
families x 3 seeds, no target labels. Compares PoolEval against the pool
baselines (B1-B4) and the confidence baselines MetaEvaluator uses (DoC, ATC).
"""
import argparse, json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._domain_eval import score_pool, print_rows, summarize   # noqa: E402
from pooleval.domains.vision import build_pool, FAMILIES                 # noqa: E402

SHIFTS = [("mnist", "usps"), ("mnist", "svhn")]


def get_pool(src, dst, seeds, epochs, cache_dir, n_target):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"vision_{src}_{dst}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    pool = build_pool(src, dst, seeds=seeds, epochs=epochs, n_target=n_target)
    np.savez(cache, **{k: v for k, v in pool.items() if k not in ("src", "dst")})
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--n-target", type=int, default=5000)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/pooleval_domains/pools"))
    ap.add_argument("--out", default="results/domain_vision.json")
    a = ap.parse_args()

    allrows, per_shift = {}, {}
    for src, dst in SHIFTS:
        print(f"\n=== {src} -> {dst} ===", flush=True)
        pool = get_pool(src, dst, tuple(range(a.seeds)), a.epochs, a.cache, a.n_target)
        rows, truth = score_pool(pool, extra_variants={
            "PoolEval (learned verif.)": dict(verifier_mode="learned"),
            "PoolEval (no anchors)": dict(use_verifier=False, use_prior=False)})
        per_shift[f"{src}->{dst}"] = rows
        print(f"  true target acc: {truth.min():.3f}-{truth.max():.3f} "
              f"(mean {truth.mean():.3f}), N={len(pool['gold'])}")
        print_rows(rows)
        for k, v in rows.items():
            allrows.setdefault(k, []).append(v)

    summary = summarize(allrows, "MEAN OVER SHIFTS")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(per_shift=per_shift, summary=summary), open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
