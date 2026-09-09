"""RQ-D10: the UNBOUNDED-answer-space port -- image captioning.

COCO validation images, pool = 4 pretrained captioners x 3 decoding seeds, with a
fifth captioner held out as the verifier. References are withheld from the
estimator and used only for scoring.

This is the first non-SQL task in the repo whose answer space is not enumerable.
Full Dawid--Skene is not merely expensive here, it is undefined: a confusion
matrix needs class identity to persist across items, and caption equivalence
classes are per-image clusters. The estimators that survive are the ones that
only ever test observations for equality.

The kernel threshold is the strictness dial (the analogue of Text2SQL's
LA0/LA1/LA2), so results are reported as a function of it rather than at one
arbitrary setting.
"""
import argparse, json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._domain_eval import (score_pool, print_rows,   # noqa: E402
                                      summarize, print_gamma)
from pooleval.domains.caption import build_pool, encode_pool, load_images  # noqa: E402


def get_pool(n_source, n_target, seeds, threshold, cache_dir):
    """Captions are generated once and re-encoded at each threshold."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"caption_{n_source}_{n_target}_{len(seeds)}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    pool = build_pool(n_source=n_source, n_target=n_target, seeds=seeds,
                      threshold=threshold)
    np.savez(cache, **{k: v for k, v in pool.items() if k != "unbounded"})
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-source", type=int, default=300)
    ap.add_argument("--n-target", type=int, default=700)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/pooleval_domains/pools"))
    ap.add_argument("--out", default="results/domain_caption.json")
    a = ap.parse_args()

    base = get_pool(a.n_source, a.n_target, tuple(range(a.seeds)),
                    a.thresholds[0], a.cache)
    caps = base["captions"]                       # [M][N] strings, all images
    n_src = base["pred_s"].shape[1]
    images, refs = load_images(caps.shape[1])     # references, for re-encoding only
    ver_caps = base.get("verifier_captions")

    allrows, per_th, gammas = {}, {}, {}
    for th in a.thresholds:
        print(f"\n=== kernel threshold {th} ===", flush=True)
        pred, gold, vg = encode_pool(list(caps), list(ver_caps), refs, th)
        sl, tl = slice(0, n_src), slice(n_src, pred.shape[1])
        K = int(max(pred[:, tl].max(), gold[tl].max(), vg[tl].max())) + 1
        pool = dict(pred=pred[:, tl], gold=gold[tl], group=base["group"],
                    prior=(pred[:, sl] == gold[None, sl]).mean(1),
                    verifier_guess=vg[tl], names=base["names"], n_classes=K,
                    pred_s=pred[:, sl], src_gold_val=gold[sl],
                    unbounded=True)      # caption clusters are per-item, not global
        rows, truth, diag = score_pool(pool, extra_variants={
            "PoolEval (learned verif.)": dict(verifier_mode="learned"),
            "PoolEval (no anchors)": dict(use_verifier=False, use_prior=False)})
        per_th[str(th)] = rows
        gammas[str(th)] = diag
        print(f"  true caption accuracy: {truth.min():.3f}-{truth.max():.3f} "
              f"(mean {truth.mean():.3f}), N={pool['pred'].shape[1]}, "
              f"widest candidate set K={K}")
        print_gamma(diag)
        print_rows(rows)
        for k, v in rows.items():
            allrows.setdefault(k, []).append(v)

    summary = summarize(allrows, f"MEAN OVER {len(a.thresholds)} KERNEL THRESHOLDS")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(per_threshold=per_th, summary=summary, gamma=gammas),
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
