"""RQ-D4: is Dawid--Skene's uniform-error assumption true on real pools?

One-coin DS assumes P(obs = k | truth = c) = (1-a)/(K-1) for every wrong k. That
single line implies two things at once:

  (i)  UNIFORMITY   -- a model's errors spread evenly over the K-1 wrong classes
  (ii) INDEPENDENCE -- two wrong models collide with probability exactly 1/(K-1)

This script tests both by COUNTING. No EM, no pseudo-labels, no estimator: just
the raw predictions and the withheld gold, so the numbers cannot be an artefact
of any method. It then shows, on the pool where one-coin DS fails hardest, what a
per-model confusion matrix sees that one-coin cannot.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pooleval.domains.multiclass_ds import full_ds, one_coin_ds   # noqa: E402

CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")
POOLS = {"node": [f"graph_{a}{b}" for a, b in ("AC", "AD", "CA", "CD", "DA", "DC")],
         "image": ["vision_mnist_usps", "vision_mnist_svhn"]}


def load(name):
    z = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def assumption_stats(pool, min_pairs=30):
    """Uniformity and independence, measured directly from predictions + gold."""
    pred, gold = np.asarray(pool["pred"]), np.asarray(pool["gold"])
    group, K = np.asarray(pool["group"]), int(np.asarray(pool["n_classes"]))
    M, null = pred.shape[0], 1.0 / (K - 1)
    wrong = pred != gold[None, :]

    # (i) uniformity: how much mass sits on a model's single favourite error?
    conc = []
    for m in range(M):
        w = pred[m][wrong[m]]
        if len(w):
            conc.append((np.bincount(w, minlength=K) / len(w)).max())

    # (ii) independence: P(same wrong answer | both wrong), by provenance
    same, diff = [], []
    for m in range(M):
        for n in range(m + 1, M):
            both = wrong[m] & wrong[n]
            if both.sum() < min_pairs:
                continue
            r = float((pred[m][both] == pred[n][both]).mean())
            (same if group[m] == group[n] else diff).append(r)
    return dict(K=K, null=null, acc=float((~wrong).mean()),
                top1_error_mass=float(np.mean(conc)),
                collide_same_group=float(np.mean(same)) if same else float("nan"),
                collide_diff_group=float(np.mean(diff)) if diff else float("nan"))


def confusion_diagnostic(pool):
    """Why a confusion matrix rescues a collapsed pool that one-coin DS cannot."""
    pred, gold = np.asarray(pool["pred"]), np.asarray(pool["gold"])
    K = int(np.asarray(pool["n_classes"]))
    truth = (pred == gold[None, :]).mean(1)
    one, full = one_coin_ds(pred, K), full_ds(pred, K)
    # A model whose output does not depend on the truth has near-identical
    # confusion rows. Row spread therefore measures "is this vote informative".
    pi = full["confusion"]
    spread = np.array([np.abs(pi[j] - pi[j].mean(0, keepdims=True)).mean()
                       for j in range(len(pi))])
    share = np.array([(np.bincount(pred[j], minlength=K) / pred.shape[1]).max()
                      for j in range(pred.shape[0])])
    return dict(names=[str(x) for x in pool["names"]], true=truth.tolist(),
                one_coin=one["acc"].tolist(), full=full["acc"].tolist(),
                top_class_share=share.tolist(), confusion_row_spread=spread.tolist(),
                latent_acc_one_coin=float((one["latent_hat"] == gold).mean()),
                latent_acc_full=float((full["latent_hat"] == gold).mean()),
                corr_spread_truth=float(np.corrcoef(spread, truth)[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ds_assumption.json")
    ap.add_argument("--diagnostic-pool", default="vision_mnist_svhn")
    a = ap.parse_args()

    out = {}
    for domain, files in POOLS.items():
        print(f"\n{'='*78}\n{domain.upper()}  -- one-coin DS assumes both columns equal 1/(K-1)\n{'='*78}")
        print(f"{'run':22s} {'K':>3} {'acc':>6} {'top-1 err mass':>15} "
              f"{'collide same-grp':>17} {'diff-grp':>10} {'DS says':>9}")
        rows = []
        for f in files:
            s = assumption_stats(load(f))
            rows.append(dict(run=f, **s))
            print(f"{f:22s} {s['K']:3d} {s['acc']:6.3f} {s['top1_error_mass']:15.3f} "
                  f"{s['collide_same_group']:17.3f} {s['collide_diff_group']:10.3f} "
                  f"{s['null']:9.3f}")
        mean = {k: float(np.mean([r[k] for r in rows]))
                for k in ("null", "top1_error_mass", "collide_same_group", "collide_diff_group")}
        print(f"{'MEAN':22s} {'':3} {'':6} {mean['top1_error_mass']:15.3f} "
              f"{mean['collide_same_group']:17.3f} {mean['collide_diff_group']:10.3f} "
              f"{mean['null']:9.3f}")
        print(f"{'  ratio to DS':22s} {'':3} {'':6} "
              f"{mean['top1_error_mass']/mean['null']:14.1f}x "
              f"{mean['collide_same_group']/mean['null']:16.1f}x "
              f"{mean['collide_diff_group']/mean['null']:9.1f}x")
        out[domain] = dict(runs=rows, mean=mean)

    d = confusion_diagnostic(load(a.diagnostic_pool))
    print(f"\n{'='*78}\nWHY A CONFUSION MATRIX RESCUES {a.diagnostic_pool}\n{'='*78}")
    print(f"{'model':18s} {'true':>6} {'one-coin':>9} {'full':>7} {'top-class share':>16} {'row spread':>11}")
    for j in np.argsort(-np.array(d["true"])):
        print(f"{d['names'][j]:18s} {d['true'][j]:6.3f} {d['one_coin'][j]:9.3f} "
              f"{d['full'][j]:7.3f} {d['top_class_share'][j]:16.3f} "
              f"{d['confusion_row_spread'][j]:11.4f}")
    print(f"\npseudo-label accuracy: one-coin {d['latent_acc_one_coin']:.3f} -> "
          f"full {d['latent_acc_full']:.3f}")
    print(f"corr(confusion row spread, true accuracy) = {d['corr_spread_truth']:+.3f}")
    out["confusion_diagnostic"] = dict(pool=a.diagnostic_pool, **d)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
