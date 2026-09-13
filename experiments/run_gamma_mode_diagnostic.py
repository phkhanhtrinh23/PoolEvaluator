"""Is the both-wrong -> P(C=0|Z=0) conversion biased, and by how much, per domain?

The conversion ``gamma^model = beta + (1 - beta) gamma^both`` is derived by conditioning on
whether the pseudo-label happens to be correct, and its last step assumes

    P(yhat = y | model j wrong) = beta,

i.e. that the pseudo-label's correctness is independent of which classifier got it wrong.
That is an empirical claim, and it is checkable on any labeled split: compute both sides.

If the gap is large the conversion is systematically biased and ``gamma_mode="model_wrong"``
(which counts the conditional directly, with no conversion) is the right default.  If the
gap is small the conversion is nearly exact, and ``both_wrong`` keeps its extra
conditioning for free.

  python experiments/run_gamma_mode_diagnostic.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval.domains.adapter import encode_classes                   # noqa: E402
from pooleval.validated_em import (LatentPlan, excess_collision,      # noqa: E402
                                   gamma_counts, gamma_from_counts,
                                   gamma_to_conditional, hard_labels,
                                   wrong_collision_matrix)
from experiments.run_validated_em import _subsample                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")


def diagnose(name, tc, group, prior):
    """tc: [M, N] labeled classes, 0 == correct. Returns the gap and its consequence."""
    ex = excess_collision(wrong_collision_matrix(tc), group)
    yhat = hard_labels(LatentPlan(tc, ex).posterior(prior))
    beta = float((yhat == 0).mean())
    # the conditional the derivation actually needs, averaged over classifiers
    cond = float(np.mean([(yhat[tc[j] != 0] == 0).mean() if (tc[j] != 0).any() else beta
                          for j in range(tc.shape[0])]))

    nb, db = gamma_counts(tc, yhat, mode="both_wrong")
    nm, dm = gamma_counts(tc, yhat, mode="model_wrong")
    g_both = gamma_from_counts(nb, db)
    g_model = gamma_from_counts(nm, dm)
    predicted = gamma_to_conditional(g_both, beta)          # what the identity gives
    exact = gamma_to_conditional(g_both, cond)              # identity with the RIGHT beta
    return dict(name=name, M=int(tc.shape[0]), N=int(tc.shape[1]),
                beta=beta, conditional=cond, gap=cond - beta,
                gamma_both=float(g_both.mean()),
                gamma_model_counted=float(g_model.mean()),
                gamma_model_predicted=float(predicted.mean()),
                gamma_model_with_true_conditional=float(exact.mean()),
                error=float(np.abs(predicted - g_model).mean()))


def main():
    rows = []
    from zoo.new_formulation_real import load_run
    src = np.load(os.path.join(ROOT, "zoo_artifacts", "source_true_class.npz"))
    for ds in ["spider", "bird"]:
        run, _ = load_run(ds)
        rows.append(diagnose(f"text2sql/{ds}", src[ds], run.group, run.prior))

    for key in ["vision_mnist_usps", "vision_mnist_svhn",
                "graph_AC", "graph_AD", "graph_CA", "graph_CD", "graph_DA", "graph_DC"]:
        path = os.path.join(POOL_CACHE, key + ".npz")
        if not os.path.exists(path):
            continue
        z = np.load(path, allow_pickle=True)
        pred_s = (z["pred_s"] if "pred_s" in z.files else z["prob_s"].argmax(-1))
        si = _subsample(pred_s.shape[1], 3000, 0)
        tc = encode_classes(pred_s[:, si], z["src_gold_val"][si])
        dom = "vision/" if key.startswith("vision") else "graph/"
        rows.append(diagnose(dom + key.split("_", 1)[1], tc, z["group"], z["prior"]))

    print(f"{'labeled split':22s}{'beta':>8s}{'P(yhat=y|j wrong)':>19s}{'gap':>8s}"
          f"{'g_model counted':>17s}{'identity predicts':>19s}{'error':>8s}")
    for r in rows:
        print(f"{r['name']:22s}{r['beta']:8.3f}{r['conditional']:19.3f}{r['gap']:+8.3f}"
              f"{r['gamma_model_counted']:17.3f}{r['gamma_model_predicted']:19.3f}"
              f"{r['error']:8.3f}")
    for dom in ("text2sql", "vision", "graph"):
        sub = [r for r in rows if r["name"].startswith(dom)]
        if sub:
            print(f"  {dom:20s} mean |gap| {np.mean([abs(r['gap']) for r in sub]):.3f}"
                  f"   mean conversion error {np.mean([r['error'] for r in sub]):.3f}")
    with open(os.path.join(ROOT, "results", "gamma_mode_diagnostic.json"), "w") as h:
        json.dump(rows, h, indent=2, default=float)
    print("\n[saved] results/gamma_mode_diagnostic.json")


if __name__ == "__main__":
    main()
