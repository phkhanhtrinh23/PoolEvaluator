"""How many target labels does the GAUGE cost?

The main SynSQL experiment shows the retrieved prior gets the pool's RELATIVE ability
right (centered MAE ~3 pts, on par with the in-split prior) but its LEVEL wrong by
15-53 pts, and that no amount of retrieval alignment fixes the level. So the natural
question is: if we buy back only the level with a few real target labels, how few is
enough?

Protocol: take the retrieved SynSQL prior, label j items drawn at random from the
target, and shift the whole prior so its mean matches the pool's observed accuracy on
those j items. Everything else stays label-free. j = 0 is the pure SynSQL prior.

Costs no extra generation -- it reuses the measured PoolRun and probe cache.

    python -m synsql.level_budget
"""
import json
import os

import numpy as np

from synsql.config import RESULTS, TOP_K
from synsql.run_prior import load_run, score_prior

JS = (0, 2, 5, 10, 20, 40, 80)
N_REP = 200


def level_shift(prior, run, j, rng):
    """Shift the prior so the pool's mean accuracy matches j labeled target items."""
    if j == 0:
        return np.asarray(prior, dtype=float)
    cols = rng.choice(run.N, size=min(j, run.N), replace=False)
    obs = (run.true_class[:, cols] == 0).mean()      # pool mean EX on the j labels
    p = np.asarray(prior, dtype=float)
    return p - p.mean() + obs


def main(n_rep=N_REP, seed=0):
    with open(os.path.join(RESULTS, "synsql_prior.json")) as f:
        R = json.load(f)
    out = {}
    print(f"{'target':14s}{'j':>4s}{'priorMAE':>10s}{'PE MAE':>9s}{'PE Ken':>8s}")
    for ds, v in R["targets"].items():
        run, _ = load_run(ds)
        prior = np.asarray(v["conditions"]["synsql-topk"]["_prior"]) \
            if "_prior" in v["conditions"]["synsql-topk"] else None
        if prior is None:                 # recompute topk prior from the probe cache
            from synsql.prior import prior_from_probes
            z = np.load(os.path.join(os.path.dirname(RESULTS), "zoo_artifacts",
                                     "synsql_probes.npz"), allow_pickle=True)
            store = {int(k): val for k, val in z["store"].item().items()}
            prior, sigma, _ = prior_from_probes(store, v["top"])
        else:
            sigma = np.full(run.M, 0.04)
        out[ds] = {}
        for j in JS:
            rng = np.random.default_rng(seed)
            pm, pe, pk = [], [], []
            reps = 1 if j == 0 else n_rep
            for _ in range(reps):
                p = level_shift(prior, run, j, rng)
                s = score_prior(run, p, sigma)
                pm.append(s["prior_mae"]); pe.append(s["pe_MAE"])
                pk.append(s["pe_Kendall"])
            out[ds][j] = dict(prior_mae=float(np.mean(pm)),
                              prior_mae_sd=float(np.std(pm)),
                              pe_MAE=float(np.mean(pe)),
                              pe_Kendall=float(np.mean(pk)))
            print(f"{ds:14s}{j:>4d}{np.mean(pm):>10.2f}{np.mean(pe):>9.2f}"
                  f"{np.mean(pk):>8.2f}")
        # reference: the in-split prior, which costs a whole labeled train split
        ins = v["conditions"]["insplit (target-labeled)"]
        out[ds]["insplit"] = dict(prior_mae=ins["prior_mae"], pe_MAE=ins["pe_MAE"],
                                  pe_Kendall=ins["pe_Kendall"])
        print(f"{ds:14s}{'ins':>4s}{ins['prior_mae']:>10.2f}{ins['pe_MAE']:>9.2f}"
              f"{ins['pe_Kendall']:>8.2f}")
    with open(os.path.join(RESULTS, "synsql_level_budget.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {RESULTS}/synsql_level_budget.json")


if __name__ == "__main__":
    main()
