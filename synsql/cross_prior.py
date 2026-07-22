"""Where should the prior come from? In-domain vs another benchmark vs SynSQL.

The SynSQL study compares a retrieved corpus prior against the target's OWN labeled
split. That leaves the middle ground untested: what if the prior comes from a
different REAL benchmark? This builds the full source x target matrix from priors
already measured (each poolrun_<ds>.npz stores its source-split per-model EX), so it
costs no generation at all.

Diagonal = the in-domain prior. Off-diagonal = cross-benchmark transfer.

    python -m synsql.cross_prior
"""
import json
import os

import numpy as np

from synsql.config import RESULTS
from synsql.run_prior import load_run, score_prior

DS = ["spider", "bird", "bird_minidev", "sqlflow", "spider2local"]


def main():
    runs, priors, sigmas = {}, {}, {}
    for ds in DS:
        run, _ = load_run(ds)
        runs[ds] = run
        priors[ds] = np.asarray(run.prior, dtype=float)
        sigmas[ds] = np.asarray(run.prior_sigma, dtype=float)

    synsql = {}
    p = os.path.join(RESULTS, "synsql_prior.json")
    if os.path.exists(p):
        with open(p) as f:
            R = json.load(f)
        for ds, v in R["targets"].items():
            synsql[ds] = v["conditions"]["synsql-topk"]

    out = {}
    print("prior MAE (rows = prior source, cols = target); diagonal = in-domain\n")
    print(f"{'prior from':16s}" + "".join(f"{d[:12]:>14s}" for d in DS))
    for src in DS:
        cells, out[src] = "", {}
        for tgt in DS:
            s = score_prior(runs[tgt], priors[src], sigmas[src])
            out[src][tgt] = s
            mark = "*" if src == tgt else " "
            cells += f"{s['prior_mae']:>13.2f}{mark}"
        print(f"{src:16s}{cells}")
    if synsql:
        cells = "".join(f"{synsql[d]['prior_mae']:>13.2f} " if d in synsql
                        else f"{'-':>14s}" for d in DS)
        print(f"{'SynSQL top-k':16s}{cells}")

    print("\ncentered MAE (level removed) -- how well the prior captures RELATIVE ability\n")
    print(f"{'prior from':16s}" + "".join(f"{d[:12]:>14s}" for d in DS))
    for src in DS:
        cells = "".join(f"{out[src][t]['prior_mae_c']:>13.2f}"
                        f"{'*' if src == t else ' '}" for t in DS)
        print(f"{src:16s}{cells}")
    if synsql:
        cells = "".join(f"{synsql[d]['prior_mae_c']:>13.2f} " if d in synsql
                        else f"{'-':>14s}" for d in DS)
        print(f"{'SynSQL top-k':16s}{cells}")

    # summary: in-domain vs best/mean cross-benchmark vs SynSQL, per target
    print("\nper target: in-domain vs cross-benchmark vs SynSQL (prior MAE / centered)\n")
    print(f"{'target':14s}{'in-domain':>12s}{'best cross':>12s}{'mean cross':>12s}"
          f"{'SynSQL':>10s}   | centered: in / cross / SynSQL")
    summary = {}
    for tgt in DS:
        ind = out[tgt][tgt]
        cross = [out[s][tgt] for s in DS if s != tgt]
        best = min(cross, key=lambda x: x["prior_mae"])
        syn = synsql.get(tgt)
        summary[tgt] = dict(
            in_domain=ind, best_cross=best,
            mean_cross_mae=float(np.mean([c["prior_mae"] for c in cross])),
            mean_cross_mae_c=float(np.mean([c["prior_mae_c"] for c in cross])),
            synsql=syn)
        print(f"{tgt:14s}{ind['prior_mae']:>12.2f}{best['prior_mae']:>12.2f}"
              f"{summary[tgt]['mean_cross_mae']:>12.2f}"
              f"{(syn['prior_mae'] if syn else float('nan')):>10.2f}   | "
              f"{ind['prior_mae_c']:.2f} / {summary[tgt]['mean_cross_mae_c']:.2f} / "
              f"{(syn['prior_mae_c'] if syn else float('nan')):.2f}")

    with open(os.path.join(RESULTS, "synsql_cross_prior.json"), "w") as f:
        json.dump(dict(matrix=out, summary=summary), f, indent=2, default=float)
    print(f"\n[saved] {RESULTS}/synsql_cross_prior.json")


if __name__ == "__main__":
    main()
