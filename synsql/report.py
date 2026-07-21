"""Render results/synsql_prior.json into the markdown tables used in the write-up.

    python -m synsql.report
"""
import json
import os

import numpy as np

from .config import RESULTS

ORDER = ["insplit (target-labeled)", "synsql-topk", "synsql-soft", "synsql-all",
         "synsql-random", "synsql-far", "synsql-topk +true level",
         "synsql-oracle (bound)"]


def main():
    with open(os.path.join(RESULTS, "synsql_prior.json")) as f:
        r = json.load(f)

    print("### Partition separability (std of d(S,T) across subsets)\n")
    tgts = list(r["targets"]) or list(next(iter(r["spread"].values())))
    print("| partition | " + " | ".join(tgts) + " |")
    print("|---" * (len(tgts) + 1) + "|")
    for p, per in r["spread"].items():
        print(f"| `{p}` | " + " | ".join(f"{per[t]['std']:.4f}" for t in tgts) + " |")

    print("\n### Does distance predict prior error?\n")
    print("| target | corr(d, prior MAE) | nearest d | farthest d |")
    print("|---|---|---|---|")
    for t, v in r["targets"].items():
        pc = v["per_cand"]
        print(f"| {t} | {v['rho_dist_vs_err']:+.3f} | {min(pc['dist']):.3f} "
              f"| {max(pc['dist']):.3f} |")

    for t, v in r["targets"].items():
        print(f"\n### {t}  (M={v['M']}, N={v['N']}, "
              f"true EX {np.mean(v['true_acc']):.3f})\n")
        print("| prior | prior MAE ↓ | bias | centered MAE ↓ | prior Kendall ↑ | "
              "PoolEval MAE ↓ | Kendall ↑ |")
        print("|---|---|---|---|---|---|---|")
        for c in ORDER:
            if c not in v["conditions"]:
                continue
            x = v["conditions"][c]
            print(f"| {c} | {x['prior_mae']:.2f} | {x['prior_bias']:+.2f} | "
                  f"{x['prior_mae_c']:.2f} | {x['prior_kendall']:.2f} | "
                  f"{x['pe_MAE']:.2f} | {x['pe_Kendall']:.2f} |")

    print("\n### Aggregate (mean over targets)\n")
    print("| prior | prior MAE ↓ | bias | centered MAE ↓ | prior Kendall ↑ | "
          "PoolEval MAE ↓ | Kendall ↑ |")
    print("|---|---|---|---|---|---|---|")
    for c in ORDER:
        vals = [v["conditions"][c] for v in r["targets"].values()
                if c in v["conditions"]]
        if not vals:
            continue
        f = lambda k: np.mean([x[k] for x in vals])       # noqa: E731
        print(f"| {c} | {f('prior_mae'):.2f} | {f('prior_bias'):+.2f} | "
              f"{f('prior_mae_c'):.2f} | {f('prior_kendall'):.2f} | "
              f"{f('pe_MAE'):.2f} | {f('pe_Kendall'):.2f} |")


if __name__ == "__main__":
    main()
