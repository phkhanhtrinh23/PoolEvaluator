"""Figure for results/subset_count.json: one x, two y axes.

  x            number of labeled subsets K used to compute the prior
  left  y      latency of computing the prior accuracy per model (ms), mean over the
               3 modalities
  right y      MAE of the final accuracy estimate (accuracy points), mean over the
               3 modalities, at expert budget 0 and 40

Nothing is computed here -- this only draws the aggregate block the experiment already
wrote, so the figure and the table can never disagree.

  python experiments/plot_subset_count.py
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C_TIME = "#1f6feb"     # left axis
C_MAE = "#d1495b"      # right axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "results", "subset_count.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "subset_count"))
    ap.add_argument("--time-field", default="t_pi_ms",
                    choices=["t_pi_ms", "t_all_ms"])
    args = ap.parse_args()

    if not os.path.exists(args.src):
        sys.exit(f"missing {args.src} -- run experiments/run_subset_count.py first")
    with open(args.src) as h:
        blob = json.load(h)
    agg, meta = blob["aggregate"], blob["metadata"]

    ks = sorted(int(k) for k in agg)
    t = np.array([agg[str(k)][args.time_field] for k in ks])
    m0 = np.array([agg[str(k)]["b0"] for k in ks])
    budgets = meta["budgets"]
    mb = np.array([agg[str(k)][f"b{budgets[-1]}"] for k in ks])
    size = meta["subset_size"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax2 = ax.twinx()

    l1, = ax.plot(ks, t, "o-", color=C_TIME, lw=2.0, ms=6,
                  label="prior-accuracy latency per model")
    l2, = ax2.plot(ks, m0, "s--", color=C_MAE, lw=1.8, ms=6, alpha=0.65,
                   label="MAE, budget 0")
    l3, = ax2.plot(ks, mb, "^-", color=C_MAE, lw=2.0, ms=6,
                   label=f"MAE, budget {budgets[-1]}")

    ax.set_xlabel(f"number of labeled subsets $K$   ({size} items each)")
    ax.set_ylabel("latency per model  (ms)", color=C_TIME)
    ax2.set_ylabel("MAE  (accuracy points)", color=C_MAE)
    ax.tick_params(axis="y", colors=C_TIME)
    ax2.tick_params(axis="y", colors=C_MAE)
    ax.spines["left"].set_color(C_TIME)
    ax.spines["right"].set_color(C_MAE)
    ax2.spines["left"].set_color(C_TIME)
    ax2.spines["right"].set_color(C_MAE)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    ax.set_xticks(ks)
    ax.set_xlim(min(ks) - 0.4, max(ks) + 0.4)
    ax.set_ylim(0, t.max() * 1.18)
    lo = min(m0.min(), mb.min())
    hi = max(m0.max(), mb.max())
    ax2.set_ylim(lo - 0.12 * (hi - lo), hi + 0.22 * (hi - lo))
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.45)
    ax.set_axisbelow(True)

    # A second x-scale in items, since "K subsets" is only meaningful with the size.
    top = ax.secondary_xaxis("top", functions=(lambda k: k * size,
                                               lambda n: n / size))
    top.set_xlabel("labeled items used for the prior", fontsize=9)
    top.set_xticks([k * size for k in ks])
    top.tick_params(labelsize=8)

    ax.legend(handles=[l1, l2, l3], loc="center left", frameon=True, framealpha=0.92,
              edgecolor="none", fontsize=9)
    ax.set_title("Cost of the prior vs. accuracy of the estimate\n"
                 f"mean over {len(agg[str(ks[0])]['per_modality'])} modalities, "
                 f"{meta['draws']} draws", fontsize=11, pad=26)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
        print(f"[saved] {args.out}.{ext}")


if __name__ == "__main__":
    main()
