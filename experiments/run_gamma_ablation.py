"""RQ-D5: how much does the collision rate gamma actually buy, on its own?

A naive comparison of `NF binary EM (g=1)` against `NF collision (g=source)` is
CONFOUNDED: the first uses `run.prior` only as an initialiser, the second keeps it
as a Beta anchor and starts beta from the source estimate. This script holds the
anchors and the beta initialisation fixed and varies ONLY gamma, which is the only
way to attribute anything to the collision correction.

It also sweeps gamma over its whole range to show WHY the effect is small in these
domains: gamma enters the likelihood only through (1-beta)*gamma, so its influence
scales with (1-beta)/beta. When the pseudo-label is good the collision term is
dwarfed by alpha*beta and gamma is nearly inert.
"""
import argparse, dataclasses, json, os, sys, warnings
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pooleval.config import Config                                        # noqa: E402
from pooleval.domains.adapter import from_predictions                     # noqa: E402
from pooleval.domains.collision import (alpha_effective_size,             # noqa: E402
                                        pseudo_labels, source_gamma)
from pooleval.new_formulation import agreement_em, collision_agreement_em  # noqa: E402

CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")
POOLS = {"node": [f"graph_{a}{b}" for a, b in ("AC", "AD", "CA", "CD", "DA", "DC")],
         "image": ["vision_mnist_usps", "vision_mnist_svhn"]}
SWEEP = [0.10, 0.25, 0.50, 0.75, 0.90, 1.0 - 1e-9]


def prepare(name):
    z = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    pool = {k: z[k] for k in z.files}
    run = from_predictions(pool["pred"], pool["gold"], pool["group"],
                           prior=pool["prior"], verifier_guess=pool["verifier_guess"])
    cfg = dataclasses.replace(Config(), real_data=True, M=run.M, N=run.N,
                              n_groups=run.n_groups)
    _, pseudo = pseudo_labels(run, cfg)
    C = (run.true_class == pseudo[None, :]).astype(float)
    return pool, run, C, alpha_effective_size(run), source_gamma(pool, cfg)


def score(a, truth):
    return float(np.abs(a - truth).mean()), float(spearmanr(a, truth).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gamma_ablation.json")
    a = ap.parse_args()
    out = {}

    for domain, files in POOLS.items():
        print(f"\n{'='*76}\n{domain.upper()}  -- anchors and beta init held FIXED, only gamma varies"
              f"\n{'='*76}")
        print(f"{'run':20s} {'g=1':>15} {'g=1/(K-1)':>15} {'g=source':>15}")
        agg, per_run = {}, {}
        for f in files:
            pool, run, C, s, src = prepare(f)
            K, G, truth = int(np.asarray(pool["n_classes"])), run.n_groups, run.true_acc
            line, row = f"{f:20s}", {}
            for lbl, gam in (("g=1", np.full(G, 1 - 1e-9)),
                             ("g=null", np.full(G, 1.0 / (K - 1))),
                             ("g=source", src["gamma"])):
                fit = collision_agreement_em(C, run.group, gam, run.prior, s,
                                             beta_init=src["beta"], beta_strength=120.0)
                m, r = score(fit["alpha"], truth)
                agg.setdefault(lbl, []).append((m, r)); row[lbl] = dict(mae=m, spearman=r)
                line += f"  {m:.4f}/{r:+.3f}"
            # the confounded comparison, for the record
            m0, r0 = score(agreement_em(C, run.prior)["alpha"], truth)
            row["g=1 unanchored (confounded)"] = dict(mae=m0, spearman=r0)
            per_run[f] = row
            print(line)
        print(f"{'MEAN':20s}" + "".join(
            f"  {np.mean([x[0] for x in agg[k]]):.4f}/{np.mean([x[1] for x in agg[k]]):+.3f}"
            for k in ("g=1", "g=null", "g=source")))
        wins = sum(x[0] < y[0] for x, y in zip(agg["g=source"], agg["g=1"]))
        print(f"  -> gamma=source beats gamma=1 on MAE in {wins}/{len(files)} runs")
        out[domain] = dict(per_run=per_run, gamma_source_wins=wins, n_runs=len(files))

    # WHY: sweep gamma and watch how little alpha moves
    print(f"\n{'='*76}\nWHY gamma is nearly inert here: it enters only as (1-beta)*gamma"
          f"\n{'='*76}")
    sweeps = {}
    for f in ("graph_AC", "vision_mnist_usps"):
        pool, run, C, s, src = prepare(f)
        G, truth = run.n_groups, run.true_acc
        print(f"\n{f}   (a 10x change in gamma is a {'':s}small change in alpha)")
        print(f"{'gamma':>8} {'fitted beta':>12} {'(1-b)*g':>9} {'mean alpha':>11} {'MAE':>8}")
        rows = []
        for g in SWEEP:
            fit = collision_agreement_em(C, run.group, np.full(G, g), run.prior, s,
                                         beta_init=src["beta"], beta_strength=0.0)
            b, m = fit["beta"], score(fit["alpha"], truth)[0]
            rows.append(dict(gamma=g, beta=b, product=(1 - b) * g,
                             mean_alpha=float(fit["alpha"].mean()), mae=m))
            print(f"{g:8.2f} {b:12.4f} {(1-b)*g:9.4f} {fit['alpha'].mean():11.4f} {m:8.4f}")
        span = max(r["mean_alpha"] for r in rows) - min(r["mean_alpha"] for r in rows)
        print(f"  total swing in mean alpha across a 10x gamma range: {span:.4f}")
        sweeps[f] = dict(rows=rows, mean_alpha_span=span)
    out["sweep"] = sweeps

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
