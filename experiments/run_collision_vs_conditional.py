"""Is the ORIGINAL collision formulation -- gamma^coll with beta, no beta-hat -- still good?

This is the head-to-head the later `both_wrong` / `model_wrong` work implicitly replaced but
never measured against on equal terms. All three arms run the SAME pipeline (same vote
discount via e, same A_mu selector, same expert, same warm-start, same refresh rule). The
ONLY difference is how the Z=0 branch of the likelihood is parameterised:

  collision     gamma_j^coll = P(r^j = yhat | both wrong), measured and frozen.
                E-step consumes d_j(beta) = 1 - (1-beta) gamma^coll, rebuilt from the LIVE
                beta every sweep. The beta M-step is therefore a bounded 1-D numerical
                maximisation. beta-hat DOES NOT EXIST in this arm: beta starts at a neutral
                0.7 and is thereafter always the EM-fitted value carried through warm-starts.

  both_wrong    gamma_j^both = 1 - gamma_j^coll, same counts, complementary direction.
                Converted once per solve by gamma^model = beta_hat + (1-beta_hat) gamma^both,
                so beta-hat IS used -- but only as a frozen constant, which buys back the
                closed-form beta M-step.

  model_wrong   gamma_j = P(r^j != yhat | r^j wrong) counted directly on the wider
                denominator. No conversion, so no beta-hat in gamma at all, and the beta
                M-step is closed form. (Current default.)

Reported alongside MAE: wall-clock per solve, since the collision arm pays for a numerical
M-step, and the recovered beta, since the three arms disagree about what beta even means.

  python experiments/run_collision_vs_conditional.py --seeds 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval.validated_em import (OracleExpert, run_validation,      # noqa: E402
                                   validated_em)
from experiments.run_validated_em import _strength, build_stats, mae  # noqa: E402
from experiments.run_random_init import load_cases                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODES = ["collision", "both_wrong", "model_wrong"]
NEUTRAL_BETA = 0.7      # the collision arm's beta_init; deliberately NOT measured


def fresh(mode, c):
    """Statistics for one arm. In collision mode beta-hat is overwritten with a neutral
    constant BEFORE any solve, so no measured pseudo-label accuracy can leak in: the
    E-step reads stats.gamma directly, and every warm-start carries the fitted beta."""
    st = build_stats(c["labeled"], c["group"], c["prior"], gamma_mode=mode)
    if mode == "collision":
        st._pseudo_acc = NEUTRAL_BETA
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 40])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-target", type=int, default=600)
    ap.add_argument("--n-labeled", type=int, default=3000)
    ap.add_argument("--prior-noise", type=float, default=0.07)
    ap.add_argument("--ig-candidates", type=int, default=25)
    ap.add_argument("--ig-iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "collision_vs_conditional.json"))
    args = ap.parse_args()

    cols = ["b0"] + [f"b{b}" for b in args.budgets]
    payload = {}
    for name, c in load_cases(args).items():
        obs = c["obs"]
        pi = np.asarray(c["prior"], float)
        s = _strength(pi, c["prior_sigma"], cap=obs.shape[1])
        print(f"\n  [{name}]  N={obs.shape[1]}  M={obs.shape[0]}", flush=True)
        print(f"    {'mode':13s}" + "".join(f"{k:>14s}" for k in cols)
              + f"{'t_b0(ms)':>11s}{'beta_b0':>10s}{'iters':>7s}")
        payload[name] = {}
        for mode in MODES:
            got = {k: [] for k in cols}
            secs, betas, iters = [], [], []
            for rep in range(args.seeds):
                seed = args.seed + 977 * rep
                t0 = time.perf_counter()
                out0 = validated_em(obs, fresh(mode, c), pi, s, max_iters=200)
                secs.append((time.perf_counter() - t0) * 1e3)
                betas.append(float(out0["beta"]))
                iters.append(int(out0["n_iters"]))
                got["b0"].append(mae(out0["acc"], c["true_acc"]))
                for b in args.budgets:
                    o = run_validation(obs, fresh(mode, c), pi, s,
                                       expert=OracleExpert(obs, allow_none=True),
                                       budget=b, select="mean_gain",
                                       ig_candidates=args.ig_candidates,
                                       ig_iters=args.ig_iters, seed=seed)
                    got[f"b{b}"].append(mae(o["acc"], c["true_acc"]))
            payload[name][mode] = dict(
                t_b0_ms=float(np.mean(secs)), beta_b0=float(np.mean(betas)),
                iters_b0=float(np.mean(iters)),
                **{k: [float(np.mean(v)), float(np.std(v))] for k, v in got.items()})
            print(f"    {mode:13s}"
                  + "".join(f"{np.mean(got[k]):8.2f}+-{np.std(got[k]):<5.2f}"
                            for k in cols)
                  + f"{np.mean(secs):>11.1f}{np.mean(betas):>10.3f}"
                    f"{np.mean(iters):>7.0f}", flush=True)

    mods = {}
    for n in payload:
        mods.setdefault(n.split("/")[0], []).append(n)

    print("\n" + "=" * 92)
    print(f"  MEAN over the {len(mods)} modalities")
    print("=" * 92)
    print(f"  {'mode':13s}" + "".join(f"{k:>12s}" for k in cols)
          + f"{'t_b0(ms)':>11s}{'iters':>8s}")
    agg = {}
    for mode in MODES:
        per_mod = {m: {f: float(np.mean([payload[n][mode][f][0] if f in cols
                                         else payload[n][mode][f] for n in names]))
                       for f in cols + ["t_b0_ms", "iters_b0", "beta_b0"]}
                   for m, names in mods.items()}
        agg[mode] = dict(per_modality=per_mod,
                         **{f: float(np.mean([per_mod[m][f] for m in per_mod]))
                            for f in cols + ["t_b0_ms", "iters_b0", "beta_b0"]})
        print(f"  {mode:13s}" + "".join(f"{agg[mode][k]:>12.2f}" for k in cols)
              + f"{agg[mode]['t_b0_ms']:>11.1f}{agg[mode]['iters_b0']:>8.0f}")

    print(f"\n  {'mode':13s}" + "".join(f"{m:>12s}" for m in mods) + "   (b0 per modality)")
    for mode in MODES:
        print(f"  {mode:13s}" + "".join(f"{agg[mode]['per_modality'][m]['b0']:>12.2f}"
                                        for m in mods))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as h:
        json.dump(dict(table=payload, aggregate=agg, metadata=vars(args)), h,
                  indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
