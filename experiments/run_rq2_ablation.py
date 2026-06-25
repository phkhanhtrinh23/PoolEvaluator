"""RQ2/RQ4 -- Breaking the gauge: anchor, verifier, correlation, kernel
(paper Table 'tab:ablation').

Removes PoolEval's requirement components one at a time on the same pools and
reports the change in accuracy MAE / flip / Top-1. Dropping the seen prior should
blow up absolute MAE (gauge ambiguity); exact-match kernel should cost MAE+flips.

  python experiments/run_rq2_ablation.py [--seeds 8]
"""
import argparse
import dataclasses
import numpy as np
from _shared import Config, PoolEval, simulate, metrics, aggregate, save_json, RESULTS


VARIANTS = {
    "PoolEval (full)":           dict(),
    "  - seen prior (req2)":     dict(use_prior=False),
    "  - exec verifier (req2)":  dict(use_verifier=False),
    "  - correlation (req3)":    dict(use_correlation=False),
    "  - graded kernel (req4)":  dict(kernel_level="L0"),
    "  - all (~ B3)":            dict(use_prior=False, use_verifier=False,
                                      use_correlation=False, kernel_level="L0",
                                      fusion="agreement_only"),
}


def main(seeds=8):
    rows = {k: [] for k in VARIANTS}
    for s in range(seeds):
        base = Config(seed=s)
        run = simulate(base)                      # same pool for all variants (paired)
        for name, overrides in VARIANTS.items():
            cfg = dataclasses.replace(base, **overrides)
            out = PoolEval(cfg).evaluate(run)
            rows[name].append(metrics.all_metrics(out["acc"], run.true_acc))
    agg = {k: aggregate(v) for k, v in rows.items()}

    print(f"\nRQ2: ablation on the diverse M=12 pool (mean +/- 95% CI over {seeds} "
          "seeds). Lower MAE/Flip better.")
    print("-" * 64)
    print(f"{'Variant':26s}{'MAE':>12s}{'Flip':>12s}{'Top1':>12s}")
    for name in VARIANTS:
        m = agg[name]
        print(f"{name:26s}{m['MAE'][0]:8.2f}    {m['Flip'][0]:8.3f}    "
              f"{m['Top1'][0]:8.2f}")
    save_json("rq2_ablation.json", {k: {mk: agg[k][mk] for mk in
              ["MAE", "Flip", "Top1"]} for k in agg})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    main(**vars(ap.parse_args()))
