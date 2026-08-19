"""Shared scoring for the domain ports (vision / graph).

Runs PoolEval and every portable baseline on a pool dict produced by
`pooleval.domains.{vision,graph}.build_pool`, and scores them against the
withheld target labels.
"""
import dataclasses, os, sys
import numpy as np
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pooleval.config import Config                                   # noqa: E402
from pooleval.inference import PoolEval                              # noqa: E402
from pooleval.domains.adapter import from_predictions                # noqa: E402
from pooleval.domains.baselines import confidence_baselines          # noqa: E402
from baselines import Independent, Majority, DawidSkene, AgreementLine  # noqa: E402

# B5 LLM-as-judge scores SQL text -- not portable to vision / graph.
POOL_BASELINES = [Independent(), Majority(), DawidSkene(), AgreementLine()]


def make_run_cfg(pool, **cfg_kw):
    run = from_predictions(pool["pred"], pool["gold"], pool["group"],
                           prior=pool["prior"], verifier_guess=pool["verifier_guess"])
    cfg = dataclasses.replace(Config(), real_data=True, M=run.M, N=run.N,
                              n_groups=run.n_groups, **cfg_kw)
    return run, cfg


def score_pool(pool, extra_variants=None):
    """-> {method: metrics} against the withheld gold labels."""
    run, cfg = make_run_cfg(pool)
    est = {"PoolEval": PoolEval(cfg).evaluate(run)["acc"]}
    for b in POOL_BASELINES:
        est[b.name] = np.asarray(b.evaluate(run, cfg))
    est.update(confidence_baselines(pool))
    for label, kw in (extra_variants or {}).items():
        r, c = make_run_cfg(pool, **kw)
        est[label] = PoolEval(c).evaluate(r)["acc"]

    truth = run.true_acc
    rows = {}
    for name, a in est.items():
        a = np.asarray(a, dtype=float)
        rows[name] = dict(mae=float(np.abs(a - truth).mean()),
                          bias=float((a - truth).mean()),
                          spearman=float(spearmanr(a, truth).statistic),
                          kendall=float(kendalltau(a, truth).statistic),
                          top1_correct=bool(int(np.argmax(a)) == int(np.argmax(truth))))
    return rows, truth


def print_rows(rows, indent="  "):
    for k, v in sorted(rows.items(), key=lambda kv: kv[1]["mae"]):
        print(f"{indent}{k:24s} MAE {v['mae']:.4f}  bias {v['bias']:+.4f}  "
              f"rho {v['spearman']:+.3f}  tau {v['kendall']:+.3f}")


def summarize(allrows, title):
    print(f"\n=== {title} ===")
    print(f"{'method':24s} {'MAE':>7} {'bias':>8} {'rho':>7} {'tau':>7} {'top1':>6}")
    summary = {}
    for k, vs in sorted(allrows.items(), key=lambda kv: np.mean([v["mae"] for v in kv[1]])):
        s = dict(mae=float(np.mean([v["mae"] for v in vs])),
                 bias=float(np.mean([v["bias"] for v in vs])),
                 spearman=float(np.mean([v["spearman"] for v in vs])),
                 kendall=float(np.mean([v["kendall"] for v in vs])),
                 top1=float(np.mean([v["top1_correct"] for v in vs])))
        summary[k] = s
        print(f"{k:24s} {s['mae']:7.4f} {s['bias']:+8.4f} {s['spearman']:+7.3f} "
              f"{s['kendall']:+7.3f} {s['top1']:6.2f}")
    return summary
