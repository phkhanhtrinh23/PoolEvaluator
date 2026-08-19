"""RQ-D3: WHY does PoolEval behave the way it does outside Text2SQL?

Three diagnostics on the cached domain pools.

  1. PRIOR WEIGHT COLLAPSE. Precision fusion weights the seen prior by
     1/prior_sigma^2 against 1/a_sigma^2 with a_sigma = sqrt(a(1-a)/N). Vision and
     graph targets have N in the thousands, so a_sigma -> 0 and the prior receives
     under 1% of the weight -- the one anchor that can correct a correlated pool is
     switched off exactly where the pool is most correlated.

  2. VERIFIER STRENGTH IS AN MAE-vs-RANKING KNOB. `verifier_strength` is added on
     the ACCURACY scale to a score whose other terms are bounded in (0,1), so it is
     worth several whole models. Raising it shrinks every model toward the verifier:
     the common-mode bias falls (MAE improves) while between-model spread collapses
     (ranking degrades).

  3. THE POSTERIOR CANNOT SELF-DIAGNOSE. Under shift the hard latent answer is wrong
     on a large fraction of items, yet the posterior remains highly peaked -- so
     confidence carries no signal about the estimator's own bias.
"""
import argparse, dataclasses, json, os, sys, warnings
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pooleval.config import Config                            # noqa: E402
from pooleval.inference import PoolEval                       # noqa: E402
from experiments._domain_eval import make_run_cfg             # noqa: E402

CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")


def load_pools(kind):
    out = {}
    for f in sorted(os.listdir(CACHE)):
        if f.startswith(kind) and f.endswith(".npz"):
            z = np.load(os.path.join(CACHE, f), allow_pickle=True)
            out[f[len(kind) + 1:-4]] = {k: z[k] for k in z.files}
    return out


def sweep(pools, **kw):
    ma, rh = [], []
    for p in pools.values():
        run, cfg = make_run_cfg(p, **kw)
        a = PoolEval(cfg).evaluate(run)["acc"]
        ma.append(np.abs(a - run.true_acc).mean())
        rh.append(spearmanr(a, run.true_acc).statistic)
    return float(np.mean(ma)), float(np.mean(rh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="graph", choices=["graph", "vision"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pools = load_pools(a.kind)
    if not pools:
        sys.exit(f"no cached {a.kind} pools in {CACHE} -- run the domain experiment first")
    res = {}

    print("### 1. prior weight under precision fusion")
    print(f"{'pool':12s} {'N':>7} {'a_sigma':>9} {'w_prior':>9}")
    pw = {}
    for name, p in pools.items():
        N = len(p["gold"]); asig = float(np.sqrt(0.25 / N))
        w = (1 / Config().prior_noise ** 2) / ((1 / Config().prior_noise ** 2) + 1 / asig ** 2)
        pw[name] = w
        print(f"{name:12s} {N:7d} {asig:9.4f} {w * 100:8.2f}%")
    res["prior_weight"] = pw

    print("\n### 2. component ablation")
    print(f"{'variant':38s} {'MAE':>8} {'rho':>8}")
    abl = {}
    for lbl, kw in [("PoolEval (full)", {}),
                    ("  -verifier", dict(use_verifier=False)),
                    ("  -correlation", dict(use_correlation=False)),
                    ("  -prior", dict(use_prior=False)),
                    ("  fusion=prior_only", dict(fusion="prior_only")),
                    ("  fusion=agreement_only", dict(fusion="agreement_only"))]:
        m, r = sweep(pools, **kw); abl[lbl.strip()] = dict(mae=m, spearman=r)
        print(f"{lbl:38s} {m:8.4f} {r:+8.3f}")
    res["ablation"] = abl

    print("\n### 3. verifier_strength: MAE vs ranking")
    print(f"{'strength':>10} {'MAE':>8} {'rho':>8}")
    vs = {}
    for s in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        m, r = sweep(pools, verifier_strength=s); vs[s] = dict(mae=m, spearman=r)
        print(f"{s:10.2f} {m:8.4f} {r:+8.3f}")
    res["verifier_strength"] = vs

    print("\n### 4. is the pseudo-groundtruth trustworthy, and does it know?")
    print(f"{'pool':12s} {'latent_hat acc':>15} {'mean max-post':>14} {'true bias':>11}")
    diag = {}
    for name, p in pools.items():
        run, cfg = make_run_cfg(p)
        o = PoolEval(cfg).evaluate(run)
        mp = float(np.mean([max(q.values()) for q in o["latent_post"]]))
        lat = np.array([max(q, key=q.get) for q in o["latent_post"]])
        bias = float((o["acc"] - run.true_acc).mean())
        diag[name] = dict(latent_acc=float((lat == 0).mean()), max_post=mp, bias=bias)
        print(f"{name:12s} {(lat == 0).mean():15.4f} {mp:14.4f} {bias:+11.4f}")
    c = np.corrcoef([d["max_post"] for d in diag.values()],
                    [abs(d["bias"]) for d in diag.values()])[0, 1]
    print(f"corr(confidence, |bias|) = {c:+.3f}   "
          f"(a working self-diagnostic would be strongly NEGATIVE)")
    res["latent_diag"] = diag
    res["conf_bias_corr"] = float(c)

    out = a.out or f"results/domain_diagnostics_{a.kind}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
