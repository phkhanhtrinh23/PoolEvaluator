"""Test the coverage/Hoeffding bound and the two effective sample sizes on REAL data.

Both derivations are proved in
``pool_text2sql_crowd_source_mixture/pdfs/{coverage_hoeffding_derivation.md,
MTM08_analysis.md}``; `pooleval/theory.py` implements them and `tests/test_theory.py`
verifies the algebra. This script asks the empirical questions the proofs cannot
answer, on the real SynSQL calibration probes and the real Text2SQL targets:

  E1  What are the geometry numbers of the selection Stage 1 actually makes?
      (coverage c, concentration ESS n_eff, weight-mismatch TV, positive weights r)
  E2  Section 16 says the current UNIFORM calibration average is the wrong estimator
      and the target-MATCHED weighted average is the right one. Is it, in practice?
  E3  Does the two-term bound hold on real data, and how much slack does it carry?
      What smoothness constant would have to be true for it to fail?
  E4  MTM08 says the anchor's ESS is s+2 with s = a0 * n0 and MAP weight s/(N+s).
      Sweep the discount a0 -- including the coverage-derived value the paper uses --
      against downstream accuracy. The theory does NOT say similarity is the right
      discount; this measures whether it is.
  E5  Item-scaled (s_beta = a0 n0) vs pair-scaled (s_beta = a0 J n0) pool-level
      strength, prior location held fixed. Section 5 argues J is nominal.
  E6  Sweep the dataset budget K. Coverage is monotone in K; the bound need not be,
      because n_eff can fall as coverage rises (Section 14).
  E7  Greedy coverage vs the brute-force optimum vs the distance-ranked top-k that
      Stage 1 currently uses. Where does alpha_K actually sit?
  E8  The load-bearing assumption itself (Section 8.1): is expected correctness
      actually smooth in this embedding? Bin every target-calibration pair by
      embedding distance and see whether the accuracy gap grows with distance the
      way a Lipschitz constant says it must.
  E9  An IN-DISTRIBUTION control. E1-E7 run at coverage ~0.57, far below anything
      the bound needs, so a valid-but-vacuous bound proves nothing either way. Split
      each target benchmark in half and calibrate on its own other half: same
      machinery, but now the smoothness and coverage preconditions have a chance of
      holding. This separates "the theorem does not help" from "SynSQL is too far
      from the target for any theorem to help".

    python experiments/run_ess_coverage.py                 # all of it, cached
    python experiments/run_ess_coverage.py --rebuild       # refit TF-IDF
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import Config, PoolEval, metrics, theory as th          # noqa: E402
from pooleval.data.simulator import PoolRun                           # noqa: E402
from pooleval.new_formulation import collision_agreement_em           # noqa: E402
from zoo.config import ZooConfig, ARTIFACT_ROOT                       # noqa: E402
from zoo.datasets import load_split                                   # noqa: E402
from synsql.config import RESULTS, ARTIFACTS, TOP_K, SEED             # noqa: E402

TARGETS = ["spider", "bird", "bird_minidev", "sqlflow", "spider2local"]
DELTA = 0.05
CACHE = os.path.join(ARTIFACTS, "ess_coverage_cache.npz")


# --------------------------------------------------------------------------- #
#  Setup: item-level embeddings for calibration probes and targets             #
# --------------------------------------------------------------------------- #
def build_cache(rebuild=False, log=print):
    """cosine[ds] = [N_target, n_cal] item-level cosine, plus per-item correctness.

    The probe store holds per-ITEM correctness (`C[M, 25]` and the SynSQL record id
    of each probe item), so we can drop to the item level the theory needs: Stage 1
    only ever used the subset-level average. Nothing here consumes a target label."""
    if os.path.exists(CACHE) and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        log(f"[cache] {CACHE}")
        return {k: z[k].item() if z[k].dtype == object and z[k].shape == () else z[k]
                for k in z.files}

    from sklearn.preprocessing import normalize
    from synsql.subsets import load_built
    from synsql.retrieve import SubsetRetriever, target_item_text

    recs, parts = load_built()
    ret = SubsetRetriever(recs, parts["db"], log=log)

    store = np.load(os.path.join(ARTIFACTS, "synsql_probes.npz"),
                    allow_pickle=True)["store"].item()
    subs = sorted(store)

    # flatten every GRADABLE probe item into one calibration pool
    rec_idx, sub_of, Ycols = [], [], []
    for s in subs:
        C = store[s]["C"]
        ok = C[0] >= 0
        for j, iid in enumerate(store[s]["ids"]):
            if not ok[j]:
                continue
            rec_idx.append(int(iid.split("-")[-1]))
            sub_of.append(s)
            Ycols.append(C[:, j].astype(float))
    Y = np.stack(Ycols, axis=1)                       # [M, n_cal]
    phi_cal = ret.X[np.asarray(rec_idx)]              # sparse, already L2-normalized
    log(f"[cal] {Y.shape[1]} gradable probe items over {len(subs)} subsets, "
        f"M={Y.shape[0]}")

    out = dict(Y=Y, sub_of=np.asarray(sub_of), subsets=np.asarray(subs),
               n_cal=Y.shape[1],
               # only inner products are ever needed downstream, and [n_cal, n_cal]
               # is tiny next to the [n_cal, 28300] feature block
               gram_cal=np.asarray((phi_cal @ phi_cal.T).todense()))
    for ds in TARGETS:
        p = os.path.join(ARTIFACT_ROOT, f"poolrun_{ds}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        N = int(d["true_class"].shape[1])
        items = load_split(ds, "dev", N, seed=SEED)
        if len(items) != N:
            log(f"[warn] {ds}: {len(items)} items vs poolrun N={N}, skipping")
            continue
        Xt = normalize(ret.vec.transform([target_item_text(i) for i in items]))
        out[f"cos_{ds}"] = np.asarray((Xt @ phi_cal.T).todense())
        out[f"acc_{ds}"] = np.asarray(d["true_acc"], dtype=float)
        # target-vs-target geometry and per-item correctness, for E8/E9 only. These
        # DO use target labels -- to evaluate the assumptions, never to fit anything.
        out[f"self_{ds}"] = np.asarray((Xt @ Xt.T).todense())
        out[f"corr_{ds}"] = (np.asarray(d["true_class"]) == 0).astype(float)
        log(f"[tgt] {ds}: N={N} cos in "
            f"[{out['cos_'+ds].min():.3f},{out['cos_'+ds].max():.3f}]")
    os.makedirs(ARTIFACTS, exist_ok=True)
    np.savez(CACHE, **out)
    log(f"[saved] {CACHE}")
    return out


def load_run(ds):
    d = np.load(os.path.join(ARTIFACT_ROOT, f"poolrun_{ds}.npz"), allow_pickle=True)
    N = d["true_class"].shape[1]
    return PoolRun(d["true_class"], d["true_acc"], d["group"], d["prior"],
                   d["prior_sigma"], np.zeros(N), np.zeros(N), d["verifier_guess"],
                   d["verifier_guess"] == 0, d["true_class"].shape[0], N,
                   int(np.asarray(d["group"]).max()) + 1)


def with_prior(run, prior, sigma):
    return PoolRun(run.true_class, run.true_acc, run.group, np.asarray(prior),
                   np.asarray(sigma), run.b, run.phi, run.verifier_guess,
                   run.verifier_correct, run.M, run.N, run.n_groups)


def sigma_from_strength(p, s):
    """Invert `collision.py`'s s = p(1-p)/sd^2 so a prior STRENGTH can be handed to
    the sigma-parameterized estimator. s = 0 -> an effectively flat anchor."""
    v = np.clip(np.asarray(p, float) * (1 - np.asarray(p, float)), 1e-4, None)
    return np.sqrt(v / max(s, 1e-6)) if s > 0 else np.full_like(v, 1e3)


# --------------------------------------------------------------------------- #
#  Selection helpers                                                           #
# --------------------------------------------------------------------------- #
def subset_blocks(cos, sub_of, subsets):
    """B[r, x] = dataset r's best match for target x, and A[r, x] the winning item."""
    S = (1.0 + cos) / 2.0                                # [N, n_cal]
    R = len(subsets)
    B = np.empty((R, S.shape[0]))
    A = np.empty((R, S.shape[0]), dtype=int)
    for r, s in enumerate(subsets):
        col = np.flatnonzero(sub_of == s)
        loc = np.argmax(S[:, col], axis=1)
        A[r] = col[loc]
        B[r] = S[np.arange(S.shape[0]), A[r]]
    return S, B, A


def geometry(S, cols):
    """All Section 3/11/12/16 quantities for one selected calibration item set."""
    Sq = S[:, cols]
    assign, _ = th.best_match(Sq)
    w = th.match_weights(assign, len(cols))
    f_cov, c = th.coverage(Sq)
    _, mean_d, mean_d2 = th.match_distances(Sq, assign)
    return dict(n=len(cols), N=Sq.shape[0], f_cov=f_cov, c=c,
                n_eff=th.n_eff(w), tv=th.tv_to_uniform(w),
                r_pos=int((w > 0).sum()), mean_d=mean_d, mean_d2=mean_d2, w=w,
                cols=np.asarray(cols))


def topk_by_distance(cos, sub_of, subsets, k):
    """Reproduce Stage 1's actual choice at the item level: rank candidate subsets by
    the cosine distance of their centroid to the target centroid, take the top k."""
    mu_t = cos.mean(axis=0)                      # mean cosine to each cal item
    score = np.array([mu_t[sub_of == s].mean() for s in subsets])
    return list(np.argsort(-score)[:k])


# --------------------------------------------------------------------------- #
def run(log=print):
    cache = build_cache(log=log)
    Y = cache["Y"]
    sub_of, subsets = cache["sub_of"], cache["subsets"]
    M, n_cal = Y.shape
    out = dict(delta=DELTA, n_cal=int(n_cal), M=int(M),
               n_subsets=int(len(subsets)), targets={})

    L_hat = None
    for ds in TARGETS:
        if f"cos_{ds}" not in cache:
            continue
        cos, true_acc = cache[f"cos_{ds}"], cache[f"acc_{ds}"]
        run_obj = load_run(ds)
        S, B, A = subset_blocks(cos, sub_of, subsets)
        res = {}
        # one pseudo-labelling, reused by every collision-EM variant below: the
        # agreement matrix must not change when only the anchor strength changes.
        pseudo = PoolEval(Config(real_data=True)).evaluate(run_obj)
        latent_hat = np.array([max(q, key=q.get) for q in pseudo["latent_post"]],
                              dtype=run_obj.true_class.dtype)
        C_agree = (run_obj.true_class == latent_hat[None, :]).astype(float)
        gamma_g = np.full(run_obj.n_groups, 0.15)

        # ---------------- E1: geometry of Stage 1's own selection --------------
        pick = topk_by_distance(cos, sub_of, subsets, TOP_K)
        cols_sel = np.flatnonzero(np.isin(sub_of, subsets[pick]))
        g = geometry(S, cols_sel)
        res["E1_geometry"] = {k: v for k, v in g.items() if k not in ("w", "cols")}
        res["E1_geometry"]["subsets"] = [int(subsets[i]) for i in pick]
        res["E1_geometry"]["raw_cos_mean"] = float(cos[:, cols_sel].max(axis=1).mean())

        # ---------------- E2: uniform vs target-matched prior ------------------
        Ysel = Y[:, cols_sel]
        pi_u = th.uniform_estimate(Ysel)
        pi_w = th.weighted_estimate(Ysel, g["w"])
        e2 = {}
        for name, p in [("uniform", pi_u), ("matched", pi_w)]:
            bias = float(np.mean(p - true_acc))
            pe = PoolEval(Config(real_data=True)).evaluate(
                with_prior(run_obj, p, sigma_from_strength(p, g["n"])))
            m = metrics.all_metrics(pe["acc"], true_acc)
            e2[name] = dict(prior=p.tolist(),
                            prior_mae=float(np.mean(np.abs(p - true_acc)) * 100),
                            prior_bias=bias * 100,
                            prior_mae_c=float(np.mean(np.abs(p - bias - true_acc)) * 100),
                            pe_MAE=float(m["MAE"]), pe_Kendall=float(m["Kendall"]),
                            pe_Top1=float(m["Top1"]))
        # Section 16.1: the gap between the two estimators must not exceed the TV term
        e2["gap_measured"] = float(np.max(np.abs(pi_u - pi_w)))
        e2["gap_bound_tv"] = float(g["tv"])
        e2["tv_bound_holds"] = bool(e2["gap_measured"] <= g["tv"] + 1e-9)
        res["E2_estimators"] = e2

        # ---------------- E3: does the bound hold, and by how much? ------------
        if L_hat is None:                              # calibration-only, target-free
            L_hat = th.estimate_lipschitz(cache["gram_cal"], Y, k=8, quantile=0.95)
            out["L_hat"] = L_hat.tolist()
            log(f"[L] kNN-smoothed 95th-pct Lipschitz proxy per model: "
                f"min {L_hat.min():.3f} med {np.median(L_hat):.3f} max {L_hat.max():.3f}")
        e3 = dict(L_hat=L_hat.tolist(), per_model=[])
        for j in range(M):
            err_w = float(abs(pi_w[j] - true_acc[j]))
            err_u = float(abs(pi_u[j] - true_acc[j]))
            Bw = th.bound_weighted(L_hat[j], g["c"], g["n_eff"], DELTA)
            Bws = th.bound_weighted(L_hat[j], g["c"], g["n_eff"], DELTA,
                                    mean_d=g["mean_d"])
            Bu = th.bound_uniform(L_hat[j], g["c"], g["n"], g["tv"], DELTA)
            Br = th.bound_realized(L_hat[j], g["c"], g["n_eff"], run_obj.N, DELTA)
            e3["per_model"].append(dict(
                model=j, err_weighted=err_w, err_uniform=err_u,
                B_weighted=Bw, B_weighted_sharp=Bws, B_uniform=Bu, B_realized=Br,
                holds_w=bool(err_w <= Bw), holds_u=bool(err_u <= Bu),
                holds_realized=bool(err_w <= Br),
                slack_w=Bw - err_w,
                L_required=th.required_lipschitz(err_w, g["c"], g["n_eff"], DELTA)))
        e3["hold_rate_weighted"] = float(np.mean([p["holds_w"] for p in e3["per_model"]]))
        e3["hold_rate_uniform"] = float(np.mean([p["holds_u"] for p in e3["per_model"]]))
        e3["n_eff_achieved"] = g["n_eff"]
        e3["n_eff_required_eps1.0"] = [
            th.required_neff(L_hat[j], g["c"], DELTA, 1.0) for j in range(M)]
        e3["n_eff_required_eps0.2"] = [
            th.required_neff(L_hat[j], g["c"], DELTA, 0.2) for j in range(M)]
        e3["feasible_max_neff"] = int(min(g["n"], run_obj.N))
        res["E3_bound"] = e3

        # ---------------- E4: the discount a0 -> s -> ESS -> accuracy ----------
        n0 = g["n"]
        e4 = []
        for a0 in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, g["c"]]:
            s = a0 * n0
            pe = PoolEval(Config(real_data=True)).evaluate(
                with_prior(run_obj, pi_w, sigma_from_strength(pi_w, s)))
            m = metrics.all_metrics(pe["acc"], true_acc)
            ce = collision_agreement_em(C_agree, run_obj.group, gamma_g, pi_w, s,
                                        beta_init=0.7, beta_strength=s)
            mc = metrics.all_metrics(ce["alpha"], true_acc)
            e4.append(dict(a0=float(a0), s=float(s), ess=th.prior_ess(s),
                           map_weight=th.map_weight(s, run_obj.N),
                           pe_MAE=float(m["MAE"]), pe_Kendall=float(m["Kendall"]),
                           ce_MAE=float(mc["MAE"]), ce_Kendall=float(mc["Kendall"]),
                           is_coverage_choice=bool(a0 == g["c"])))
        res["E4_discount"] = e4

        # ---------------- E5: item-scaled vs pair-scaled s_beta ----------------
        s_item = g["c"] * n0
        e5 = []
        for tag, sb in [("s_beta = a0 n0 (item)", s_item),
                        ("s_beta = a0 J n0 (pair)", s_item * M),
                        ("s_beta = 0", 0.0)]:
            ce = collision_agreement_em(C_agree, run_obj.group, gamma_g, pi_w, s_item,
                                        beta_init=0.7, beta_strength=sb)
            m = metrics.all_metrics(ce["alpha"], true_acc)
            e5.append(dict(scaling=tag, s_beta=float(sb), beta=float(ce["beta"]),
                           MAE=float(m["MAE"]), Kendall=float(m["Kendall"]),
                           ess_beta=th.prior_ess(sb)))
        res["E5_beta_scaling"] = e5

        # ---------------- E6: budget sweep -- is the bound monotone in K? ------
        e6 = []
        for K in range(1, 11):
            pk = topk_by_distance(cos, sub_of, subsets, K)
            ck = np.flatnonzero(np.isin(sub_of, subsets[pk]))
            gk = geometry(S, ck)
            pw = th.weighted_estimate(Y[:, ck], gk["w"])
            pu = th.uniform_estimate(Y[:, ck])
            e6.append(dict(K=K, n=gk["n"], c=gk["c"], n_eff=gk["n_eff"],
                           tv=gk["tv"], r_pos=gk["r_pos"],
                           B_med=float(np.median([
                               th.bound_weighted(L_hat[j], gk["c"], gk["n_eff"], DELTA)
                               for j in range(M)])),
                           prior_mae_matched=float(np.mean(np.abs(pw - true_acc)) * 100),
                           prior_mae_uniform=float(np.mean(np.abs(pu - true_acc)) * 100)))
        res["E6_budget"] = e6

        # ---------------- E7: greedy vs optimum vs distance top-k --------------
        e7 = []
        for K in (2, 3):
            gr, tr = th.greedy_cover(B, K)
            opt_set, opt = th.optimal_cover(B, K)
            dk = topk_by_distance(cos, sub_of, subsets, K)
            f_dist = float(B[dk].max(axis=0).sum())
            gg = geometry(S, np.flatnonzero(np.isin(sub_of, subsets[gr])))
            gd = geometry(S, np.flatnonzero(np.isin(sub_of, subsets[dk])))
            e7.append(dict(
                K=K, alpha_K=th.alpha_K(K), f_opt=opt, f_greedy=tr[-1],
                f_distance_topk=f_dist, ratio_greedy=tr[-1] / opt,
                ratio_distance=f_dist / opt,
                greedy_meets_alphaK=bool(tr[-1] >= th.alpha_K(K) * opt - 1e-9),
                n_eff_greedy=gg["n_eff"], n_eff_distance=gd["n_eff"],
                mae_greedy=float(np.mean(np.abs(
                    th.weighted_estimate(Y[:, gg["cols"]], gg["w"]) - true_acc)) * 100),
                mae_distance=float(np.mean(np.abs(
                    th.weighted_estimate(Y[:, gd["cols"]], gd["w"]) - true_acc)) * 100)))
        res["E7_greedy"] = e7


        # ---------------- E8: is expected correctness smooth at all? -----------
        # Section 8.1 assumes |p_j(x) - p_j(z)| <= L_j ||phi(x) - phi(z)||. That is a
        # statement about EXPECTED correctness, so estimate the two expectations by
        # bin means: slice every target-calibration pair by distance and compare the
        # two sides' mean correctness within each slice. If the gap does not grow
        # with distance, the transfer term is not measuring anything real.
        Yt = cache[f"corr_{ds}"]                       # [M, N] target correctness
        D_all = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cos))       # [N, n_cal]
        qs = np.quantile(D_all, np.linspace(0, 1, 11))
        e8 = dict(bins=[], spearman=[])
        xi, zi = np.meshgrid(np.arange(cos.shape[0]), np.arange(n_cal), indexing="ij")
        for b in range(10):
            m = (D_all >= qs[b]) & (D_all <= qs[b + 1] if b == 9 else D_all < qs[b + 1])
            if m.sum() < 50:
                continue
            xs, zs = xi[m], zi[m]
            gaps = [float(abs(Yt[j][xs].mean() - Y[j][zs].mean())) for j in range(M)]
            dbar = float(D_all[m].mean())
            e8["bins"].append(dict(bin=b, d_mean=dbar, n_pairs=int(m.sum()),
                                   gap_median=float(np.median(gaps)),
                                   L_implied=float(np.median(gaps)) / max(dbar, 1e-9)))
        if len(e8["bins"]) > 2:
            from scipy.stats import spearmanr
            dv = [x["d_mean"] for x in e8["bins"]]
            gv = [x["gap_median"] for x in e8["bins"]]
            e8["spearman"] = float(spearmanr(dv, gv).correlation)
            e8["L_lower_bound"] = float(max(x["L_implied"] for x in e8["bins"]))
        res["E8_smoothness"] = e8

        # ---------------- E9: in-distribution control --------------------------
        # Calibrate on half of the target benchmark and evaluate on the other half.
        # Nothing about the machinery changes -- only whether coverage and smoothness
        # have any chance of holding.
        self_cos = cache[f"self_{ds}"]
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(self_cos.shape[0])
        half = len(perm) // 2
        cal_i, tgt_i = np.sort(perm[:half]), np.sort(perm[half:])
        S9 = (1.0 + self_cos[np.ix_(tgt_i, cal_i)]) / 2.0
        a9, _ = th.best_match(S9)
        w9 = th.match_weights(a9, len(cal_i))
        f9, c9 = th.coverage(S9)
        _, md9, _ = th.match_distances(S9, a9)
        Ycal, Ytgt = Yt[:, cal_i], Yt[:, tgt_i]
        acc9 = Ytgt.mean(axis=1)
        pu9, pw9 = th.uniform_estimate(Ycal), th.weighted_estimate(Ycal, w9)
        L9 = th.estimate_lipschitz(self_cos[np.ix_(cal_i, cal_i)], Ycal, k=8)
        ne9 = th.n_eff(w9)
        e9 = dict(n_cal=int(len(cal_i)), N=int(len(tgt_i)), c=c9, n_eff=ne9,
                  tv=th.tv_to_uniform(w9), mean_d=md9,
                  L_hat=float(np.median(L9)),
                  mae_uniform=float(np.mean(np.abs(pu9 - acc9)) * 100),
                  mae_matched=float(np.mean(np.abs(pw9 - acc9)) * 100),
                  B_med=float(np.median([th.bound_weighted(L9[j], c9, ne9, DELTA)
                                         for j in range(M)])),
                  err_med=float(np.median(np.abs(pw9 - acc9))),
                  n_eff_required_eps0_2=float(np.median(
                      [th.required_neff(L9[j], c9, DELTA, 0.2) for j in range(M)])),
                  hold_rate=float(np.mean([abs(pw9[j] - acc9[j])
                                           <= th.bound_weighted(L9[j], c9, ne9, DELTA)
                                           for j in range(M)])))
        res["E9_in_distribution"] = e9

        out["targets"][ds] = res
        _report(ds, res, true_acc, log)

    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, "ess_coverage.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    log(f"\n[saved] {p}")
    _summary(out, log)
    return out


# --------------------------------------------------------------------------- #
def _report(ds, r, true_acc, log):
    g = r["E1_geometry"]
    log(f"\n=== {ds}  (true EX mean {true_acc.mean():.3f}) ===")
    log(f"E1  N={g['N']}  n={g['n']}  c={g['c']:.4f} (raw cos {g['raw_cos_mean']:.4f})"
        f"  n_eff={g['n_eff']:.2f}  positive weights={g['r_pos']}/{g['n']}"
        f"  TV(uniform,matched)={g['tv']:.4f}")
    e2 = r["E2_estimators"]
    log(f"E2  {'estimator':10s}{'priorMAE':>10s}{'bias':>9s}{'MAE_c':>8s}"
        f"{'PE MAE':>9s}{'PE Ken':>8s}")
    for k in ("uniform", "matched"):
        v = e2[k]
        log(f"    {k:10s}{v['prior_mae']:>10.2f}{v['prior_bias']:>+9.2f}"
            f"{v['prior_mae_c']:>8.2f}{v['pe_MAE']:>9.2f}{v['pe_Kendall']:>8.2f}")
    log(f"    max|uniform-matched|={e2['gap_measured']:.4f} <= TV={e2['gap_bound_tv']:.4f}"
        f"  -> {e2['tv_bound_holds']}")
    e3 = r["E3_bound"]
    ew = [p["err_weighted"] for p in e3["per_model"]]
    Bw = [p["B_weighted"] for p in e3["per_model"]]
    Lr = [p["L_required"] for p in e3["per_model"]]
    log(f"E3  bound holds {e3['hold_rate_weighted']*100:.0f}% (weighted) / "
        f"{e3['hold_rate_uniform']*100:.0f}% (uniform);  median B={np.median(Bw):.3f} "
        f"vs median |err|={np.median(ew):.3f}")
    log(f"    n_eff achieved {e3['n_eff_achieved']:.2f}; required for B<=1: "
        f"{np.median(e3['n_eff_required_eps1.0']):.1f} (median), for B<=0.2: "
        f"{np.median(e3['n_eff_required_eps0.2']):.1f}; feasible max "
        f"{e3['feasible_max_neff']}")
    log(f"    L that would make the bound tight: median {np.median(Lr):.3f} "
        f"vs L_hat median {np.median(e3['L_hat']):.3f}")
    log("E4  " + f"{'a0':>6s}{'s':>8s}{'ESS':>8s}{'MAPw':>7s}"
        f"{'PE MAE':>9s}{'PE Ken':>8s}{'CE MAE':>9s}")
    for e in r["E4_discount"]:
        tag = " <- coverage" if e["is_coverage_choice"] else ""
        log(f"    {e['a0']:>6.3f}{e['s']:>8.1f}{e['ess']:>8.1f}{e['map_weight']:>7.3f}"
            f"{e['pe_MAE']:>9.2f}{e['pe_Kendall']:>8.2f}{e['ce_MAE']:>9.2f}{tag}")
    log("E5  " + f"{'s_beta scaling':26s}{'s_beta':>9s}{'beta':>7s}{'MAE':>8s}{'Ken':>7s}")
    for e in r["E5_beta_scaling"]:
        log(f"    {e['scaling']:26s}{e['s_beta']:>9.1f}{e['beta']:>7.3f}"
            f"{e['MAE']:>8.2f}{e['Kendall']:>7.2f}")
    log("E6  " + f"{'K':>3s}{'n':>5s}{'c':>8s}{'n_eff':>8s}{'TV':>7s}{'B_med':>8s}"
        f"{'MAE_m':>8s}{'MAE_u':>8s}")
    for e in r["E6_budget"]:
        log(f"    {e['K']:>3d}{e['n']:>5d}{e['c']:>8.4f}{e['n_eff']:>8.2f}"
            f"{e['tv']:>7.3f}{e['B_med']:>8.3f}{e['prior_mae_matched']:>8.2f}"
            f"{e['prior_mae_uniform']:>8.2f}")
    log("E7  " + f"{'K':>3s}{'alpha_K':>9s}{'greedy/opt':>12s}{'dist/opt':>10s}"
        f"{'neff_g':>8s}{'neff_d':>8s}{'MAE_g':>8s}{'MAE_d':>8s}")
    for e in r["E7_greedy"]:
        log(f"    {e['K']:>3d}{e['alpha_K']:>9.3f}{e['ratio_greedy']:>12.4f}"
            f"{e['ratio_distance']:>10.4f}{e['n_eff_greedy']:>8.2f}"
            f"{e['n_eff_distance']:>8.2f}{e['mae_greedy']:>8.2f}{e['mae_distance']:>8.2f}")
    e8 = r["E8_smoothness"]
    log("E8  " + f"{'d_mean':>8s}{'pairs':>9s}{'acc gap':>9s}{'L implied':>11s}")
    for b in e8["bins"]:
        log(f"    {b['d_mean']:>8.3f}{b['n_pairs']:>9d}{b['gap_median']:>9.4f}"
            f"{b['L_implied']:>11.4f}")
    log(f"    spearman(distance, accuracy gap) = {e8.get('spearman', float('nan')):+.3f}"
        f"   -> smoothness {'consistent' if e8.get('spearman', -1) > 0 else 'NOT SUPPORTED'}")
    e9 = r["E9_in_distribution"]
    log(f"E9  in-distribution control: n={e9['n_cal']} N={e9['N']}  c={e9['c']:.4f}"
        f"  n_eff={e9['n_eff']:.2f}  L_hat={e9['L_hat']:.3f}")
    log(f"    prior MAE uniform {e9['mae_uniform']:.2f}  matched {e9['mae_matched']:.2f}"
        f"   B_med {e9['B_med']:.3f} vs |err| {e9['err_med']:.3f}"
        f"   holds {e9['hold_rate']*100:.0f}%")


def _summary(out, log):
    T = out["targets"]
    if not T:
        return
    log("\n" + "=" * 78 + "\nSUMMARY over " + ", ".join(T) + "\n" + "=" * 78)
    log(f"{'target':14s}{'c':>8s}{'n_eff':>8s}{'TV':>7s}"
        f"{'MAE_unif':>10s}{'MAE_match':>11s}{'PE_unif':>9s}{'PE_match':>10s}")
    for ds, r in T.items():
        g, e = r["E1_geometry"], r["E2_estimators"]
        log(f"{ds:14s}{g['c']:>8.4f}{g['n_eff']:>8.2f}{g['tv']:>7.3f}"
            f"{e['uniform']['prior_mae']:>10.2f}{e['matched']['prior_mae']:>11.2f}"
            f"{e['uniform']['pe_MAE']:>9.2f}{e['matched']['pe_MAE']:>10.2f}")
    du = np.mean([r["E2_estimators"]["matched"]["prior_mae"]
                  - r["E2_estimators"]["uniform"]["prior_mae"] for r in T.values()])
    dp = np.mean([r["E2_estimators"]["matched"]["pe_MAE"]
                  - r["E2_estimators"]["uniform"]["pe_MAE"] for r in T.values()])
    log(f"\nmatched - uniform:  prior MAE {du:+.2f} pts,  PoolEval MAE {dp:+.2f} pts")
    hold = np.mean([r["E3_bound"]["hold_rate_weighted"] for r in T.values()])
    log(f"weighted bound holds on {hold*100:.1f}% of model/target pairs")
    log("\nSection 8.1 smoothness -- spearman(pair distance, accuracy gap):")
    for ds, r in T.items():
        log(f"  {ds:14s}{r['E8_smoothness'].get('spearman', float('nan')):+8.3f}")
    log("\nIn-distribution control (calibrate on the target's own other half):")
    log(f"  {'target':14s}{'c':>8s}{'n_eff':>8s}{'MAE_unif':>10s}{'MAE_match':>11s}"
        f"{'B_med':>8s}{'|err|':>8s}")
    for ds, r in T.items():
        e = r["E9_in_distribution"]
        log(f"  {ds:14s}{e['c']:>8.4f}{e['n_eff']:>8.2f}{e['mae_uniform']:>10.2f}"
            f"{e['mae_matched']:>11.2f}{e['B_med']:>8.3f}{e['err_med']:>8.3f}")
    d9 = np.mean([r["E9_in_distribution"]["mae_matched"]
                  - r["E9_in_distribution"]["mae_uniform"] for r in T.values()])
    log(f"  matched - uniform, in distribution: {d9:+.2f} pts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.rebuild and os.path.exists(CACHE):
        os.remove(CACHE)
    run()
