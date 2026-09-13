"""Test the validated-EM reformulation: measured `e` and `gamma`, plus the i-EM
judge loop, against the estimators already in the repo.

Three domains, run in this order:

    --domain text2sql   real Text-to-SQL zoo (10 LLMs x 7 provenance groups)
    --domain vision     image classification, MNIST -> {USPS, SVHN}
    --domain graph      node classification, citation-network shifts

Two protocols for where the labeled statistics come from:

  A "source"  the genuine pre-built labeled split -- Spider/BIRD train, or the
              source-domain validation set for vision/graph. No target label is
              touched. This is the deployable setting.
  B "holdout" a labeled slice of the target items themselves, with the estimator
              scored on the disjoint remainder. This is the in-distribution
              control: it says what the statistics are worth when the domain gap
              is removed, and it is the only protocol available for the datasets
              whose source generations cannot be reproduced offline.

Every number is a measurement; nothing is copied from the derivation.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import Config, PoolEval, metrics                       # noqa: E402
from pooleval.new_formulation import (CollisionAwareNewFormulationPoolEval,  # noqa: E402
                                      NewFormulationPoolEval)
from pooleval.domains.collision import alpha_effective_size          # noqa: E402
from pooleval.estimators import (horvitz_thompson, ht_variance,             # noqa: E402
                                 inverse_variance_fuse, model_assisted,
                                 model_assisted_variance)
from pooleval.validated_em import (LabeledStatistics, LatentPlan, NoisyExpert,  # noqa: E402
                                   OracleExpert, excess_collision, hard_labels,
                                   run_validation, validated_em,
                                   wrong_collision_matrix)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Building the labeled statistics                                             #
# --------------------------------------------------------------------------- #
def build_stats(true_class_src, group, prior, gamma_mode="model_wrong",
                update_rule="average", temp_scope="validated_all"):
    """Measure `e` and `gamma` on a labeled split.

    `e` needs no pseudo-label (it compares answers against gold), so it is measured
    first and then used to discount the votes that produce the split's pseudo-labels,
    which is what `gamma` is conditioned on. Same order the deployed loop uses.
    """
    tc = np.asarray(true_class_src)
    e_excess = excess_collision(wrong_collision_matrix(tc), group)
    yhat = hard_labels(LatentPlan(tc, e_excess).posterior(prior))
    return LabeledStatistics(tc, yhat, group=group, gamma_mode=gamma_mode,
                             update_rule=update_rule, temp_scope=temp_scope)


def mae(estimate, truth):
    return 100.0 * float(np.mean(np.abs(np.asarray(estimate) - np.asarray(truth))))


# --------------------------------------------------------------------------- #
#  One dataset / shift                                                         #
# --------------------------------------------------------------------------- #
def evaluate_case(name, obs, true_acc, group, prior, prior_sigma, tc_labeled,
                  budgets=(0, 5, 10, 20, 40), expert_accuracy=1.0, seed=0,
                  ig_candidates=30, ig_iters=5, verbose=True, ablations=True,
                  prior_ess_cap=True, prior_ess=None, acq_seeds=5):
    """Score every method on one (labeled split, target split) pair."""
    obs = np.asarray(obs)
    M, N = obs.shape
    prior = np.asarray(prior, dtype=float)
    strength = (float(prior_ess) if prior_ess is not None
                else _strength(prior, prior_sigma, cap=N if prior_ess_cap else None))
    cfg = Config(real_data=True, M=M, N=N, n_groups=int(np.max(group)) + 1)
    rows, diag = {}, {}

    rows["prior only (no target data)"] = mae(prior, true_acc)

    run = _pool_run(obs, true_acc, group, prior, prior_sigma)
    old = PoolEval(cfg).evaluate(run, obs=obs)
    rows["PoolEval-SQL (current paper)"] = mae(old["acc"], true_acc)

    nf = NewFormulationPoolEval(cfg).evaluate(run, obs=obs, pseudo_out=old)
    rows["binary agreement EM (gamma=1)"] = mae(nf["acc"], true_acc)

    stats = build_stats(tc_labeled, group, prior)
    gamma_group = _group_collision(tc_labeled, group, stats)
    coll = CollisionAwareNewFormulationPoolEval(
        cfg, gamma_group, strength, beta_init=stats.pseudo_accuracy,
        beta_strength=120.0).evaluate(run, obs=obs, pseudo_out=old)
    rows["collision EM (group gamma, .tex)"] = mae(coll["acc"], true_acc)

    diag.update(M=M, N=N, n_labeled=int(np.asarray(tc_labeled).shape[1]),
                alpha_strength=strength, beta_labeled=stats.pseudo_accuracy,
                gamma_labeled=stats.gamma.tolist(),
                gamma_conditional=stats.conditional_gamma().tolist(),
                e_mean_offdiag=float(stats.e[~np.eye(M, dtype=bool)].mean()),
                e_excess_mean=float(stats.e_excess().mean()),
                true_acc=np.asarray(true_acc).tolist(),
                pool_true_mean=float(np.mean(true_acc)))

    def fresh():
        return build_stats(tc_labeled, group, prior)

    def expert_for(allow_none=False):
        if expert_accuracy >= 1.0:
            return OracleExpert(obs, allow_none=allow_none)
        return NoisyExpert(obs, accuracy=expert_accuracy, seed=seed,
                           allow_none=allow_none)

    base = validated_em(obs, fresh(), prior, strength, max_iters=200)
    rows["validated EM (no judge)"] = mae(base["acc"], true_acc)
    diag["beta_fitted_no_judge"] = float(base["beta"])
    diag["em_iters_no_judge"] = int(base["n_iters"])

    # The fourth variant lets the expert REJECT every candidate. On a pool whose
    # accuracy is low, the highest-information items are exactly the ones where no
    # model is right, so a judge restricted to choosing among the pool's answers
    # spends most of its budget with nothing to pick -- the candidate-coverage gap.
    variants = [("info_gain", False), ("entropy", False), ("random", False),
                ("info_gain", True)]
    for budget in budgets:
        if budget == 0:
            continue
        for select, allow_none in variants:
            st = fresh()
            t0 = time.time()
            out = run_validation(obs, st, prior, strength,
                                 expert=expert_for(allow_none),
                                 budget=budget, select=select,
                                 ig_candidates=ig_candidates, ig_iters=ig_iters,
                                 seed=seed)
            suffix = ", may reject all" if allow_none else ""
            label = f"validated EM + judge b={budget} [{select}{suffix}]"
            rows[label] = mae(out["acc"], true_acc)
            diag[label] = dict(confirmed=out["confirmed"], overruled=out["overruled"],
                               abstained=out["abstained"], beta=float(out["beta"]),
                               entropy=float(out["entropy"]),
                               seconds=round(time.time() - t0, 1))
            if verbose:
                print(f"    {label:52s} MAE {rows[label]:6.2f}   "
                      f"confirm {out['confirmed']:3d} overrule {out['overruled']:3d} "
                      f"abstain {out['abstained']:3d}  ({time.time()-t0:.0f}s)",
                      flush=True)

    b = max(budgets)
    rows.update(_acquisition_block(obs, true_acc, group, prior, strength, tc_labeled,
                                   b, lambda: expert_for(True), fresh, seed,
                                   ig_candidates, ig_iters, print if verbose else None,
                                   n_seeds=acq_seeds))
    if ablations:
        rows.update(_ablations(obs, true_acc, group, prior, strength, tc_labeled,
                               b, expert_accuracy, seed, ig_candidates, ig_iters))
    return rows, diag


def design_estimates(out, obs, refit):
    """Turn a run's AUDIT sample into design-based accuracy estimates.

    Only the pilot items qualify.  They were drawn by simple random sampling, so their
    inclusion probability is exactly ``pilot / N`` and Horvitz-Thompson applies.  The
    actively chosen items are deliberately excluded: adaptive selection without
    replacement leaves no clean marginal inclusion probability, and an estimator cannot
    undo a bias whose size it does not know.  They still earn their keep by improving the
    model that ``model_assisted`` leans on.

    CROSS-FITTING is not optional here.  The fitted model has the audit items PINNED to
    the expert's answer, so its predictions there are the labels themselves: residuals
    would be exactly zero, the correction term would vanish, and ``model_assisted``
    would silently collapse into the plain model average -- the very bias it exists to
    remove.  ``refit`` therefore re-solves with the audit labels withheld, so the
    prediction on an audited item never saw that item's label.
    """
    audit = list(out.get("audit") or [])
    if not audit or out.get("audit_pi") in (None, 0):
        return {}
    idx = np.asarray(sorted(audit), dtype=int)
    answers = out["answers"]
    values = np.stack([(obs[:, i] == answers[i]).astype(float) for i in idx], axis=1)
    pi = np.full(len(idx), float(out["audit_pi"]))
    N = obs.shape[1]

    held_out = {k: v for k, v in out["constraints"].items() if k not in set(audit)}
    model_tau = refit(held_out)

    ht = horvitz_thompson(values, pi, N)
    ma = model_assisted(model_tau, values, idx, pi, N)
    fused, _ = inverse_variance_fuse(
        [ma, out["acc"]],
        [np.clip(model_assisted_variance(model_tau, values, idx, pi, N), 1e-9, None),
         np.clip(out["a_sigma"] ** 2, 1e-9, None)])
    return dict(ht=ht, model_assisted=ma, fused=fused,
                ht_var=float(np.mean(ht_variance(values, pi, N))),
                ma_var=float(np.mean(model_assisted_variance(model_tau, values, idx,
                                                             pi, N))),
                n_audit=len(idx))


def _acquisition_block(obs, true_acc, group, prior, strength, tc_labeled, budget,
                       expert_factory, fresh, seed, ig_candidates, ig_iters, log,
                       n_seeds=5):
    """The head-to-head the whole design question turns on.

    Four acquisition rules under one budget, plus -- for the designs that carry valid
    inclusion probabilities -- the design-based estimators that are unbiased whatever the
    model believes.

    Averaged over ``n_seeds`` repetitions, which is not optional: an audit of ~20 items
    has a standard error near 0.1 on a proportion, so a single draw says almost nothing
    about a 2-point difference between acquisition rules.
    """
    rows = {}
    half = max(1, budget // 2)
    settings = [
        ("IG [label entropy, Hung et al.]", dict(select="info_gain", pilot=0)),
        ("IG [accuracy variance, A_mu]", dict(select="mean_gain", pilot=0)),
        ("A_mu sampled (eps-mixed softmax)", dict(select="mean_gain_sampled", pilot=0)),
        ("random sampling", dict(select="random", pilot=0)),
        ("hybrid: random audit + A_mu", dict(select="mean_gain", pilot=half)),
        ("hybrid: random audit + IG", dict(select="info_gain", pilot=half)),
    ]
    for label, kw in settings:
        t0 = time.time()
        acc = {"model": [], "ht": [], "model_assisted": [], "fused": []}
        for rep in range(n_seeds):
            out = run_validation(obs, fresh(), prior, strength,
                                 expert=expert_factory(), budget=budget,
                                 ig_candidates=ig_candidates, ig_iters=ig_iters,
                                 seed=seed + 1000 * rep, **kw)
            acc["model"].append(mae(out["acc"], true_acc))

            def refit(held, _out=out):
                return validated_em(obs, fresh(), prior, strength, constraints=held,
                                    max_iters=200)["tau"]

            design = design_estimates(out, obs, refit)
            for key in ("ht", "model_assisted", "fused"):
                if key in design:
                    acc[key].append(mae(design[key], true_acc))
        rows[f"ACQ  {label}"] = float(np.mean(acc["model"]))
        for key, name in (("ht", "Horvitz-Thompson"),
                          ("model_assisted", "model-assisted"),
                          ("fused", "inverse-variance fused")):
            if acc[key]:
                rows[f"ACQ  {label} -> {name}"] = float(np.mean(acc[key]))
        if log:
            extra = "".join(f"  {n} {np.mean(acc[k]):6.2f}"
                            for k, n in (("ht", "HT"), ("model_assisted", "MA"),
                                         ("fused", "fused")) if acc[k])
            log(f"    {label:38s} model {rows[f'ACQ  {label}']:6.2f}"
                f" (sd {np.std(acc['model']):.2f}){extra}"
                f"   ({time.time() - t0:.0f}s, {n_seeds} seeds)")
    return rows


def _ablations(obs, true_acc, group, prior, strength, tc_labeled, budget,
               expert_accuracy, seed, ig_candidates, ig_iters):
    """Turn one ingredient off at a time, everything else held fixed."""
    def expert():
        return (OracleExpert(obs) if expert_accuracy >= 1.0
                else NoisyExpert(obs, accuracy=expert_accuracy, seed=seed))

    def stats(**kw):
        return build_stats(tc_labeled, group, prior, **kw)

    out = {}
    common = dict(budget=budget, select="info_gain", ig_candidates=ig_candidates,
                  ig_iters=ig_iters, seed=seed)

    out["ABL  - e discount (votes undiscounted)"] = mae(
        run_validation(obs, stats(), prior, strength, expert=expert(),
                       use_discount=False, **common)["acc"], true_acc)
    # model_wrong is now the DEFAULT, so the ablation is the other two readings.
    out["ABL  gamma mode = both_wrong"] = mae(
        run_validation(obs, stats(gamma_mode="both_wrong"), prior, strength,
                       expert=expert(), **common)["acc"], true_acc)
    out["ABL  gamma mode = pseudo_wrong"] = mae(
        run_validation(obs, stats(gamma_mode="pseudo_wrong"), prior, strength,
                       expert=expert(), **common)["acc"], true_acc)
    out["ABL  update rule = pooled counts"] = mae(
        run_validation(obs, stats(update_rule="counts"), prior, strength,
                       expert=expert(), **common)["acc"], true_acc)
    out["ABL  temp from latest item only"] = mae(
        run_validation(obs, stats(temp_scope="latest"), prior, strength,
                       expert=expert(), **common)["acc"], true_acc)
    out["ABL  refresh on overrule only"] = mae(
        run_validation(obs, stats(), prior, strength, expert=expert(),
                       update_stats_on_confirm=False, **common)["acc"], true_acc)
    out["ABL  clamp confirmed items too"] = mae(
        run_validation(obs, stats(), prior, strength, expert=expert(),
                       clamp_on_confirm=True, **common)["acc"], true_acc)
    out["ABL  expert accuracy 0.80"] = mae(
        run_validation(obs, stats(), prior, strength,
                       expert=NoisyExpert(obs, accuracy=0.80, seed=seed),
                       **common)["acc"], true_acc)
    out["ABL  expert accuracy 0.80, may reject all"] = mae(
        run_validation(obs, stats(), prior, strength,
                       expert=NoisyExpert(obs, accuracy=0.80, seed=seed,
                                          allow_none=True),
                       **common)["acc"], true_acc)
    return out


# --------------------------------------------------------------------------- #
#  Plumbing                                                                    #
# --------------------------------------------------------------------------- #
def _strength(prior, prior_sigma, cap=None):
    """Beta-anchor strength implied by the declared prior sd: ``s = p(1-p)/sd^2``.

    ``cap`` bounds it so the anchor can never outweigh the target evidence. This is not
    cosmetic. Declaring ``sd`` as the binomial standard error of the labeled split makes
    ``s`` equal to that split's size, which asserts that the source accuracy transfers
    with sampling-error-only uncertainty -- false under domain shift, and the failure
    mode is severe: on MNIST -> SVHN the source prior is 86 accuracy points wrong, so an
    anchor of ESS 3000 against 600 target items pins every estimate to a catastrophically
    wrong number. It is the same point ``docs/ess_coverage.md`` makes about the prior's
    usable effective sample size being bounded by the TRANSFER term, not the source
    sample size.
    """
    sigma = np.asarray(prior_sigma, dtype=float)
    num = np.asarray(prior) * (1.0 - np.asarray(prior))
    ok = (sigma > 0) & (num > 0)
    s = float(np.median(num[ok] / sigma[ok] ** 2)) if ok.any() else 120.0
    return float(min(s, cap)) if cap is not None else s


def _pool_run(obs, true_acc, group, prior, prior_sigma):
    from pooleval.data.simulator import PoolRun
    M, N = obs.shape
    vg = np.full(N, 10 ** 9, dtype=np.int64)
    return PoolRun(true_class=obs, true_acc=np.asarray(true_acc), group=np.asarray(group),
                   prior=np.asarray(prior), prior_sigma=np.asarray(prior_sigma),
                   b=np.zeros(N), phi=np.zeros(N), verifier_guess=vg,
                   verifier_correct=np.zeros(N, dtype=bool), M=M, N=N,
                   n_groups=int(np.max(group)) + 1)


def _group_collision(tc_labeled, group, stats, smoothing=1.0):
    """The .tex's per-group collision rate, for the estimator it belongs to."""
    tc = np.asarray(tc_labeled)
    G = int(np.max(group)) + 1
    yhat = hard_labels(LatentPlan(tc, stats.e_excess()).posterior(
        np.full(tc.shape[0], 0.7)))
    hits = np.zeros(G)
    elig = np.zeros(G)
    pseudo_wrong = yhat != 0
    for m in range(tc.shape[0]):
        g = int(group[m])
        valid = (tc[m] != 0) & pseudo_wrong
        elig[g] += valid.sum()
        hits[g] += (valid & (tc[m] == yhat)).sum()
    return (hits + smoothing) / (elig + 2.0 * smoothing)


def split_indices(N, frac, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    cut = int(round(frac * N))
    return np.sort(order[:cut]), np.sort(order[cut:])


# --------------------------------------------------------------------------- #
#  Domain 1: Text-to-SQL                                                       #
# --------------------------------------------------------------------------- #
TEXT2SQL = ["spider", "bird", "sqlflow", "bird_minidev", "spider2local"]
SOURCE_OK = ["spider", "bird"]        # source split reproduces offline, ids all match
SRC_CACHE = os.path.join(ROOT, "zoo_artifacts", "source_true_class.npz")


def text2sql_source_matrices(datasets=SOURCE_OK, rebuild=False, log=print):
    """Re-execute the cached source-split SQL to recover its equivalence classes.

    `poolrun_*.npz` stores only the source ACCURACY (the seen prior), not the class
    ids, and `e` needs to know WHICH wrong answer each model gave. Re-running the
    saved generations against the live SQLite DBs recovers the full matrix; no API
    call is involved. The result is cached because it takes a couple of minutes.
    """
    if os.path.exists(SRC_CACHE) and not rebuild:
        z = np.load(SRC_CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    import json as _json
    from zoo.build import true_class_matrix
    from zoo.config import ARTIFACT_ROOT, ZooConfig
    from zoo.datasets import load_split
    cfg = ZooConfig()
    names = [m.name for m in cfg.manifest]
    out = {}
    for ds in datasets:
        items = load_split(ds, "source", cfg.n_source, cfg.seed)
        preds = _json.load(open(os.path.join(ARTIFACT_ROOT, f"preds_{ds}_src.json")))
        covered = [it for it in items if all(it["id"] in preds[n] for n in names)]
        log(f"  [{ds}] executing {len(covered)} labeled source items x {len(names)} models")
        tc, _ = true_class_matrix(covered, names, preds, cfg.exec_timeout,
                                  log=lambda *a: None)
        out[ds] = tc
        log(f"  [{ds}] source EX = {(tc == 0).mean(axis=1).round(3).tolist()}")
    np.savez(SRC_CACHE, **out)
    return out


def run_text2sql(args, log=print):
    from zoo.new_formulation_real import load_run, subset_run
    src = text2sql_source_matrices(rebuild=args.rebuild_source, log=log)
    payload = {}
    for ds in args.datasets:
        run, names = load_run(ds)
        payload[ds] = {}
        if ds in src:
            log(f"\n[text2sql/{ds}] protocol A (source split, no target labels), "
                f"labeled N={src[ds].shape[1]}, target N={run.N}")
            rows, diag = evaluate_case(
                ds, run.true_class, run.true_acc, run.group, run.prior,
                run.prior_sigma, src[ds], budgets=args.budgets,
                expert_accuracy=args.expert_accuracy, seed=args.seed,
                ig_candidates=args.ig_candidates, ig_iters=args.ig_iters,
                ablations=args.ablations, prior_ess=args.prior_ess,
                acq_seeds=args.acq_seeds)
            payload[ds]["source"] = dict(rows=rows, diagnostics=diag, names=names.tolist())
            _print_rows(f"{ds} / protocol A (source split)", rows, log)

        cal, ev = split_indices(run.N, args.holdout_frac, seed=args.seed)
        log(f"\n[text2sql/{ds}] protocol B (target holdout), "
            f"labeled N={len(cal)}, eval N={len(ev)}")
        sub = subset_run(run, ev)
        rows, diag = evaluate_case(
            ds, sub.true_class, sub.true_acc, run.group, run.prior, run.prior_sigma,
            run.true_class[:, cal], budgets=args.budgets,
            expert_accuracy=args.expert_accuracy, seed=args.seed,
            ig_candidates=args.ig_candidates, ig_iters=args.ig_iters,
            ablations=args.ablations, prior_ess=args.prior_ess,
                acq_seeds=args.acq_seeds)
        payload[ds]["holdout"] = dict(rows=rows, diagnostics=diag,
                                      names=names.tolist())
        _print_rows(f"{ds} / protocol B (target holdout)", rows, log)
    return payload


# --------------------------------------------------------------------------- #
#  Domains 2 and 3: image classification and node classification               #
# --------------------------------------------------------------------------- #
POOL_CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")
VISION_SHIFTS = [("mnist", "usps"), ("mnist", "svhn")]
GRAPH_SHIFTS = ["AC", "AD", "CA", "CD", "DA", "DC"]


def _subsample(n, k, seed):
    if k is None or k >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def _domain_case(pool, args, log, tag):
    from pooleval.domains.adapter import encode_classes
    pred = np.asarray(pool["pred"])
    gold = np.asarray(pool["gold"])
    group = np.asarray(pool["group"])
    prior = np.asarray(pool["prior"])
    pred_s = (np.asarray(pool["pred_s"]) if "pred_s" in pool
              else np.asarray(pool["prob_s"]).argmax(-1))
    gold_s = np.asarray(pool["src_gold_val"])

    ti = _subsample(pred.shape[1], args.n_target, args.seed)
    si = _subsample(pred_s.shape[1], args.n_labeled, args.seed)
    obs = encode_classes(pred[:, ti], gold[ti])
    tc_src = encode_classes(pred_s[:, si], gold_s[si])
    true_acc = (obs == 0).mean(axis=1)
    # NOT the binomial SE of the labeled split: see _strength(). The domain ports
    # declare the seen prior's uncertainty as a fixed sd (pooleval.Config.prior_noise,
    # and pooleval.domains.adapter.from_predictions), which is what every other
    # vision/graph experiment in this repo uses.
    prior_sigma = np.full(len(prior), args.prior_noise, dtype=float)

    log(f"\n[{tag}] labeled N={len(si)}, target N={len(ti)}, M={pred.shape[0]}, "
        f"K={int(np.asarray(pool['n_classes']))}, true acc "
        f"{true_acc.min():.3f}-{true_acc.max():.3f}")
    rows, diag = evaluate_case(tag, obs, true_acc, group, prior, prior_sigma,
                               tc_src, budgets=args.budgets,
                               expert_accuracy=args.expert_accuracy, seed=args.seed,
                               ig_candidates=args.ig_candidates,
                               ig_iters=args.ig_iters, ablations=args.ablations,
                               prior_ess=args.prior_ess, acq_seeds=args.acq_seeds)
    diag["n_classes"] = int(np.asarray(pool["n_classes"]))
    _print_rows(tag, rows, log)
    return dict(rows=rows, diagnostics=diag)


def run_vision(args, log=print):
    payload = {}
    for srcname, dst in VISION_SHIFTS:
        path = os.path.join(POOL_CACHE, f"vision_{srcname}_{dst}.npz")
        if not os.path.exists(path):
            log(f"  [skip] no cached pool at {path}")
            continue
        z = np.load(path, allow_pickle=True)
        payload[f"{srcname}->{dst}"] = _domain_case(
            {k: z[k] for k in z.files}, args, log, f"vision/{srcname}->{dst}")
    return payload


def run_graph(args, log=print):
    payload = {}
    for shift in GRAPH_SHIFTS:
        path = os.path.join(POOL_CACHE, f"graph_{shift}.npz")
        if not os.path.exists(path):
            log(f"  [skip] no cached pool at {path}")
            continue
        z = np.load(path, allow_pickle=True)
        payload[shift] = _domain_case({k: z[k] for k in z.files}, args, log,
                                      f"graph/{shift}")
    return payload


# --------------------------------------------------------------------------- #
#  Reporting                                                                   #
# --------------------------------------------------------------------------- #
def _print_rows(title, rows, log=print):
    log(f"\n  {title}  --  MAE in accuracy points (lower is better)")
    log("  " + "-" * 62)
    best = min(rows.values())
    for name, value in rows.items():
        mark = "  <-- best" if value == best else ""
        log(f"  {name:52s} {value:7.2f}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", nargs="+",
                    default=["text2sql", "vision", "graph"],
                    choices=["text2sql", "vision", "graph"])
    ap.add_argument("--datasets", nargs="+", default=TEXT2SQL)
    ap.add_argument("--budgets", nargs="+", type=int, default=[0, 5, 10, 20, 40])
    ap.add_argument("--holdout-frac", type=float, default=0.4)
    ap.add_argument("--expert-accuracy", type=float, default=1.0)
    ap.add_argument("--ig-candidates", type=int, default=30)
    ap.add_argument("--ig-iters", type=int, default=5)
    ap.add_argument("--n-target", type=int, default=400)
    ap.add_argument("--n-labeled", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--acq-seeds", type=int, default=5,
                    help="repetitions to average the acquisition head-to-head over")
    ap.add_argument("--prior-ess", type=float, default=None,
                    help="override the Beta anchor strength s directly (0 = no anchor)")
    ap.add_argument("--prior-noise", type=float, default=0.07,
                    help="declared sd of the seen prior for the vision/graph ports")
    ap.add_argument("--no-ablations", dest="ablations", action="store_false")
    ap.add_argument("--rebuild-source", action="store_true")
    ap.add_argument("--out", default=os.path.join(RESULTS, "validated_em.json"))
    args = ap.parse_args()

    payload = {}
    started = time.time()
    for domain in args.domain:
        print(f"\n{'=' * 72}\n  DOMAIN: {domain}\n{'=' * 72}", flush=True)
        t0 = time.time()
        payload[domain] = {"text2sql": run_text2sql, "vision": run_vision,
                           "graph": run_graph}[domain](args)
        print(f"\n[{domain}] done in {time.time() - t0:.0f}s", flush=True)
        # checkpoint after every domain so a long run is never all-or-nothing
        _save(args.out, payload, args, started)


def _save(path, payload, args, started):
    payload["metadata"] = dict(
        budgets=args.budgets, holdout_frac=args.holdout_frac,
        expert_accuracy=args.expert_accuracy, ig_candidates=args.ig_candidates,
        ig_iters=args.ig_iters, n_target=args.n_target, n_labeled=args.n_labeled,
        seed=args.seed, total_seconds=round(time.time() - started, 1))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
