"""Does execution across a SUITE of database instances change `e`, `gamma`, and the estimates?

The same pool, the same items and the same queries are graded under two kernels and
every downstream quantity is re-measured:

  single   one database instance, exact canonical result match. What the repo does.
  suite    the same, refined by row-sampled database instances: same class iff the same
           result on EVERY instance. See zoo/exec_suite.py.

The refinement is sound -- equivalent queries agree everywhere, so the suite can only
split classes one instance merged -- so this is not a question of whether it is safe but
of how much false collision the single instance was carrying.

Because the suite also re-grades CORRECTNESS -- a model that matched gold on the shipped
instance by coincidence is demoted -- the ground truth itself moves. Both gradings are
reported, and MAE against each, because comparing an estimate scored against one truth
with an estimate scored against another would be meaningless.

  python experiments/run_exec_suite.py --datasets spider --variants 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import Config, PoolEval, metrics                        # noqa: E402
from pooleval.new_formulation import CollisionAwareNewFormulationPoolEval  # noqa: E402
from pooleval.partitions import partition_quality                     # noqa: E402
from pooleval.validated_em import (OracleExpert, run_validation,      # noqa: E402
                                   validated_em, wrong_collision_matrix)
from experiments.run_validated_em import (_group_collision, _pool_run,  # noqa: E402
                                          _strength, build_stats, mae)
from zoo import exec_suite as es                                      # noqa: E402
from zoo.config import ARTIFACT_ROOT, ZooConfig                       # noqa: E402
from zoo.datasets import load_split                                   # noqa: E402

def _singleton_share(classes):
    """Fraction of models whose answer is shared with nobody -- how fragmented an item is,
    read straight off the partition rather than through an entropy."""
    _, counts = np.unique(classes, return_counts=True)
    return float((counts == 1).sum() / len(classes))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SCRATCH = os.environ.get(
    "POOLEVAL_SCRATCH",
    "/tmp/claude-1000/-home-trinh-Dropbox-thesis-pool-text2sql-eval-code/"
    "b9ee40bb-d014-423e-951d-8a67010c584a/scratchpad")
VARIANT_DIR = os.path.join(SCRATCH, "dbvariants")

SPLIT_FILE = {"dev": "preds_{ds}_dev.json", "source": "preds_{ds}_src.json"}


# --------------------------------------------------------------------------- #
#  Building both class matrices for one split                                  #
# --------------------------------------------------------------------------- #
def variants_for(items, k, keep_frac, max_db_mb, budget_mb, log):
    """Row-sampled instances per DATABASE, reused by every item on that database.

    Two hard limits, because the machine has ~10 GB free and BIRD's databases run to
    hundreds of megabytes each: skip any database above ``max_db_mb``, and stop once
    the cumulative copy budget is spent. A skipped database simply falls back to
    single-instance evidence, which is still sound -- just less discriminating -- and
    the coverage is reported rather than hidden.
    """
    spent = 0.0
    out, skipped = {}, []
    for path in sorted({it["db_path"] for it in items}):
        size = os.path.getsize(path) / 1e6
        if size > max_db_mb or spent + k * size > budget_mb:
            out[path] = []
            skipped.append((os.path.basename(path), round(size, 1)))
            continue
        folder = os.path.join(VARIANT_DIR, os.path.basename(os.path.dirname(path)))
        out[path] = es.build_variants(path, folder, k=k, keep_frac=keep_frac, log=log)
        spent += k * size
    if skipped:
        log(f"    [variants] skipped {len(skipped)} database(s) over the size/budget "
            f"cap: {skipped[:4]}{' ...' if len(skipped) > 4 else ''}")
    log(f"    [variants] {sum(len(v) for v in out.values())} instance(s) built, "
        f"{spent:.0f} MB")
    return out


def grade_split(ds, split, k, keep_frac, max_db_mb, budget_mb, log):
    """-> single-instance classes, suite classes, both correctness gradings, diagnostics."""
    cfg = ZooConfig()
    names = [m.name for m in cfg.manifest]
    n = cfg.n_target if split == "dev" else cfg.n_source
    items = load_split(ds, split, n, cfg.seed)
    preds = json.load(open(os.path.join(ARTIFACT_ROOT,
                                        SPLIT_FILE[split].format(ds=ds))))
    items = [it for it in items if all(it["id"] in preds[nm] for nm in names)]
    log(f"  [{ds}/{split}] {len(items)} items x {len(names)} models")
    vmap = variants_for(items, k, keep_frac, max_db_mb, budget_mb, log)

    M, N = len(names), len(items)
    single = np.zeros((M, N), dtype=np.int64)
    suite = np.zeros((M, N), dtype=np.int64)
    ok_single = np.zeros((M, N), dtype=bool)
    ok_suite = np.zeros((M, N), dtype=bool)
    n_informative = np.zeros(N, dtype=int)
    split_items = merged_items = 0

    for i, it in enumerate(items):
        sqls = [preds[nm][it["id"]] for nm in names]
        informative = es.informative_instances(it["gold_sql"], vmap[it["db_path"]],
                                               cfg.exec_timeout)
        n_informative[i] = len(informative)
        ev = es.item_evidence(sqls, it["gold_sql"], it["db_path"], informative,
                              cfg.exec_timeout)
        single[:, i] = es.single_instance_classes(ev)
        suite[:, i] = es.suite_classes(ev)
        ok_single[:, i], ok_suite[:, i] = es.correct_flags(ev)
        ne, nsm = len(set(single[:, i].tolist())), len(set(suite[:, i].tolist()))
        split_items += nsm > ne
        merged_items += nsm < ne

    # The single-instance partition is the one the suite must only ever REFINE. Pairwise
    # precision below 1 would mean the suite merged a pair the single instance kept
    # apart, which the soundness argument says is impossible -- so it is measured, not
    # assumed.
    quality = [partition_quality(suite[:, i], single[:, i]) for i in range(N)]
    diag = dict(
        n_items=N, n_models=M,
        mean_informative_variants=float(n_informative.mean()),
        items_with_no_variant=int((n_informative == 0).sum()),
        clusters_single=float(np.mean([len(set(single[:, i].tolist())) for i in range(N)])),
        clusters_suite=float(np.mean([len(set(suite[:, i].tolist())) for i in range(N)])),
        items_split=int(split_items), items_merged=int(merged_items),
        pair_precision_vs_single=float(np.mean([q["precision"] for q in quality])),
        pair_recall_vs_single=float(np.mean([q["recall"] for q in quality])),
        acc_single=ok_single.mean(axis=1).tolist(),
        acc_suite=ok_suite.mean(axis=1).tolist(),
        cells_demoted=int((ok_single & ~ok_suite).sum()),
        mean_overestimate=float((ok_single.mean(axis=1) - ok_suite.mean(axis=1)).mean()),
        ranking_changed=bool(not np.array_equal(np.argsort(-ok_single.mean(axis=1)),
                                                np.argsort(-ok_suite.mean(axis=1)))),
        singleton_share_single=float(np.mean([_singleton_share(single[:, i])
                                              for i in range(N)])),
        singleton_share_suite=float(np.mean([_singleton_share(suite[:, i])
                                             for i in range(N)])),
        names=names,
    )
    return single, suite, ok_single, ok_suite, diag


# --------------------------------------------------------------------------- #
#  Scoring both kernels through the whole estimator stack                      #
# --------------------------------------------------------------------------- #
def score(obs, tc_labeled, group, prior, prior_sigma, true_acc, budgets, seed,
          ig_candidates, ig_iters):
    M, N = obs.shape
    strength = _strength(prior, prior_sigma, cap=N)
    cfg = Config(real_data=True, M=M, N=N, n_groups=int(np.max(group)) + 1)
    rows = {"prior only": mae(prior, true_acc)}

    run = _pool_run(obs, true_acc, group, prior, prior_sigma)
    old = PoolEval(cfg).evaluate(run, obs=obs)
    rows["PoolEval-SQL"] = mae(old["acc"], true_acc)

    stats = build_stats(tc_labeled, group, prior)
    rows["collision EM (.tex)"] = mae(CollisionAwareNewFormulationPoolEval(
        cfg, _group_collision(tc_labeled, group, stats), strength,
        beta_init=stats.beta, beta_strength=120.0
    ).evaluate(run, obs=obs, pseudo_out=old)["acc"], true_acc)

    rows["validated EM (no judge)"] = mae(
        validated_em(obs, build_stats(tc_labeled, group, prior), prior, strength,
                     max_iters=200)["acc"], true_acc)
    for b in budgets:
        out = run_validation(obs, build_stats(tc_labeled, group, prior), prior,
                             strength, expert=OracleExpert(obs, allow_none=True),
                             budget=b, select="info_gain",
                             ig_candidates=ig_candidates, ig_iters=ig_iters, seed=seed)
        rows[f"validated EM + judge b={b}"] = mae(out["acc"], true_acc)

    e = wrong_collision_matrix(tc_labeled)
    off = ~np.eye(M, dtype=bool)
    stat = dict(e_offdiag_mean=float(e[off].mean()), e_offdiag_max=float(e[off].max()),
                gamma_mean=float(stats.gamma.mean()),
                gamma_conditional_mean=float(stats.conditional_gamma().mean()),
                beta=float(stats.beta))
    return rows, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["spider"])
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--keep-frac", type=float, default=0.6)
    ap.add_argument("--max-db-mb", type=float, default=150.0)
    ap.add_argument("--budget-mb", type=float, default=2500.0)
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 40])
    ap.add_argument("--ig-candidates", type=int, default=50)
    ap.add_argument("--ig-iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exec_suite.json"))
    args = ap.parse_args()

    cfg = ZooConfig()
    group = np.asarray(cfg.group_ids())
    payload = {}
    for ds in args.datasets:
        t0 = time.time()
        print(f"\n{'=' * 72}\n  {ds}\n{'=' * 72}", flush=True)
        src = grade_split(ds, "source", args.variants, args.keep_frac, args.max_db_mb,
                          args.budget_mb, print)
        tgt = grade_split(ds, "dev", args.variants, args.keep_frac, args.max_db_mb,
                          args.budget_mb, print)
        one_src, suite_src, ok_one_src, _, diag_src = src
        one_tgt, suite_tgt, ok_one_tgt, ok_suite_tgt, diag_tgt = tgt

        prior_single = ok_one_src.mean(axis=1)
        # Anchor strength is capped at N so the prior can never outweigh the target
        # evidence; see experiments/run_validated_em.py::_strength.
        prior_sigma = np.sqrt(np.clip(prior_single * (1 - prior_single), 1e-6, None)
                              / one_src.shape[1])
        truth = dict(single=ok_one_tgt.mean(axis=1), suite=ok_suite_tgt.mean(axis=1))

        table = {}
        for kernel, obs, labeled in (("single", one_tgt, one_src),
                                     ("suite", suite_tgt, suite_src)):
            for truth_name, true_acc in truth.items():
                rows, stat = score(obs, labeled, group, prior_single, prior_sigma,
                                   true_acc, args.budgets, args.seed,
                                   args.ig_candidates, args.ig_iters)
                table[f"{kernel} kernel / {truth_name} truth"] = rows
                table.setdefault("_stats", {})[f"{kernel} kernel"] = stat
        payload[ds] = dict(source=diag_src, target=diag_tgt, table=table,
                           seconds=round(time.time() - t0, 1))
        _print(ds, diag_tgt, table)

    payload["metadata"] = vars(args)
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"\n[saved] {args.out}")


def _print(ds, diag, table):
    print(f"\n  {ds}: kernel diagnostics on the target split")
    print(f"    informative variants / item      {diag['mean_informative_variants']:.2f}"
          f"   (items with none: {diag['items_with_no_variant']})")
    print(f"    clusters / item                  {diag['clusters_single']:.2f} single "
          f"-> {diag['clusters_suite']:.2f} suite")
    print(f"    items split / merged             {diag['items_split']} / {diag['items_merged']}")
    print(f"    pair precision vs single         {diag['pair_precision_vs_single']:.4f}"
          f"   (1.0 = the suite only ever refined, never merged)")
    print(f"    share of models alone in a class {diag['singleton_share_single']:.4f} single "
          f"-> {diag['singleton_share_suite']:.4f} suite")
    print(f"    single-instance acc overestimates suite acc by {diag['mean_overestimate']*100:.2f} pts"
          f"   ({diag['cells_demoted']} cells demoted, ranking changed: {diag['ranking_changed']})")
    stats = table.get("_stats", {})
    for kernel, st in stats.items():
        print(f"    [{kernel:16s}] e_offdiag {st['e_offdiag_mean']:.4f}  "
              f"gamma {st['gamma_mean']:.4f}  P(C=0|Z=0) {st['gamma_conditional_mean']:.4f}  "
              f"beta {st['beta']:.4f}")

    cols = [k for k in table if k != "_stats"]
    methods = list(table[cols[0]])
    width = max(len(m) for m in methods) + 2
    print(f"\n  {ds}: MAE in accuracy points (lower is better)")
    print("    " + " " * width + "".join(f"{c:>28s}" for c in cols))
    for m in methods:
        print(f"    {m:<{width}s}" + "".join(f"{table[c][m]:28.2f}" for c in cols))


if __name__ == "__main__":
    main()
