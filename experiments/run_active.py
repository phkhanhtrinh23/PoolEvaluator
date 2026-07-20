"""Active PoolEval-SQL -- label-efficient judge-in-the-loop evaluation (NEW).

Not part of the original paper. Formalizes and measures the CANDIDATE-COVERAGE
limitation of label-free consensus and shows a small, targeted expert (judge) budget
resolves it.

The limitation. PoolEval-SQL can only pick a latent answer from the result classes
the pool PRODUCED. When a near-clone clique unanimously emits the same WRONG result
and the correct answer is absent from the pool, consensus confidently credits the
wrong class: the clique's estimated accuracy inflates and the deployment ranking can
invert. No unlabeled agreement can invent the missing correct answer.

The fix. Spend a label-free judge (gpt-5-mini: reads the question, executes the SQL,
aligns result-to-question) on only the most ambiguous items, chosen by GREEDY
SUBMODULAR maximization (>= (1-1/e) OPT, Nemhauser-Wolsey-Fisher 1978); pin each
judged item as a hard EM constraint (possibly a class no model produced); re-solve
with incremental EM.

Experiments:
  A  candidate-gap DGP: pure PoolEval vs strategies at a fixed budget (illustrative).
  B  budget sweep 0/1/5/10% x {random, uncertainty, disagreement, ranking_impact,
     hybrid_submodular}; metrics vs true accuracy.
  C  abstention: how often the near-clone-collusion flag fires on trap items.

Run:  python experiments/run_active.py [--seeds 20]
Writes results/active_budget.json, results/active_illustrative.json.
"""
import argparse
import numpy as np

from _shared import (Config, PoolEval, metrics, aggregate, save_json, METRIC_KEYS,
                     print_table)
from pooleval import ActivePoolEval, ActiveConfig, SimulatedJudge
from pooleval.data.simulator import PoolRun, sigmoid


STRATEGIES = ["random", "uncertainty", "disagreement", "ranking_impact",
              "hybrid_submodular"]


# --------------------------------------------------------------------------- #
#  Candidate-coverage-failure DGP                                              #
# --------------------------------------------------------------------------- #
def dgp_candidate_gap(cfg, rng, rho=0.24, clique_size=3, clique_acc=0.46,
                      good_acc=0.70, verifier_fool=1.0):
    """A near-clone clique (group 0, `clique_size` members, LOW true accuracy) is
    made to WIN under label-free consensus by traps: on a fraction `rho` of items the
    whole clique emits the SAME wrong result and the correct answer is produced by no
    one. Consensus credits the clique -> its estimated accuracy inflates above the
    genuinely-better independent singletons, flipping the deployment choice.

    Group layout: group 0 = clique (clique_size near-clones); the remaining models are
    independent singletons in their own groups. Accuracies are FIXED (not resampled)
    so the ground-truth ranking is identical across seeds and the flip is a clean
    function of the judge budget: two 'good' singletons at `good_acc` should rank top;
    the clique sits at `clique_acc`; the rest spread in between. The verifier is
    systematically fooled on traps (`verifier_fool`), because a trap is by definition
    a wrong result that executes and looks plausible -- so the anchor cannot rescue
    candidate coverage.
    """
    M, N = cfg.M, cfg.N
    group = np.array([0] * clique_size + list(range(1, M - clique_size + 1)))
    G = int(group.max()) + 1

    # FIXED true accuracies: one clearly-best 'good' singleton on top, the clique low,
    # the rest of the independent singletons spread in between. A single unambiguous
    # top model makes Top1 a clean read of clique-recovery (not a near-tie coin flip).
    acc = np.linspace(0.52, 0.63, M)
    acc[:clique_size] = clique_acc
    good = [clique_size]                          # the model that SHOULD rank top
    acc[good[0]] = good_acc

    # Correctness is a direct Bernoulli at each model's accuracy (NO item-difficulty
    # coupling): keeps the effective accuracies equal to the nominal ones so the flip
    # margin is a clean, predictable function of the trap rate rho. b is kept only as
    # a mild difficulty input for the estimator's IRT term.
    b = rng.normal(0, 0.4, size=N)
    trap = rng.random(N) < rho                   # trap items (correct answer absent)

    true_class = np.zeros((M, N), dtype=np.int64)
    idio = 100000
    clique_wrong = 300000 + np.arange(N)         # the clique's shared wrong result
    for m in range(M):
        correct = rng.random(N) < acc[m]
        for i in range(N):
            if trap[i]:
                if m < clique_size:
                    true_class[m, i] = clique_wrong[i]     # clique colludes on wrong
                else:
                    # correct answer is ABSENT: every non-clique model is wrong with
                    # its OWN idiosyncratic result -> the pool cannot produce class 0.
                    true_class[m, i] = idio; idio += 1
            else:
                if correct[i]:
                    true_class[m, i] = 0
                else:
                    true_class[m, i] = idio; idio += 1
    true_acc = (true_class == 0).mean(axis=1)

    bias = rng.normal(0, cfg.prior_bias, size=M)
    prior = np.clip(true_acc + bias + rng.normal(0, cfg.prior_noise, size=M),
                    0.01, 0.99)
    prior_sigma = np.full(M, max(cfg.prior_noise, 1e-3))

    # verifier: independent channel on normal items, but FOOLED on traps (a wrong
    # query that runs and returns clique_wrong looks plausible) -> the anchor cannot
    # fix candidate coverage, which only a stronger judge can.
    vc = rng.random(N) < cfg.verifier_acc
    vguess = np.where(vc, 0, rng.integers(1, 1000, size=N))
    fooled = trap & (rng.random(N) < verifier_fool)
    vguess = np.where(fooled, clique_wrong, vguess)
    vc = vguess == 0
    phi = rng.uniform(0.2, 1.0, size=N)
    run = PoolRun(true_class, true_acc, group, prior, prior_sigma, b, phi,
                  vguess, vc, M, N, G)
    return run, dict(trap=trap, good=good, clique=list(range(clique_size)))


# --------------------------------------------------------------------------- #
#  Experiment A -- illustrative single comparison                              #
# --------------------------------------------------------------------------- #
def experiment_illustrative(seed=0):
    cfg = Config(real_data=True, seed=seed)      # DGP already gives obs classes
    rng = np.random.default_rng(2000 + seed)
    run, meta = dgp_candidate_gap(cfg, rng)
    good = meta["good"]

    clique = meta["clique"]
    base = PoolEval(cfg).evaluate(run)
    rows = {}
    rows["PoolEval-SQL (label-free)"] = _row(base, run, good, clique)
    for strat in STRATEGIES:
        out = ActivePoolEval(cfg, ActiveConfig(budget=80, rounds=8,
                                               strategy=strat)).run(run)
        rows["Active: " + strat] = _row(out, run, good, clique)
    return dict(rows=rows,
                truth=dict(true_acc=run.true_acc.tolist(),
                           true_top1=int(np.argmax(run.true_acc)),
                           good_models=good, clique=clique,
                           clique_true=round(float(run.true_acc[clique].mean()), 3),
                           good_true=round(float(run.true_acc[good].mean()), 3),
                           frac_trap=float(meta["trap"].mean())))


def _row(out, run, good, clique):
    m = metrics.all_metrics(out["acc"], run.true_acc)
    est_top1 = int(np.argmax(out["acc"]))
    return dict(MAE=round(m["MAE"], 2), Flip=round(m["Flip"], 3),
                Kendall=round(m["Kendall"], 3), Top1=m["Top1"],
                clique_est=round(float(out["acc"][clique].mean()), 3),
                good_est=round(float(out["acc"][good].mean()), 3),
                est_top1=est_top1, top1_is_good=int(est_top1 in good),
                judge_calls=int(out.get("judge_calls", 0)))


# --------------------------------------------------------------------------- #
#  Experiment B -- budget sweep x strategy                                     #
# --------------------------------------------------------------------------- #
def experiment_budget(seeds, budgets_pct=(0.0, 0.01, 0.05, 0.10)):
    N = Config().N
    budgets = [int(round(p * N)) for p in budgets_pct]
    # rows[strategy][budget] = list of metric dicts over seeds
    rows = {s: {b: [] for b in budgets} for s in STRATEGIES}
    pure = {b: [] for b in budgets}                 # b=0 pure PoolEval reference
    for s in range(seeds):
        cfg = Config(real_data=True, seed=s)
        rng = np.random.default_rng(2000 + s)
        run, meta = dgp_candidate_gap(cfg, rng)
        base = PoolEval(cfg).evaluate(run)
        base_m = metrics.all_metrics(base["acc"], run.true_acc)
        for b in budgets:
            pure[b].append(base_m)                  # budget-0 baseline is pure
        for strat in STRATEGIES:
            for b in budgets:
                if b == 0:
                    rows[strat][b].append(base_m)
                    continue
                rounds = min(b, 8)
                out = ActivePoolEval(cfg, ActiveConfig(budget=b, rounds=rounds,
                                                       strategy=strat)).run(run)
                rows[strat][b].append(
                    metrics.all_metrics(out["acc"], run.true_acc))
    agg = {s: {str(b): aggregate(rows[s][b]) for b in budgets} for s in STRATEGIES}
    return dict(budgets=budgets, budgets_pct=list(budgets_pct), N=N,
                strategies=agg,
                pure_reference={str(b): aggregate(pure[b]) for b in budgets})


# --------------------------------------------------------------------------- #
#  Experiment C -- abstention                                                  #
# --------------------------------------------------------------------------- #
def experiment_abstention(seeds):
    prec, rec, meffs = [], [], []
    for s in range(seeds):
        cfg = Config(real_data=True, seed=s)
        rng = np.random.default_rng(2000 + s)
        run, meta = dgp_candidate_gap(cfg, rng)
        ap = ActivePoolEval(cfg, ActiveConfig(budget=0, rounds=1))
        out = ap.run(run)
        ab = out["abstain"]
        flagged = ab["per_item"]
        trap = meta["trap"]
        if flagged.sum() > 0:
            prec.append(float((flagged & trap).sum() / flagged.sum()))
        rec.append(float((flagged & trap).sum() / max(1, trap.sum())))
        meffs.append(ab["Meff"])
    return dict(flag_precision_on_trap=float(np.mean(prec)) if prec else 0.0,
                flag_recall_on_trap=float(np.mean(rec)),
                mean_Meff=float(np.mean(meffs)),
                note="Per-item flag = full-consensus item backed by <2 provenance "
                     "groups (near-clone collusion). Meff is the effective "
                     "independent-model count on the gap DGP.")


# --------------------------------------------------------------------------- #
#  Printing                                                                    #
# --------------------------------------------------------------------------- #
def _print_illustrative(res):
    print("\n=== A. Candidate-gap DGP (single seed) ===")
    t = res["truth"]
    print(f"true top-1 = m{t['true_top1']} (good, true acc {t['good_true']}); "
          f"clique {t['clique']} (true acc {t['clique_true']}); "
          f"trap items = {t['frac_trap']:.0%} of N")
    hdr = ["MAE", "Flip", "Kendall", "Top1", "clique_est", "good_est",
           "est_top1", "top1_is_good", "judge_calls"]
    print(f"{'method':28s}" + "".join(f"{h:>13s}" for h in hdr))
    for name, r in res["rows"].items():
        print(f"{name:28s}" + "".join(f"{str(r[h]):>13s}" for h in hdr))


def _print_budget(res):
    print("\n=== B. Budget sweep on the candidate-gap DGP "
          f"(N={res['N']}, budgets {res['budgets']}) ===")
    for metric in ["Top1", "Flip", "MAE"]:
        arrow = "up" if metric == "Top1" else "down"
        print(f"\n-- {metric} ({arrow} better) vs judge budget --")
        head = ["strategy"] + [f"{b}({p:.0%})" for b, p in
                               zip(res["budgets"], res["budgets_pct"])]
        print("".join(f"{h:>16s}" for h in head))
        for strat, per_b in res["strategies"].items():
            cells = [strat]
            for b in res["budgets"]:
                m, ci = per_b[str(b)][metric]
                cells.append(f"{m:.2f}")
            print("".join(f"{c:>16s}" for c in cells))


def _print_abstention(res):
    print("\n=== C. Abstention (near-clone collusion flag on trap items) ===")
    print(f"mean M_eff on gap DGP     = {res['mean_Meff']:.2f}")
    print(f"flag precision on traps   = {res['flag_precision_on_trap']:.2f}")
    print(f"flag recall on traps      = {res['flag_recall_on_trap']:.2f}")


def main(seeds=20):
    print(f"[Active PoolEval-SQL] candidate-coverage experiments ({seeds} seeds)")

    illus = experiment_illustrative(seed=0)
    _print_illustrative(illus)
    save_json("active_illustrative.json", illus)

    budget = experiment_budget(seeds)
    _print_budget(budget)
    save_json("active_budget.json", budget)

    ab = experiment_abstention(seeds)
    _print_abstention(ab)
    save_json("active_abstention.json", ab)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    main(**vars(ap.parse_args()))
