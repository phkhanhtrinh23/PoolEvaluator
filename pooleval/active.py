"""Active PoolEval-SQL -- label-efficient, judge-in-the-loop pool evaluation.

Pure PoolEval-SQL is label-free but it can only ever choose among the result classes
the pool actually produced. When every model is wrong -- and especially when a
near-clone bloc agrees on the SAME wrong result -- the correct class is ABSENT from
the candidate set, so consensus confidently credits a wrong answer and the ranking
can invert. No amount of unlabeled agreement can invent the missing correct answer;
this is the *candidate-coverage* limitation, distinct from (and deeper than) the
collusion the paper already discusses.

Active PoolEval-SQL keeps the framework unchanged and adds a strong, label-free
*judge* (gpt-5-mini): it reads the natural-language question, reads/executes each
candidate SQL, and aligns result-to-question -- it is *not* handed a gold label. The
judge is expensive, so we spend it only on the most ambiguous items, picked by
GREEDY SUBMODULAR MAXIMIZATION of a facility-location objective (Nemhauser-Wolsey-
Fisher 1978: greedy >= (1-1/e) OPT for a monotone submodular function under a
cardinality budget, and the bound is tight). Each judged item enters EM as a hard
constraint P(z_i = judge answer) = 1 -- possibly a class no model produced -- and we
re-solve with INCREMENTAL EM (warm-started, not restarted). When consensus rests on
too few independent provenance groups and the verifier is weak, the estimator
ABSTAINS ("insufficient independent evidence -- expert validation required") instead
of reporting a confident wrong ranking.
"""
import heapq
from dataclasses import dataclass
import numpy as np

from . import kernel as kernelmod
from .inference import PoolEval


# --------------------------------------------------------------------------- #
#  Judge (label-free strong verifier)                                          #
# --------------------------------------------------------------------------- #
class SimulatedJudge:
    """Perfect label-free verifier used in the simulator (accuracy 1.0 on the items
    it is asked about). It returns the equivalence class of the CORRECT result:

      * the obs class shared by the models that are truly correct, or
      * a fresh class id -- absent from `obs` -- when NO model is correct (the judge
        supplies a correct result no model produced).

    That second branch is the candidate-coverage fix: pinning z_i to a class outside
    the pool's outputs makes every model score wrong on i, which unlabeled consensus
    can never do. In the real zoo this class comes from executing the judge's chosen /
    written SQL (see RealJudge); here it is read off the withheld ground truth.
    """

    def __init__(self):
        self.calls = 0
        self._fresh = 20_000_000

    def query(self, run, obs, i):
        self.calls += 1
        correct = np.where(run.true_class[:, i] == 0)[0]
        if len(correct) == 0:                       # correct answer not in the pool
            self._fresh += 1
            return self._fresh
        vals, counts = np.unique(obs[correct, i], return_counts=True)
        return int(vals[np.argmax(counts)])


# --------------------------------------------------------------------------- #
#  Acquisition: per-item value + item-item similarity                          #
# --------------------------------------------------------------------------- #
def _entropy(post):
    p = np.array(list(post.values()), dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _norm(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def item_scores(out, run, obs):
    """Per-item acquisition signals. Everything an unlabeled estimator can see about
    where the ranking is fragile:

      entropy      soft-consensus uncertainty (split votes)
      disagree     fraction of models off the plurality class
      corr_risk    plurality is dominated by ONE provenance group's near-clones
                   (few independent groups back it) -> the collusion trap, which can
                   look perfectly confident by vote count
      vconflict    the execution verifier points away from the pool's plurality
      impact       resolving i could flip a currently-CLOSE pair of models

    Returns a dict of length-N arrays plus a combined `value` used by the submodular
    selector.
    """
    M, N = obs.shape
    group = run.group
    acc = out["acc"]
    vguess = run.verifier_guess

    entropy = np.array([_entropy(p) for p in out["latent_post"]])
    disagree = np.zeros(N)
    corr_risk = np.zeros(N)
    vconflict = np.zeros(N)
    # per-group dissent signature (for the similarity kernel)
    G = run.n_groups
    feat = np.zeros((N, G + 2))

    for i in range(N):
        col = obs[:, i]
        vals, counts = np.unique(col, return_counts=True)
        k_star = vals[np.argmax(counts)]
        plur = counts.max()
        supporters = np.where(col == k_star)[0]
        disagree[i] = 1.0 - plur / M
        # independent backing of the winning class = #distinct groups voting it
        indep = len(np.unique(group[supporters]))
        corr_risk[i] = 1.0 - indep / max(plur, 1)
        vconflict[i] = float(vguess[i] != k_star)
        # similarity features: per-group dissent fraction + verifier conflict + entropy
        for g in range(G):
            gm = np.where(group == g)[0]
            if len(gm):
                feat[i, g] = np.mean(col[gm] != k_star)
        feat[i, G] = vconflict[i]
        feat[i, G + 1] = entropy[i]

    # ranking impact: sum of closeness of currently-close model pairs that DISAGREE
    # on item i. tau scales "close"; only pairs within a few points matter.
    order = np.argsort(-acc)
    tau = 0.03
    close = np.exp(-np.abs(acc[:, None] - acc[None, :]) / tau)
    np.fill_diagonal(close, 0.0)
    neq = obs.T[:, :, None] != obs.T[:, None, :]        # [N,M,M] disagreement
    impact = 0.5 * np.einsum("ab,nab->n", close, neq.astype(float))

    # Acquisition value driving the submodular selector. It is led by `corr_risk` --
    # a plurality carried by a SINGLE provenance group -- which is ~0 everywhere except
    # on the near-clone-collusion items where the ranking is actually fragile, and so
    # (unlike raw disagreement or entropy) pinpoints candidate-gap traps regardless of
    # how noisy the benign items are. Small entropy / ranking-impact terms only break
    # ties once every colluded item is spent, degrading gracefully to uncertainty
    # sampling.
    value = corr_risk + 0.05 * _norm(entropy) + 0.02 * _norm(impact)
    value = np.clip(value, 1e-6, None)
    return dict(entropy=entropy, disagree=disagree, corr_risk=corr_risk,
                vconflict=vconflict, impact=impact, value=value, feat=feat,
                order=order)


def _similarity(feat):
    """Cosine similarity between item dissent signatures, in [0,1]. Two items are
    'similar' when the same groups dissent / the verifier conflicts the same way ->
    validating one informs the other, so the submodular objective avoids spending two
    judge calls on redundant near-duplicate items."""
    n = np.linalg.norm(feat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    fn = feat / n
    S = fn @ fn.T
    return np.clip(S, 0.0, 1.0)


# --------------------------------------------------------------------------- #
#  Greedy submodular selection (facility location)                             #
# --------------------------------------------------------------------------- #
def greedy_submodular(value, S, budget, preselected=None, lam=0.15):
    """Maximize the BLENDED monotone-submodular objective

        f(A) = sum_{i in A} value_i            (modular: validate high-value items)
             + lam * sum_i value_i * max_{j in A} S_ij   (facility location: coverage)

    under |A| <= budget via LAZY greedy (Minoux 1978 / CELF). f is monotone submodular
    (a modular function plus a facility-location function), so greedy is within
    (1-1/e) of optimal and the bound is tight (Nemhauser-Wolsey-Fisher 1978).

    The two terms handle the two regimes the paper needs:
      * INDEPENDENT ambiguity (candidate-gap traps -- each must be validated on its
        own): the modular term dominates, so f reduces to top-value and greedy is
        exactly optimal. Pure facility location would wrongly call redundant-looking
        traps 'covered' after one pick; the modular term keeps validating them.
      * REDUNDANT ambiguity (near-duplicate items where one label generalizes): the
        facility term suppresses wasted duplicate picks.

    `preselected` items (judged in earlier rounds) seed the coverage. Returns the list
    of NEWLY selected item indices (excludes preselected)."""
    N = len(value)
    preselected = list(preselected or [])
    # cover_i tracks max_{j in A} S_ij (a similarity); value weights it at gain time.
    cover = np.zeros(N)
    for j in preselected:
        cover = np.maximum(cover, S[:, j])

    def gain(j):
        return float(value[j] + lam * np.sum(value * np.maximum(0.0, S[:, j] - cover)))

    chosen = []
    available = [k for k in range(N) if k not in set(preselected)]
    # lazy-greedy heap of (-upper_bound_gain, last_updated_round, j)
    heap = [(-gain(j), 0, j) for j in available]
    heapq.heapify(heap)
    rounds = 0
    while heap and len(chosen) < budget:
        neg_g, updated, j = heapq.heappop(heap)
        if j in chosen:
            continue
        if updated == rounds:
            chosen.append(j)
            cover = np.maximum(cover, S[:, j])
            rounds += 1
        else:
            heapq.heappush(heap, (-gain(j), rounds, j))
    return chosen


def select(strategy, scores, S, k, preselected):
    """Pick k NEW items under the given acquisition strategy."""
    N = len(scores["value"])
    taken = set(preselected)
    avail = [i for i in range(N) if i not in taken]
    if k <= 0 or not avail:
        return []
    if strategy == "random":
        rng = np.random.default_rng(1234 + len(preselected))
        return list(rng.choice(avail, size=min(k, len(avail)), replace=False))
    if strategy == "hybrid_submodular":
        return greedy_submodular(scores["value"], S, k + len(preselected),
                                 preselected=preselected)[:k]
    key = {"uncertainty": "entropy", "disagreement": "disagree",
           "ranking_impact": "impact"}[strategy]
    s = scores[key]
    return [i for i in sorted(avail, key=lambda i: -s[i])[:k]]


# --------------------------------------------------------------------------- #
#  Abstention                                                                  #
# --------------------------------------------------------------------------- #
def abstention(out, run, obs, meff_min=2.0, min_groups=2, min_plurality=2):
    """Flag items where the estimator lacks enough INDEPENDENT evidence to be trusted,
    so the deployment ranking should defer to the judge instead of a confident guess.

    Per item: the winning result class is a plurality (>= min_plurality votes) but is
    backed by fewer than `min_groups` distinct provenance groups -- i.e. a near-clone
    bloc carries the consensus. That is exactly the candidate-gap / collusion trap: the
    vote count looks confident while the independent support is thin, and the verifier
    can be fooled by a plausible wrong query. Global: the effective independent-model
    count M_eff is too small for the pool to self-certify at all.
    """
    M, N = obs.shape
    group = run.group
    per_item = np.zeros(N, dtype=bool)
    for i in range(N):
        col = obs[:, i]
        vals, counts = np.unique(col, return_counts=True)
        k_star = vals[np.argmax(counts)]
        plur = counts.max()
        supporters = np.where(col == k_star)[0]
        indep = len(np.unique(group[supporters]))
        if plur >= min_plurality and indep < min_groups:
            per_item[i] = True
    return dict(global_abstain=bool(out["Meff"] < meff_min),
                Meff=float(out["Meff"]),
                per_item=per_item,
                n_flagged=int(per_item.sum()))


# --------------------------------------------------------------------------- #
#  Orchestrator                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class ActiveConfig:
    budget: int = 20          # judge calls allowed (== #items validated)
    rounds: int = 5           # validation rounds (re-score + incremental EM each)
    strategy: str = "hybrid_submodular"
    incremental_iters: int = 8   # EM iters per incremental (warm-started) re-solve
    early_stop: bool = False     # stop once the deployment decision is stable
    win_stop: float = 0.95       # top-1 win-probability threshold to stop
    meff_min: float = 2.0
    min_groups: int = 2


class ActivePoolEval:
    def __init__(self, cfg, active: ActiveConfig = None, judge=None):
        self.cfg = cfg
        self.pe = PoolEval(cfg)
        self.a = active or ActiveConfig()
        self.judge = judge or SimulatedJudge()

    def run(self, run):
        a = self.a
        # fix the observation once so every round scores the same kernel output
        obs = kernelmod.apply(run.true_class, self.cfg,
                              level=self.cfg.kernel_level,
                              rng=np.random.default_rng(self.cfg.seed + 7))
        out = self.pe.evaluate(run, obs=obs)                     # pure PoolEval (b=0)
        constraints = {}
        history = [dict(n_labels=0, acc=out["acc"].copy())]
        per_round = max(1, int(np.ceil(a.budget / max(1, a.rounds))))
        prev_top3 = set(out["ranking"][:3])
        stopped = None

        while len(constraints) < a.budget:
            k = min(per_round, a.budget - len(constraints))
            scores = item_scores(out, run, obs)
            S = _similarity(scores["feat"]) if a.strategy == "hybrid_submodular" \
                else None
            picks = select(a.strategy, scores, S, k, list(constraints.keys()))
            if not picks:
                break
            for i in picks:
                constraints[i] = self.judge.query(run, obs, i)
            # INCREMENTAL EM: warm-start from the current fit, few iters
            out = self.pe.evaluate(
                run, obs=obs, constraints=constraints,
                init=dict(a=out["acc"], b=out["b"], u=out["u"]),
                max_iters=a.incremental_iters)
            history.append(dict(n_labels=len(constraints), acc=out["acc"].copy()))

            top3 = set(out["ranking"][:3])
            if a.early_stop:
                r = out["ranking"]
                wp = self.pe.win_prob(out, r[0], r[1])
                if wp >= a.win_stop and top3 == prev_top3:
                    stopped = dict(reason="stable", n_labels=len(constraints),
                                   win_prob=wp)
                    break
            prev_top3 = top3

        out.update(abstain=abstention(out, run, obs, a.meff_min, a.min_groups),
                   constraints=constraints, judge_calls=self.judge.calls,
                   history=history, stopped=stopped, obs=obs)
        return out
