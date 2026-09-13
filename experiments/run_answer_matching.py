"""When the pool is VLMs / GLMs, does matching answers by meaning recover what exact
string match destroys -- and does it do so CONSISTENTLY, across shifts and across kinds
of surface variation?

Image and node classification stop having a closed OUTPUT space the moment the pool is
made of vision-language or graph-language models: the answer is "dog", "a dog", "It's a
dog", "The answer is dog" -- one meaning, four surfaces.  Every estimator in this repo
reads agreement through a single equality test, so that mis-count lands directly in
``e`` (correlated error is UNDERCOUNTED: two clones failing alike in different words
look independent) and in ``gamma = P(C=0|Z=0)`` (OVERCOUNTED toward 1: a wrong model
that paraphrased the pseudo-label's own wrong answer is scored as disagreeing).

PROTOCOL, stated plainly.  The cached pools hold label INDICES from CNNs and GNNs, not
generated text.  To isolate the surface-form effect we RENDER those predictions as free
text with per-model phrasing styles, so the underlying accuracy, the agreement pattern,
and the correlated-error structure are all held exactly fixed and the only thing that
changes is how the answers are written.  This is a controlled simulation of VLM/GLM
surface variation, not real VLM output: it measures how much a matcher can recover, not
how a particular VLM behaves.

The oracle partition is therefore known -- two answers mean the same thing iff they
render the same underlying label -- which is what makes the matchers measurable at all.

A matcher is only worth keeping if it helps everywhere and never hurts, because the two
directions of error are not symmetric: a matcher that MERGES two different answers
manufactures agreement and corrupts every downstream estimate, while one that SPLITS an
answer only loses signal.  So every shift and both surface conditions are run, and the
report shows the worst case alongside the mean.

  python experiments/run_answer_matching.py --domain vision graph --surface easy hard
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooleval import Config, PoolEval                                 # noqa: E402
from pooleval.answer_matching import (EmbeddingMatcher, EntailmentMatcher,  # noqa: E402
                                      ExactMatcher, LabelGroundingMatcher,
                                      NormalizeMatcher)
from pooleval.partitions import partition_quality, to_repo_classes    # noqa: E402
from pooleval.validated_em import (OracleExpert, run_validation,      # noqa: E402
                                   validated_em, wrong_collision_matrix)
from experiments.run_validated_em import (_pool_run, _strength, _subsample,  # noqa: E402
                                          build_stats, mae)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
POOL_CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")

DIGITS = ["zero", "one", "two", "three", "four",
          "five", "six", "seven", "eight", "nine"]
# Cora/CiteSeer-style topics; the pools carry K=5 classes with no stored names.
TOPICS = ["Theory", "Reinforcement Learning", "Genetic Algorithms",
          "Neural Networks", "Probabilistic Methods", "Case Based",
          "Rule Learning"]

# One voice per model, the way different VLMs answer differently but consistently.
#
# "easy" varies only the FRAME around the label -- articles, copulas, hedges, meta
# prefixes. A regex can peel all of it, which is the point: it is the floor of the
# problem, and any matcher that fails here fails outright.
FRAME_STYLES = [
    "{label}",
    "a {label}",
    "It's a {label}",
    "The answer is {label}",
    "I think it is a {label}",
    "This is a {label}",
    "answer: {label}",
]

# "hard" also varies the LABEL ITSELF -- numeral vs word, acronym vs expansion,
# synonym, description. No amount of string surgery unifies "3" with "three" or "NN"
# with "Neural Networks", so this is where a meaning model has to earn its place. Real
# VLM and GLM output contains both kinds of variation; the easy set alone would let a
# normaliser look better than it deserves.
ALIASES = {
    "zero": ["0", "zero", "the digit 0", "a handwritten zero", "numeral 0"],
    "one": ["1", "one", "the digit 1", "a handwritten one", "numeral 1"],
    "two": ["2", "two", "the digit 2", "a handwritten two", "numeral 2"],
    "three": ["3", "three", "the digit 3", "a handwritten three", "numeral 3"],
    "four": ["4", "four", "the digit 4", "a handwritten four", "numeral 4"],
    "five": ["5", "five", "the digit 5", "a handwritten five", "numeral 5"],
    "six": ["6", "six", "the digit 6", "a handwritten six", "numeral 6"],
    "seven": ["7", "seven", "the digit 7", "a handwritten seven", "numeral 7"],
    "eight": ["8", "eight", "the digit 8", "a handwritten eight", "numeral 8"],
    "nine": ["9", "nine", "the digit 9", "a handwritten nine", "numeral 9"],
    "Theory": ["Theory", "theoretical work", "theoretical computer science",
               "a theory paper"],
    "Reinforcement Learning": ["Reinforcement Learning", "RL", "reward-based learning",
                               "an RL paper"],
    "Genetic Algorithms": ["Genetic Algorithms", "GA", "evolutionary computation",
                           "a genetic algorithms paper"],
    "Neural Networks": ["Neural Networks", "NN", "deep learning",
                        "connectionist models", "a neural network paper"],
    "Probabilistic Methods": ["Probabilistic Methods", "Bayesian methods",
                              "probabilistic modelling", "a probabilistic paper"],
    "Case Based": ["Case Based", "CBR", "case-based reasoning", "a CBR paper"],
    "Rule Learning": ["Rule Learning", "rule induction", "learning rules",
                      "a rule learning paper"],
}


def verbalize(pred, label_names, seed=0, jitter=0.25, surface="easy"):
    """[M, N] label indices -> [M, N] strings, one house style per model.

    ``jitter`` is the chance a model departs from its own style on an item, so no
    matcher can succeed by keying on model identity alone.

    ``surface="hard"`` draws the label's own wording from :data:`ALIASES` as well as the
    frame, so the same meaning appears as "3", "three" and "the digit 3".
    """
    rng = np.random.default_rng(seed)
    M, N = pred.shape
    own_frame = rng.integers(0, len(FRAME_STYLES), size=M)
    own_alias = rng.integers(0, 16, size=M)
    out = np.empty((M, N), dtype=object)
    for m in range(M):
        frames = np.where(rng.random(N) < jitter,
                          rng.integers(0, len(FRAME_STYLES), size=N), own_frame[m])
        alias_pick = np.where(rng.random(N) < jitter,
                              rng.integers(0, 16, size=N), own_alias[m])
        for i in range(N):
            canonical = label_names[pred[m, i]]
            if surface == "hard":
                options = ALIASES.get(canonical, [canonical])
                word = options[alias_pick[i] % len(options)]
                # a fully spelled-out alias already carries its own frame
                out[m, i] = (word if word.startswith(("a ", "an ", "the "))
                             else FRAME_STYLES[frames[i]].format(label=word))
            else:
                out[m, i] = FRAME_STYLES[frames[i]].format(label=canonical)
    return out


def classes_under(matcher, text, pred, gold, warm=None):
    """[M, N] strings -> repo-convention class ids, per item, under one matcher."""
    M, N = text.shape
    if warm is not None:
        warm(sorted({str(x) for x in text.ravel()}))
    classes = np.zeros((M, N), dtype=np.int64)
    for i in range(N):
        labels = matcher.cluster([str(x) for x in text[:, i]])
        hit = np.flatnonzero(pred[:, i] == gold[i])
        gold_cluster = int(labels[hit[0]]) if len(hit) else None
        classes[:, i] = to_repo_classes(labels, gold_cluster)
    return classes


def oracle_classes(pred, gold):
    """What the matcher would recover if it were perfect: cluster == underlying label."""
    return np.where(pred == gold[None, :], 0, 1 + pred).astype(np.int64)


def build_matchers(names, args, shared):
    """Assemble the matcher list for one label vocabulary.

    ``shared`` carries the embedding and entailment backends across shifts so their
    caches -- which are keyed on answer strings, and the answer vocabulary is tiny and
    repeats -- are paid for once rather than once per shift.
    """
    em = shared.get("embedding")
    ent = shared.get("entailment")
    out = [("exact string", ExactMatcher(), None),
           ("normalize", NormalizeMatcher(), None)]
    if em is not None:
        out.append((f"embedding (cos>={args.embedding_threshold})", em, None))
    if ent is not None:
        out.append((f"bidir entailment (>={args.entailment_threshold})", ent, ent.warm))
        out.append(("normalize + entailment", _Chain(NormalizeMatcher(), ent), ent.warm))
    out.append(("label grounding", LabelGroundingMatcher(names, backup=em), None))
    # The same matcher, told which surfaces its classes appear as. A closed label set
    # almost always arrives with the vocabulary its classes are named in, so withholding
    # it measures a handicap nobody deploying this would actually accept.
    declared = {k: v for k, v in ALIASES.items() if k in names}
    out.append(("label grounding + declared aliases",
                LabelGroundingMatcher(names, backup=em, aliases=declared), None))
    out.append(("ORACLE (underlying label)", None, None))
    return out


def evaluate(name, pool, args, shared, log=print):
    pred = np.asarray(pool["pred"])
    gold = np.asarray(pool["gold"])
    group = np.asarray(pool["group"])
    prior = np.asarray(pool["prior"])
    pred_s = (np.asarray(pool["pred_s"]) if "pred_s" in pool
              else np.asarray(pool["prob_s"]).argmax(-1))
    gold_s = np.asarray(pool["src_gold_val"])
    K = int(np.asarray(pool["n_classes"]))
    names = (DIGITS if K == 10 else TOPICS)[:K]

    ti = _subsample(pred.shape[1], args.n_target, args.seed)
    si = _subsample(pred_s.shape[1], args.n_labeled, args.seed)
    pred_t, gold_t = pred[:, ti], gold[ti]
    pred_l, gold_l = pred_s[:, si], gold_s[si]

    surface = getattr(args, "surface_current", "easy")
    text_t = verbalize(pred_t, names, seed=args.seed, jitter=args.jitter,
                       surface=surface)
    text_l = verbalize(pred_l, names, seed=args.seed + 1, jitter=args.jitter,
                       surface=surface)
    true_acc = (pred_t == gold_t[None, :]).mean(axis=1)
    prior_sigma = np.full(len(prior), args.prior_noise)
    N = pred_t.shape[1]
    strength = _strength(prior, prior_sigma, cap=N)

    log(f"\n[{name}] M={pred.shape[0]} K={K} target N={N} labeled N={len(si)}  "
        f"true acc {true_acc.min():.3f}-{true_acc.max():.3f}  "
        f"distinct surfaces {len({str(x) for x in text_t.ravel()})}")

    oracle_t = oracle_classes(pred_t, gold_t)
    oracle_l = oracle_classes(pred_l, gold_l)

    matchers = build_matchers(names, args, shared)

    rows = {}
    for label, matcher, warm in matchers:
        t0 = time.time()
        if matcher is None:
            cls_t, cls_l = oracle_t, oracle_l
        else:
            cls_t = classes_under(matcher, text_t, pred_t, gold_t, warm)
            cls_l = classes_under(matcher, text_l, pred_l, gold_l, warm)
        q = [partition_quality(cls_t[:, i], oracle_t[:, i]) for i in range(N)]
        stats = build_stats(cls_l, group, prior)
        e = wrong_collision_matrix(cls_l)
        off = ~np.eye(e.shape[0], dtype=bool)

        cfg = Config(real_data=True, M=cls_t.shape[0], N=N,
                     n_groups=int(group.max()) + 1)
        run = _pool_run(cls_t, true_acc, group, prior, prior_sigma)
        pe = mae(PoolEval(cfg).evaluate(run, obs=cls_t)["acc"], true_acc)
        vem = mae(validated_em(cls_t, build_stats(cls_l, group, prior), prior,
                               strength, max_iters=200)["acc"], true_acc)
        judged = mae(run_validation(
            cls_t, build_stats(cls_l, group, prior), prior, strength,
            expert=OracleExpert(cls_t, allow_none=True), budget=args.budget,
            select="random", seed=args.seed)["acc"], true_acc)

        rows[label] = dict(
            clusters=float(np.mean([len(set(cls_t[:, i].tolist())) for i in range(N)])),
            precision=float(np.mean([x["precision"] for x in q])),
            recall=float(np.mean([x["recall"] for x in q])),
            e_offdiag=float(e[off].mean()),
            gamma=float(stats.conditional_gamma().mean()),
            beta=float(stats.pseudo_accuracy),
            mae_pooleval=pe, mae_validated=vem, mae_judged=judged,
            seconds=round(time.time() - t0, 1))
        log(f"    {label:34s} clus {rows[label]['clusters']:5.2f}  "
            f"P {rows[label]['precision']:.3f} R {rows[label]['recall']:.3f}  "
            f"e {rows[label]['e_offdiag']:.4f}  g {rows[label]['gamma']:.3f}  "
            f"MAE pool {pe:6.2f} / vEM {vem:6.2f} / +judge {judged:6.2f}  "
            f"({rows[label]['seconds']}s)")
    return rows


class _Chain:
    """Two matchers in series: equivalent if EITHER says so.

    Used for normalise-then-entail. The two miss different things -- normalisation
    cannot see that "canine" and "dog" agree, entailment stumbles on meta-frames like
    "The answer is dog" -- so their union recovers more than either. Taking the union
    can only MERGE more, so it is the risky direction and is measured, not assumed.
    """

    def __init__(self, *matchers):
        self.matchers = matchers

    def equivalent(self, a, b):
        return any(m.equivalent(a, b) for m in self.matchers)

    def cluster(self, answers):
        from pooleval.partitions import greedy_meaning_clusters
        answers = list(answers)
        labels, _ = greedy_meaning_clusters(
            len(answers), lambda i, j: self.equivalent(answers[i], answers[j]))
        return labels


VISION = [("mnist", "usps"), ("mnist", "svhn")]
GRAPH = ["AC", "AD", "CA", "CD", "DA", "DC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", nargs="+", default=["vision", "graph"])
    ap.add_argument("--surface", nargs="+", choices=["easy", "hard"],
                    default=["easy", "hard"])
    ap.add_argument("--n-target", type=int, default=400)
    ap.add_argument("--n-labeled", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--prior-noise", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-embedding", dest="embedding", action="store_false")
    ap.add_argument("--no-entailment", dest="entailment", action="store_false")
    ap.add_argument("--embedding-threshold", type=float, default=0.75)
    ap.add_argument("--entailment-threshold", type=float, default=0.5)
    ap.add_argument("--shifts", nargs="+", default=None)
    ap.add_argument("--out", default=os.path.join(RESULTS, "answer_matching.json"))
    args = ap.parse_args()

    shared = {}
    if args.embedding:
        shared["embedding"] = EmbeddingMatcher(threshold=args.embedding_threshold)
    if args.entailment:
        shared["entailment"] = EntailmentMatcher(threshold=args.entailment_threshold)

    keys = []
    for domain in args.domain:
        keys += (["vision_%s_%s" % t for t in VISION] if domain == "vision"
                 else ["graph_%s" % t for t in GRAPH])
    if args.shifts:
        keys = [k for k in keys if any(t in k for t in args.shifts)]

    payload = {}
    for surface in args.surface:
        args.surface_current = surface
        for key in keys:
            path = os.path.join(POOL_CACHE, key + ".npz")
            if not os.path.exists(path):
                print(f"  [skip] {path}")
                continue
            z = np.load(path, allow_pickle=True)
            payload[f"{surface}/{key}"] = evaluate(
                f"{surface}/{key}", {k: z[k] for k in z.files}, args, shared)
    payload["metadata"] = {k: v for k, v in vars(args).items()
                           if k != "surface_current"}
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as h:
        json.dump(payload, h, indent=2, default=float)
    summarize(payload)
    print(f"\n[saved] {args.out}")


def summarize(payload):
    """Per-matcher worst case and mean over every shift, per surface condition.

    The WORST column is the one that decides whether a matcher is safe to keep: a
    matcher that helps on average while hurting somewhere is not a matcher you can
    deploy without knowing in advance which case you are in.
    """
    rows = {k: v for k, v in payload.items() if k != "metadata"}
    for surface in ("easy", "hard"):
        cases = {k: v for k, v in rows.items() if k.startswith(surface + "/")}
        if not cases:
            continue
        names = list(next(iter(cases.values())))
        base = {c: cases[c]["exact string"]["mae_pooleval"] for c in cases}
        print(f"\n{'=' * 96}\n  {surface.upper()} surface -- {len(cases)} shifts"
              f"\n{'=' * 96}")
        print(f"  {'matcher':36s}{'minP':>7s}{'meanP':>7s}{'meanR':>7s}"
              f"{'meanMAE':>9s}{'worstMAE':>10s}{'mean gain':>11s}{'worst gain':>12s}")
        for n in names:
            p = [cases[c][n]["precision"] for c in cases]
            r = [cases[c][n]["recall"] for c in cases]
            m = [cases[c][n]["mae_pooleval"] for c in cases]
            gain = [base[c] - cases[c][n]["mae_pooleval"] for c in cases]
            print(f"  {n:36s}{min(p):7.3f}{np.mean(p):7.3f}{np.mean(r):7.3f}"
                  f"{np.mean(m):9.2f}{max(m):10.2f}{np.mean(gain):+11.2f}"
                  f"{min(gain):+12.2f}")


if __name__ == "__main__":
    main()
