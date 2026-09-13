"""Partitions of a pool's answers: building them, renumbering them, and scoring one
against another.

This module holds no notion of MEANING.  Deciding which answers count as the same is the
task comparator's job -- executed-result agreement across database instances for
Text-to-SQL (``zoo/exec_suite.py``), or an answer matcher for free-text classification
(``pooleval/answer_matching.py``).  Here we only build the partition once that decision
rule exists, and measure it.
"""
import numpy as np

__all__ = ["to_repo_classes", "partition_quality", "greedy_meaning_clusters",
           "clusters_from_similarity", "connected_components"]


def greedy_meaning_clusters(n, equivalent, order=None):
    """Assign each answer to the first group whose REPRESENTATIVE it matches.

    ``equivalent(i, j) -> bool`` is the decision rule.  Greedy rather than transitive
    closure because the useful rules are not transitive: bidirectional entailment can
    hold for (a, b) and (b, c) and fail for (a, c).  Taking connected components would
    then CHAIN a to c, and in a framework that reads agreement as evidence, merging is
    the expensive direction.  First-match-against-a-representative cannot chain, so it
    errs toward splitting.  ``order`` pins the sweep so the partition is reproducible.
    """
    order = list(range(n)) if order is None else list(order)
    labels = np.full(n, -1, dtype=np.int64)
    reps = []
    for i in order:
        for c, rep in enumerate(reps):
            if equivalent(rep, i):
                labels[i] = c
                break
        else:
            labels[i] = len(reps)
            reps.append(i)
    return labels, np.asarray(reps, dtype=np.int64)


def clusters_from_similarity(S, threshold, order=None):
    """:func:`greedy_meaning_clusters` with ``equivalent(i, j) = S[i, j] >= threshold``."""
    S = np.asarray(S, dtype=float)
    return greedy_meaning_clusters(S.shape[0],
                                   lambda a, b: bool(S[a, b] >= threshold), order)


def connected_components(adjacency):
    """Transitive closure of a symmetric relation.  Kept as the COMPARISON, not a
    default: it is what taking an entailment graph's components does, and it is exactly
    the chaining failure :func:`greedy_meaning_clusters` avoids."""
    A = np.asarray(adjacency, dtype=bool)
    n = A.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    nxt = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = nxt
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(A[u]):
                if labels[v] < 0:
                    labels[v] = nxt
                    stack.append(int(v))
        nxt += 1
    return labels


def to_repo_classes(labels, gold_cluster=None, offset=1):
    """Map group ids onto the repo convention: ``0 == correct``, positive ``== a shared
    wrong answer``, so that two models giving the SAME wrong answer collide and that
    collision stays visible to ``e``.

    ``gold_cluster`` is the group the gold answer falls in, or ``None`` when no model
    produced it -- in which case every class is non-zero and every model scores wrong.
    """
    labels = np.asarray(labels)
    out = (np.where(labels == gold_cluster, 0, labels + offset)
           if gold_cluster is not None else labels + offset)
    return out.astype(np.int64)


def partition_quality(predicted, gold):
    """Pairwise precision / recall of one partition against another.

    The asymmetry is the point.  LOW PRECISION means the comparator merged answers that
    differ, which manufactures agreement: ``e`` inflates, the latent posterior sharpens
    on a class with less independent support than it appears to have, and the estimator
    becomes confidently wrong.  Low recall merely leaves signal on the table.

    A comparator that merges NOTHING has no pairs for precision to score.  It is reported
    as 0.0 rather than a flattering vacuous 1.0; ``n_predicted_same`` is what tells
    "undefined" apart from "bad", so read the two together.
    """
    predicted = np.asarray(predicted)
    gold = np.asarray(gold)
    if predicted.shape != gold.shape:
        raise ValueError("predicted and gold partitions must have the same shape")
    iu = np.triu_indices(len(predicted), k=1)
    same_pred = (predicted[:, None] == predicted[None, :])[iu]
    same_gold = (gold[:, None] == gold[None, :])[iu]
    tp = float(np.sum(same_pred & same_gold))
    precision = tp / max(float(np.sum(same_pred)), 1.0)
    recall = tp / max(float(np.sum(same_gold)), 1.0)
    return dict(precision=precision, recall=recall,
                f1=2 * precision * recall / max(precision + recall, 1e-12),
                n_pairs=int(len(same_pred)),
                n_predicted_same=int(same_pred.sum()),
                n_gold_same=int(same_gold.sum()))
