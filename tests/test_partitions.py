"""Partition bookkeeping and the asymmetry that makes a comparator safe or dangerous."""
import numpy as np
import pytest

from pooleval.partitions import partition_quality, to_repo_classes


def test_gold_group_becomes_class_zero_and_wrong_groups_keep_identity():
    labels = np.array([0, 0, 1, 2])
    out = to_repo_classes(labels, gold_cluster=1)
    assert out[2] == 0                      # the gold group
    assert out[0] == out[1] != 0            # two models sharing one wrong answer
    assert out[3] != out[0] and out[3] != 0


def test_no_correct_model_leaves_every_class_nonzero():
    out = to_repo_classes(np.array([0, 1, 1]), gold_cluster=None)
    assert np.all(out != 0)
    assert out[1] == out[2]                 # the shared wrong answer still collides


def test_partition_quality_is_perfect_against_itself():
    gold = np.array([0, 0, 1, 1])
    q = partition_quality(gold, gold)
    assert q["precision"] == 1.0 and q["recall"] == 1.0 and q["f1"] == pytest.approx(1.0)


def test_over_merging_shows_up_as_low_precision_not_low_recall():
    """The dangerous direction: it finds every true pair, but most pairs it finds are
    spurious -- which is how manufactured agreement looks."""
    q = partition_quality(np.zeros(4, dtype=int), np.array([0, 0, 1, 1]))
    assert q["recall"] == 1.0
    assert q["precision"] == pytest.approx(2 / 6)


def test_over_splitting_costs_recall_and_scores_no_pairs_for_precision():
    q = partition_quality(np.arange(4), np.array([0, 0, 1, 1]))
    assert q["recall"] == 0.0
    assert q["n_predicted_same"] == 0 and q["precision"] == 0.0


def test_a_refinement_never_loses_precision():
    """Splitting a correct partition can only drop recall; every surviving pair is
    still a true pair. This is the guarantee the execution suite relies on."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        gold = rng.integers(0, 3, size=10)
        # split each gold group in two -- a strict refinement of it
        finer = gold * 2 + rng.integers(0, 2, size=10)
        q = partition_quality(finer, gold)
        assert q["precision"] == 1.0 or q["n_predicted_same"] == 0


def test_partition_quality_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        partition_quality(np.zeros(3), np.zeros(4))
