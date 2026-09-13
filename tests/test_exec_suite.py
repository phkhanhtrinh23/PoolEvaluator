"""Text-to-SQL equivalence by execution across a suite of database instances.

These run against real SQLite databases built on the fly, so they exercise the actual
comparator rather than a mock of it.
"""
import os
import sqlite3

import numpy as np
import pytest

from zoo.exec_suite import (build_variants, classes_for_item, correct_flags,
                            execution_signature, informative_instances,
                            item_evidence, single_instance_classes, suite_classes)


@pytest.fixture
def toy_db(tmp_path):
    """One table, ten rows. `t.n` runs 1..10 and `t.flag` is 1 on exactly one row."""
    path = str(tmp_path / "toy.sqlite")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER, flag INTEGER)")
    con.executemany("INSERT INTO t VALUES (?,?,?)",
                    [(i, i, 1 if i == 3 else 0) for i in range(1, 11)])
    con.commit()
    con.close()
    return path


# --------------------------------------------------------------------------- #
#  Building the suite                                                          #
# --------------------------------------------------------------------------- #
def test_variants_are_smaller_copies_and_are_deterministic(toy_db, tmp_path):
    a = build_variants(toy_db, str(tmp_path / "va"), k=3, keep_frac=0.5)
    assert len(a) == 3
    counts = []
    for path in a:
        with sqlite3.connect(path) as con:
            counts.append(con.execute("SELECT COUNT(*) FROM t").fetchone()[0])
    assert all(0 <= c <= 10 for c in counts)
    assert min(counts) < 10                     # something really was pruned

    b = build_variants(toy_db, str(tmp_path / "vb"), k=3, keep_frac=0.5)
    for pa, pb in zip(a, b):
        with sqlite3.connect(pa) as ca, sqlite3.connect(pb) as cb:
            assert (ca.execute("SELECT * FROM t ORDER BY id").fetchall()
                    == cb.execute("SELECT * FROM t ORDER BY id").fetchall())


def test_variants_differ_from_each_other(toy_db, tmp_path):
    """A suite of identical instances would carry no information at all."""
    paths = build_variants(toy_db, str(tmp_path / "v"), k=3, keep_frac=0.5)
    rows = []
    for path in paths:
        with sqlite3.connect(path) as con:
            rows.append(tuple(con.execute("SELECT id FROM t ORDER BY id").fetchall()))
    assert len(set(rows)) > 1


# --------------------------------------------------------------------------- #
#  The soundness guarantee                                                     #
# --------------------------------------------------------------------------- #
def test_equivalent_queries_agree_on_every_instance(toy_db, tmp_path):
    """The whole construction rests on this: a paraphrase is never split."""
    variants = build_variants(toy_db, str(tmp_path / "v"), k=3, keep_frac=0.5)
    suite = [toy_db] + variants
    a, _ = execution_signature("SELECT n FROM t WHERE n > 5 ORDER BY n", suite)
    b, _ = execution_signature("SELECT n FROM t WHERE 5 < n ORDER BY n", suite)
    c, _ = execution_signature("SELECT t.n FROM t WHERE NOT (t.n <= 5) ORDER BY t.n",
                               suite)
    assert a == b == c


# `n = 10` and `ORDER BY n DESC LIMIT 1` both return 10 on the full table, and diverge
# the moment row 10 is pruned: the first returns nothing, the second returns the new
# maximum. This is the shape of a real false collision -- a filter that happens to pick
# out the same row as an aggregate.
COINCIDENT = ["SELECT n FROM t WHERE n = 10",
              "SELECT n FROM t ORDER BY n DESC LIMIT 1"]


def test_the_suite_splits_a_coincidental_collision(toy_db, tmp_path):
    variants = build_variants(toy_db, str(tmp_path / "v"), k=6, keep_frac=0.4)
    one = item_evidence(COINCIDENT, COINCIDENT[0], toy_db, [])
    many = item_evidence(COINCIDENT, COINCIDENT[0], toy_db, variants)
    assert one["signatures"][0] == one["signatures"][1]      # merged on one instance
    assert many["signatures"][0] != many["signatures"][1]    # separated by the suite


def test_the_suite_only_ever_refines_the_single_instance_partition(toy_db, tmp_path):
    variants = build_variants(toy_db, str(tmp_path / "v"), k=4, keep_frac=0.5)
    sqls = ["SELECT n FROM t WHERE n = 10",
            "SELECT n FROM t ORDER BY n DESC LIMIT 1",
            "SELECT n FROM t ORDER BY n",
            "SELECT t.n FROM t ORDER BY t.n",
            "SELECT n FROM t WHERE n > 100"]
    ev = item_evidence(sqls, "SELECT n FROM t WHERE n = 3", toy_db, variants)
    one = single_instance_classes(ev)
    many = suite_classes(ev)
    for i in range(len(sqls)):
        for j in range(len(sqls)):
            if many[i] == many[j]:
                assert one[i] == one[j], "the suite merged a pair one instance split"


# --------------------------------------------------------------------------- #
#  Informativeness filter                                                      #
# --------------------------------------------------------------------------- #
def test_instances_where_gold_returns_nothing_are_dropped(toy_db, tmp_path):
    """Otherwise pruning eventually empties every result and everything collides."""
    variants = build_variants(toy_db, str(tmp_path / "v"), k=3, keep_frac=0.5)
    kept = informative_instances("SELECT n FROM t WHERE n = 999", variants)
    assert kept == []
    kept = informative_instances("SELECT n FROM t", variants)
    assert len(kept) == len(variants)


# --------------------------------------------------------------------------- #
#  Class ids and grading                                                       #
# --------------------------------------------------------------------------- #
def test_the_gold_matching_class_becomes_zero_and_errors_never_collide(toy_db):
    sqls = ["SELECT n FROM t WHERE n = 3",       # correct
            "SELECT n FROM t WHERE n = 3",       # correct, same answer
            "SELECT n FROM t WHERE n = 4",       # a wrong answer
            "SELECT nope FROM t",                # syntax/column error
            "SELECT alsonope FROM t"]            # a different error
    classes, diag = classes_for_item(sqls, "SELECT n FROM t WHERE n = 3", toy_db)
    assert classes[0] == classes[1] == 0
    assert classes[2] != 0
    assert classes[3] != classes[4]              # two errors are not an agreement
    assert diag["gold_found"] and diag["suite_size"] == 1


def test_correct_flags_demote_a_coincidental_match(toy_db, tmp_path):
    variants = build_variants(toy_db, str(tmp_path / "v"), k=6, keep_frac=0.4)
    ev = item_evidence(COINCIDENT, COINCIDENT[0], toy_db, variants)
    single, suite = correct_flags(ev)
    assert single[0] and single[1]        # both look right on the shipped instance
    assert suite[0] and not suite[1]      # only the gold query survives the suite


def test_no_correct_model_leaves_every_class_nonzero(toy_db):
    classes, diag = classes_for_item(["SELECT n FROM t WHERE n = 4"],
                                     "SELECT n FROM t WHERE n = 3", toy_db)
    assert np.all(classes != 0) and not diag["gold_found"]
