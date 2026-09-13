"""Text-to-SQL equivalence by EXECUTION alone, across a suite of database instances.

Two queries mean the same thing iff they return the same table on every database
instance.  That is the whole rule here -- there is no clustering step, no embedding,
and no syntactic tie-breaker, because for SQL none is needed: the semantics of a query
IS its behaviour on databases, so more instances is the direct test rather than a proxy
for one.

Why more than one instance.  Comparing results on the single shipped database is
unsound in one specific direction: two genuinely different queries can coincide there by
accident -- ``LIMIT 1`` when only one row qualifies, a filter that happens to exclude
nothing, an aggregate over a column that happens to be constant.  Each accident is a
FALSE COLLISION, and a false collision is the expensive kind of error in this
framework: it tells ``e`` that two models fail together when they do not, and it tells
the latent posterior that a wrong class has more independent support than it really has.

The guarantee.  Equivalent queries agree on EVERY instance, so extra instances can only
ever split a class that one instance merged -- never split a truly equivalent pair.  The
refinement is sound by construction, which is why no threshold is exposed anywhere in
this module.  This is the pairwise-equivalence half of test-suite accuracy (Zhong, Yu &
Klein, *Semantic Evaluation for Text-to-SQL with Distilled Test Suites*, EMNLP 2020),
and it is what ``pooleval.kernel``'s LA2 -- "multi-instance (re-run on t
constraint-preserving sub-instances)" -- has always described but only ever SIMULATED
through a precision parameter; in real-data mode that kernel is the identity.

CORRECTNESS IS STILL DECIDED ON THE ORIGINAL DATABASE, against the gold query, exactly
as before.  The variants only decide which WRONG answers are the same wrong answer, so
row sampling changing a gold result on a variant is harmless by construction.
:func:`correct_flags` additionally reports the stricter test-suite grading as a separate
quantity, never substituted silently.
"""
import hashlib
import os
import shutil
import sqlite3

import numpy as np

from pooleval.partitions import to_repo_classes
from .execute import result_key


# --------------------------------------------------------------------------- #
#  1.  The test suite: row-sampled database variants                           #
# --------------------------------------------------------------------------- #
def _salt(db_path, variant):
    h = hashlib.sha1(f"{os.path.basename(db_path)}::{variant}".encode()).hexdigest()
    return int(h[:8], 16)


def _tables_and_views(con):
    rows = con.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'").fetchall()
    return [(t, n, s) for t, n, s in rows if s]


def build_variants(db_path, out_dir, k=3, keep_frac=0.6, max_rows=200_000, log=None):
    """Create ``k`` row-sampled copies of a database, cached on disk.

    Sampling is a deterministic salted hash of the row id, never ``RANDOM()`` -- the
    equivalence relation itself must be reproducible across runs, or no measurement
    built on it means anything.

    Foreign keys are deliberately not repaired.  A dangling reference simply makes a
    join drop rows, and that is exactly the perturbation we want: the kind of change
    that separates two queries which coincided on the full instance.  Realism is not
    required of a test suite; discrimination is.
    """
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for v in range(k):
        target = os.path.join(out_dir, f"v{v}_" + os.path.basename(db_path))
        if os.path.exists(target):
            made.append(target)
            continue
        try:
            _materialise(db_path, target, _salt(db_path, v), keep_frac, max_rows)
            made.append(target)
        except Exception as err:                             # pragma: no cover
            if log:
                log(f"    [variant] {os.path.basename(db_path)} v{v} failed: {err}")
            if os.path.exists(target):
                os.remove(target)
    return made


def _materialise(src_path, dst_path, salt, keep_frac, max_rows):
    tmp = dst_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.text_factory = lambda b: b.decode("utf-8", "ignore")
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute("ATTACH DATABASE ? AS src", (src_path,))
    with sqlite3.connect(src_path) as probe:
        probe.text_factory = lambda b: b.decode("utf-8", "ignore")
        objects = _tables_and_views(probe)
    cut = int(round(keep_frac * 1000))
    for kind, name, ddl in objects:
        try:
            con.execute(ddl)
        except sqlite3.Error:
            continue
        if kind != "table":
            continue                                  # a view is recreated, not filled
        quoted = name.replace('"', '""')
        for statement in (
            f'INSERT INTO "{quoted}" SELECT * FROM src."{quoted}" '
            f'WHERE ((rowid * 2654435761) + {salt}) % 1000 < {cut} LIMIT {max_rows}',
            f'INSERT INTO "{quoted}" SELECT * FROM src."{quoted}" LIMIT {max_rows}',
        ):
            try:
                con.execute(statement)
                break
            except sqlite3.Error:
                continue
    con.commit()
    con.close()
    shutil.move(tmp, dst_path)


# --------------------------------------------------------------------------- #
#  2.  Execution across the suite                                              #
# --------------------------------------------------------------------------- #
def execution_signature(sql, db_paths, timeout=5.0):
    """``(signature, n_ok)`` -- the tuple of canonical result keys across instances.

    ``n_ok`` counts instances on which the query actually executed, so a query that
    agrees everywhere is distinguishable from one that failed everywhere.
    """
    keys = []
    n_ok = 0
    for path in db_paths:
        key, _err = result_key(path, sql, timeout)
        keys.append(key)
        n_ok += key is not None
    return tuple(keys), n_ok


def informative_instances(gold_sql, db_paths, timeout=5.0):
    """Keep only instances on which the GOLD query still returns rows.

    Without this filter row sampling is actively harmful: prune enough and every query
    returns the empty table, all of them collide, and the variant MANUFACTURES the
    agreement it was added to disprove. An instance where the gold query still finds
    rows is one where a distinguishing answer is at least possible.
    """
    keep = []
    for path in db_paths:
        key, _err = result_key(path, gold_sql, timeout)
        if key is not None and _nonempty(key):
            keep.append(path)
    return keep


def _nonempty(key):
    """``zoo.execute.result_key`` returns ``(arity, canonical_rows)``; a zero-row result
    is the empty table regardless of arity."""
    try:
        return bool(len(key[1]))
    except (TypeError, IndexError):          # pragma: no cover - defensive
        return False


# --------------------------------------------------------------------------- #
#  3.  One item: from SQL strings to repo-convention class ids                 #
# --------------------------------------------------------------------------- #
def item_evidence(sqls, gold_sql, db_path, variant_paths=(), timeout=5.0):
    """Execute one item's pool answers once, against the whole suite, and keep it all.

    Every partition and every grading below derives from this single pass, so queries
    are never executed twice and the one-instance and suite kernels are guaranteed to be
    compared on identical evidence.
    """
    suite = [db_path] + list(variant_paths)
    signatures, n_ok = [], []
    for sql in sqls:
        sig, ok = execution_signature(sql, suite, timeout)
        signatures.append(sig)
        n_ok.append(ok)
    gold_sig, gold_ok = execution_signature(gold_sql, suite, timeout)
    return dict(suite=suite, signatures=signatures, n_ok=np.asarray(n_ok),
                gold_signature=gold_sig, gold_ok=int(gold_ok))


def _classes(keys, gold_key, n_ok=None):
    """Group answers by key; a query that never executed keeps a class of its own.

    Two execution errors are not an agreement -- they carry no answer to agree on -- so
    they are never merged, matching the repo's negative-id convention in
    ``zoo/execute.py``.
    """
    index, labels = {}, []
    for i, key in enumerate(keys):
        failed = key is None if n_ok is None else n_ok[i] == 0
        labels.append(index.setdefault(("err", i) if failed else ("ok", key),
                                       len(index)))
    labels = np.asarray(labels, dtype=np.int64)
    gold_cluster = None
    if gold_key is not None:
        for i, key in enumerate(keys):
            if key is not None and key == gold_key:
                gold_cluster = int(labels[i])
                break
    return to_repo_classes(labels, gold_cluster)


def single_instance_classes(ev):
    """The current kernel: the ORIGINAL instance only, exact canonical result match."""
    return _classes([sig[0] for sig in ev["signatures"]], ev["gold_signature"][0])


def suite_classes(ev):
    """Equivalence across the whole suite: same class iff the same result EVERYWHERE.

    A strict refinement of :func:`single_instance_classes` -- it can split a class that
    one instance merged and can never merge two the original kept apart.
    """
    gold_first = ev["gold_signature"][0]
    gold_full = ev["gold_signature"] if ev["gold_ok"] else None
    labels = _classes(ev["signatures"], gold_full, n_ok=ev["n_ok"])
    if gold_full is None or (labels == 0).any() or gold_first is None:
        return labels
    # The gold query ran on the original instance but no model matched it everywhere,
    # so no class is correct; the ids above are already all non-zero.
    return labels


def correct_flags(ev):
    """``(single_instance, suite)`` correctness per model.

    The first is agreement with gold on the shipped instance -- execution accuracy. The
    second additionally requires agreement on every instance, which is test-suite
    accuracy: it removes models that matched gold here by coincidence.
    """
    gold = ev["gold_signature"]
    single, suite = [], []
    for sig in ev["signatures"]:
        hit = gold[0] is not None and sig[0] is not None and sig[0] == gold[0]
        single.append(hit)
        suite.append(hit and all(a is not None and a == b
                                 for a, b in zip(sig[1:], gold[1:])))
    return np.asarray(single), np.asarray(suite)


def classes_for_item(sqls, gold_sql, db_path, variant_paths=(), timeout=5.0):
    """Convenience wrapper: one item, straight from SQL strings to class ids."""
    ev = item_evidence(sqls, gold_sql, db_path, variant_paths, timeout)
    classes = suite_classes(ev)
    return classes, dict(n_clusters=int(len(set(classes.tolist()))),
                         n_executed=int((ev["n_ok"] > 0).sum()),
                         gold_found=bool((classes == 0).any()),
                         suite_size=len(ev["suite"]))
