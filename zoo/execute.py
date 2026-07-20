"""Execute SQL on the live SQLite DB, compare result sets (Spider-style), and
assign result-EQUIVALENCE CLASSES used as the estimator's observations.

Class convention (per item i, shared across models):
  * 0            -> result matches gold (correct)   [class 0 == "correct", as the sim]
  * >0 int       -> a distinct WRONG result set; two models with the same canonical
                    result get the SAME id  ->  this is the real agreement signal
  * negative id  -> execution error / timeout (unique per cell: no agreement)

Comparison is order-insensitive UNLESS the query has a top-level ORDER BY, values
are type-normalized (numeric-as-float, trimmed strings) -- the LA1 canonical
comparator the paper describes.
"""
import re
import sqlite3
import threading
from typing import List, Tuple, Any, Optional


def _canon_val(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    s = str(v).strip()
    try:
        return round(float(s), 6)
    except ValueError:
        return s.lower()


def _canon_rows(rows: List[Tuple], ordered: bool) -> Tuple:
    canon = [tuple(_canon_val(x) for x in r) for r in rows]
    if not ordered:
        canon = sorted(canon, key=lambda t: tuple((v is None, str(v)) for v in t))
    return tuple(canon)


_ORDERBY = re.compile(r"\border\s+by\b", re.I)


def run_query(db_path: str, sql: str, timeout: float = 5.0):
    """Return (ok, rows|None, error|None). Interrupts on timeout."""
    if not sql or not sql.strip():
        return False, None, "empty"
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:  # noqa
        return False, None, f"connect:{e}"
    timed_out = {"v": False}

    def _interrupt():
        timed_out["v"] = True
        conn.interrupt()

    timer = threading.Timer(timeout, _interrupt)
    timer.start()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        arity = len(cur.description) if cur.description else 0
        return True, (rows, arity), None
    except Exception as e:  # noqa
        return False, None, ("timeout" if timed_out["v"] else str(e)[:120])
    finally:
        timer.cancel()
        conn.close()


def result_key(db_path: str, sql: str, timeout: float):
    """Canonical hashable key of a query's result, or None if it failed.

    The key folds in column arity so that two EMPTY results with different column
    counts don't spuriously agree (an execution-accuracy edge case on empty gold)."""
    ok, payload, err = run_query(db_path, sql, timeout)
    if not ok:
        return None, err
    rows, arity = payload
    ordered = bool(_ORDERBY.search(sql or ""))
    return (arity, _canon_rows(rows, ordered)), None


def assign_classes(gold_key, pred_keys):
    """Map each model's canonical result key to an equivalence-class id for one item.

    gold_key: canonical key of the gold query's result (may be None if gold failed).
    pred_keys: list over models of canonical keys (None = exec error).
    Returns list[int] class ids (0 = correct, >0 shared wrong, <0 unique error).
    """
    classes = []
    wrong_map = {}
    next_wrong = 1
    err_id = -1
    for k in pred_keys:
        if k is None:
            classes.append(err_id); err_id -= 1            # unique error, no agreement
        elif gold_key is not None and k == gold_key:
            classes.append(0)                               # correct
        else:
            if k not in wrong_map:
                wrong_map[k] = next_wrong; next_wrong += 1
            classes.append(wrong_map[k])                    # shared wrong result
    return classes


def is_nonempty(k) -> bool:
    """A canonical key (arity, rows) that executed and returned at least one row."""
    return k is not None and len(k[1]) > 0
