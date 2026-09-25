"""Official result-table scorers for benchmarks whose gold answers are result tables.

* ``spider2_match`` ports ``compare_pandas_table`` / ``compare_multi_pandas_table`` of the
  Spider 2.0-lite evaluation suite: every gold column (or each ``condition_cols`` column)
  must equal some predicted column, element-wise within 1e-2, optionally ignoring order;
  a prediction is correct when it matches any of the gold tables.
* ``entsql_match`` ports the EntSQL ``evaluate.py`` rule: ignoring headers and orders,
  every gold row's non-empty normalized values must be contained in a distinct
  predicted row (numeric tolerance 0.002 absolute or 1% relative).

Values are compared as the official scripts see them after a CSV round trip: empty
cells are missing and numeric strings are numbers.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence


def _read_csv(path: str | Path) -> tuple[list[str], list[list[str]]]:
    for encoding in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            with open(path, newline="", encoding=encoding) as handle:
                rows = list(csv.reader(handle))
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 always decodes
        rows = []
    return (rows[0], rows[1:]) if rows else ([], [])


# -------------------------------------------------------------------------- Spider 2.0


def _spider2_cell(value: Any) -> Any:
    """pandas.read_csv typing followed by Spider 2.0's ``normalize`` (missing -> 0)."""
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    if text == "":
        return 0
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() and "." not in text and "e" not in text.lower() else number


def _vectors_match(v1: list[Any], v2: list[Any], ignore_order: bool, tolerance: float = 1e-2) -> bool:
    if ignore_order:
        key = lambda x: (x is None, str(x), isinstance(x, (int, float)))  # noqa: E731
        v1, v2 = sorted(v1, key=key), sorted(v2, key=key)
    if len(v1) != len(v2):
        return False
    for a, b in zip(v1, v2):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), abs_tol=tolerance):
                return False
        elif a != b:
            return False
    return True


def _columns(rows: Sequence[Sequence[Any]], width: int) -> list[list[Any]]:
    return [[_spider2_cell(row[c]) if c < len(row) else 0 for row in rows] for c in range(width)]


def spider2_match(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    gold_paths: Sequence[str | Path],
    condition_cols: Any = None,
    ignore_order: bool = False,
) -> bool:
    if not gold_paths:
        return False
    predicted = _columns(rows, len(columns))
    multi = len(gold_paths) > 1
    if condition_cols in (None, [], [[]], [None]):
        per_gold = [[] for _ in gold_paths]
    elif multi and not all(isinstance(entry, list) for entry in condition_cols):
        per_gold = [condition_cols for _ in gold_paths]
    elif multi:
        per_gold = list(condition_cols)
    else:
        per_gold = [condition_cols]
    for path, condition in zip(gold_paths, per_gold):
        header, gold_rows = _read_csv(path)
        gold = _columns(gold_rows, len(header))
        if condition:
            chosen = condition if isinstance(condition, (list, tuple)) else [condition]
            gold = [gold[int(c)] for c in chosen if int(c) < len(gold)]
        if all(any(_vectors_match(g, p, ignore_order) for p in predicted) for g in gold):
            return True
    return False


# ------------------------------------------------------------------------------ EntSQL


def _entsql_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    if text.strip() == "":
        return ""
    text = text.replace('"', "").replace("'", "").strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    text = text.replace(",", "")
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    places = min(len(text.rstrip("0").split(".")[-1]), 4) if "." in text else 0
    return round(number, places)


def _numeric_match(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        diff = abs(a - b)
        return diff <= 0.002 or (a != 0 and diff / abs(a) <= 0.01)
    return a == b


def _contains(row: list[Any], needed: list[Any]) -> bool:
    available = list(row)
    for value in needed:
        for index, candidate in enumerate(available):
            if _numeric_match(value, candidate):
                del available[index]
                break
        else:
            return False
    return True


def entsql_match(rows: Sequence[Sequence[Any]], gold_path: str | Path) -> bool:
    _, gold_rows = _read_csv(gold_path)
    gold = [[v for v in (_entsql_value(x) for x in row) if v != ""] for row in gold_rows]
    predicted = [[_entsql_value(x) for x in row] for row in rows]
    if not gold:
        return not predicted
    used: set[int] = set()
    for needed in gold:
        if not needed:
            continue
        match = next(
            (i for i, row in enumerate(predicted) if i not in used and _contains(row, needed)), None
        )
        if match is None:
            return False
        used.add(match)
    return True
