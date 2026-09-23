"""Safe SQLite execution and EX-Extended answer equivalence."""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence


_READ_ONLY = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_ORDERED = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def _value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 8)
    text = str(value).strip()
    try:
        return round(float(text), 8)
    except ValueError:
        return text.casefold()


def execute(db_path: str | Path, sql: str, timeout: float = 5.0) -> tuple[Any, ...] | None:
    """Return a canonical result key, or ``None`` for invalid/failed SQL."""
    if not sql or not _READ_ONLY.match(sql):
        return None
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    timed_out = False

    def interrupt() -> None:
        nonlocal timed_out
        timed_out = True
        connection.interrupt()

    timer = threading.Timer(timeout, interrupt)
    timer.start()
    try:
        cursor = connection.execute(sql)
        rows = [tuple(_value(v) for v in row) for row in cursor.fetchall()]
        if not _ORDERED.search(sql):
            rows.sort(key=repr)
        width = len(cursor.description or ())
        return width, tuple(rows)
    except sqlite3.DatabaseError:
        return None
    finally:
        timer.cancel()
        connection.close()


def signature(sql: str, database_paths: Iterable[str | Path], timeout: float = 5.0) -> tuple[Any, ...]:
    """Query behavior on original + five modified instances (EX-Extended)."""
    return tuple(execute(path, sql, timeout) for path in database_paths)


def correct(predicted: tuple[Any, ...], gold: tuple[Any, ...]) -> bool:
    return all(p is not None and p == g for p, g in zip(predicted, gold))


def response_classes(signatures: Sequence[tuple[Any, ...]]) -> list[int]:
    """Map equal successful signatures together; every all-error response is unique."""
    classes: dict[tuple[Any, ...], int] = {}
    output: list[int] = []
    next_class = 0
    next_error = -1
    for value in signatures:
        if not value or all(part is None for part in value):
            output.append(next_error)
            next_error -= 1
        elif value in classes:
            output.append(classes[value])
        else:
            classes[value] = next_class
            output.append(next_class)
            next_class += 1
    return output
