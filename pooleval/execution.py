"""Safe read-only SQL execution (SQLite, PostgreSQL, MySQL) and EX-Extended equivalence."""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from .sql_dialects import dialect, execute_rows


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


def _sqlite_rows(db_path: str | Path, sql: str, timeout: float) -> tuple[list[str], list[tuple[Any, ...]]] | None:
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)

    def interrupt() -> None:
        connection.interrupt()

    timer = threading.Timer(timeout, interrupt)
    timer.start()
    try:
        cursor = connection.execute(sql)
        rows = [tuple(row) for row in cursor.fetchall()]
        columns = [column[0] for column in (cursor.description or ())]
        return columns, rows
    except sqlite3.DatabaseError:
        return None
    finally:
        timer.cancel()
        connection.close()


def execute_raw(database: Any, sql: str, timeout: float = 5.0) -> tuple[list[str], list[tuple[Any, ...]]] | None:
    """Column names and raw rows of a read-only query, or ``None`` if it fails.

    ``database`` is a SQLite file path or a ``postgresql://`` / ``mysql://`` locator.
    """
    if not sql or not _READ_ONLY.match(sql):
        return None
    if dialect(database) == "sqlite":
        return _sqlite_rows(database, sql, timeout)
    return execute_rows(database, sql, timeout)


def execute(database: Any, sql: str, timeout: float = 5.0) -> tuple[Any, ...] | None:
    """Return a canonical result key, or ``None`` for invalid/failed SQL."""
    result = execute_raw(database, sql, timeout)
    if result is None:
        return None
    columns, raw = result
    rows = [tuple(_value(v) for v in row) for row in raw]
    if not _ORDERED.search(sql):
        rows.sort(key=repr)
    return len(columns), tuple(rows)


def signature(sql: str, database_paths: Iterable[Any], timeout: float = 5.0) -> tuple[Any, ...]:
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
