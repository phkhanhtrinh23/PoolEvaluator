"""Build the five deterministic modified database instances used by EX-Extended.

SQLite databases are copied file by file here; PostgreSQL and MySQL databases are
handled by ``pooleval.sql_dialects`` with the same edits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .data import TEXT2SQL_DATASETS, load_text2sql


@dataclass(frozen=True)
class InstanceRecord:
    source: str
    instance: str
    variant: int
    changed_rows: int
    integrity: str


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _tables(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def _mutated(value: object, salt: int) -> object:
    if value is None:
        return salt
    if isinstance(value, bool):
        return int(not value)
    if isinstance(value, int):
        return value + 1 + salt % 7
    if isinstance(value, float):
        return value + 0.125 * (1 + salt % 5)
    blob = str(value)
    return f"{blob}__pe{salt % 997}" if blob else f"pooleval_{salt % 997}"


def _mutate_table(connection: sqlite3.Connection, table: str, variant: int, cap_rows: int) -> int:
    quoted = _quote(table)
    try:
        info = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
        rows = connection.execute(f"SELECT rowid, * FROM {quoted} ORDER BY rowid LIMIT ?", (cap_rows,)).fetchall()
    except sqlite3.DatabaseError:
        return 0
    if not info or not rows:
        return 0
    changed = 0
    # Variants 2 and 5 delete different deterministic slices.
    if variant in {2, 5}:
        step = 10 if variant == 2 else 5
        rowids = [row[0] for index, row in enumerate(rows) if index % step == 0]
        for rowid in rowids:
            try:
                changed += connection.execute(f"DELETE FROM {quoted} WHERE rowid=?", (rowid,)).rowcount
            except sqlite3.DatabaseError:
                pass
        return changed

    # Variants 1/3/4 update values. Variant 3/4 also attempt an inserted clone.
    mutable = [column for column in info if not column[5]] or info
    column = mutable[(variant - 1) % len(mutable)]
    column_index = 1 + int(column[0])
    stride = 7 if variant == 1 else 5
    for index, row in enumerate(rows):
        if index % stride:
            continue
        value = _mutated(row[column_index], variant * 101 + index)
        try:
            changed += connection.execute(
                f"UPDATE {quoted} SET {_quote(column[1])}=? WHERE rowid=?", (value, row[0])
            ).rowcount
        except sqlite3.DatabaseError:
            pass
    if variant in {3, 4}:
        values = list(rows[0][1:])
        for column_info in info:
            idx = int(column_info[0])
            if column_info[5] or idx == int(column[0]):
                values[idx] = _mutated(values[idx], variant * 313 + idx)
        columns = ", ".join(_quote(column_info[1]) for column_info in info)
        marks = ", ".join("?" for _ in info)
        try:
            before = connection.total_changes
            connection.execute(f"INSERT OR IGNORE INTO {quoted} ({columns}) VALUES ({marks})", values)
            changed += connection.total_changes - before
        except sqlite3.DatabaseError:
            pass
    return changed


def build_instances(
    db_path: str | Path,
    output_dir: str | Path,
    count: int = 5,
    cap_rows_per_table: int = 100,
) -> list[InstanceRecord]:
    """Copy one source database and make ``count`` deterministic, queryable variants."""
    from .sql_dialects import build_server_instances, dialect

    if dialect(db_path) != "sqlite":
        return [
            InstanceRecord(str(db_path), instance, variant, changed, "ok")
            for variant, (instance, changed) in enumerate(
                build_server_instances(db_path, count, cap_rows_per_table), start=1
            )
        ]
    source = Path(db_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha1(str(source).encode()).hexdigest()[:10]
    destination = Path(output_dir).resolve() / f"{source.stem}-{digest}"
    destination.mkdir(parents=True, exist_ok=True)
    records: list[InstanceRecord] = []
    for variant in range(1, count + 1):
        target = destination / f"instance_{variant}.sqlite"
        temporary = target.with_suffix(".tmp")
        shutil.copy2(source, temporary)
        changed = 0
        with sqlite3.connect(temporary) as connection:
            connection.execute("PRAGMA foreign_keys=OFF")
            for table in _tables(connection):
                changed += _mutate_table(connection, table, variant, cap_rows_per_table)
            connection.commit()
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        temporary.replace(target)
        records.append(InstanceRecord(str(source), str(target), variant, changed, integrity))
    # Record portable names, not absolute local paths, so shared artifacts reveal no user directories.
    portable = [
        {**asdict(record), "source": f"{source.parent.name}/{source.name}", "instance": Path(record.instance).name}
        for record in records
    ]
    metadata = destination / "instances.json"
    metadata.write_text(json.dumps(portable, indent=2), encoding="utf-8")
    return records


def build_dataset_instances(
    dataset: str,
    split: str,
    output_dir: str | Path,
    limit_databases: int | None = None,
    **layout: str,
) -> list[InstanceRecord]:
    items = load_text2sql(dataset, split, **layout)
    databases: dict[Any, None] = {}
    for item in items:
        databases.setdefault(item.db_path, None)
    selected = list(databases)[:limit_databases]
    records: list[InstanceRecord] = []
    for index, path in enumerate(selected, start=1):
        print(f"[{dataset}] {index}/{len(selected)} {path}")
        records.extend(build_instances(path, Path(output_dir) / dataset))
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*TEXT2SQL_DATASETS, "all"], default="all")
    parser.add_argument("--split", choices=["target", "train"], default="target")
    parser.add_argument("--output-dir", default="db_instances")
    parser.add_argument("--limit-databases", type=int)
    parser.add_argument("--text2sql-root", default="datasets/text2sql")
    parser.add_argument("--bird-metadata-root", default="datasets/text2sql/bird")
    parser.add_argument("--bird-database-root", default="datasets/text2sql/bird")
    args = parser.parse_args(argv)
    datasets = TEXT2SQL_DATASETS if args.dataset == "all" else (args.dataset,)
    total = 0
    layout = {
        "text2sql_root": args.text2sql_root,
        "bird_metadata_root": args.bird_metadata_root,
        "bird_database_root": args.bird_database_root,
    }
    for dataset in datasets:
        try:
            records = build_dataset_instances(
                dataset,
                args.split,
                args.output_dir,
                limit_databases=args.limit_databases,
                **layout,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            if args.dataset != "all":
                raise
            print(f"[{dataset}] skipped: {exc}")
            continue
        total += len(records)
        for record in records:
            print(f"  v{record.variant}: rows={record.changed_rows} integrity={record.integrity} {record.instance}")
    print(f"built {total} modified database instances")


if __name__ == "__main__":
    main()
