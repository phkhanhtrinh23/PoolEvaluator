"""Text2SQL benchmark loading (layouts described in the README).

Spider and BIRD are loaded here; Spider 2.0, BEAVER, ScienceBenchmark, EntSQL, and
LiveSQLBench are loaded by ``pooleval.benchmarks``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Text2SQLItem:
    """One question.  ``db_path`` is a SQLite file or a ``postgresql://``/``mysql://`` locator.

    Gold is ``gold_sql`` (scored with EX-Extended) or, when a benchmark releases result
    tables instead, ``gold_results`` scored by ``gold_matcher`` ("spider2" or "entsql")
    with the JSON options in ``gold_spec``.  Items without gold can still be evaluated
    label-free; true accuracy is then not reported.
    """

    id: str
    dataset: str
    db_id: str
    question: str
    gold_sql: str | None
    db_path: Any
    schema: dict[str, Any]
    evidence: str = ""
    dialect: str = "sqlite"
    gold_results: tuple[str, ...] = ()
    gold_matcher: str | None = None
    gold_spec: str = ""

    @property
    def has_gold(self) -> bool:
        return bool(self.gold_sql) or bool(self.gold_results)


def dataset_layout(
    dataset: str,
    split: str,
    text2sql_root: str | Path = "datasets/text2sql",
    bird_metadata_root: str | Path = "datasets/text2sql/bird",
    bird_database_root: str | Path = "datasets/text2sql/bird",
) -> tuple[Path, list[Path]]:
    root = Path(text2sql_root)
    bird_meta = Path(bird_metadata_root)
    bird_db = Path(bird_database_root)
    side = "dev" if split in {"dev", "target"} else "train"
    if dataset == "spider":
        return (
            root / "spider" / f"sft_spider_{side}_text2sql.json",
            [root / "spider" / "database", root / "spider" / "test_database"],
        )
    if dataset == "bird":
        roots = [
            bird_db / "dev" / "dev_databases",
            bird_db / "train" / "train_databases",
        ]
        return bird_meta / f"sft_bird_{side}_text2sql.json", roots
    raise KeyError(f"unknown Text2SQL dataset: {dataset}")


def resolve_database(db_id: str, roots: Iterable[Path]) -> Path:
    for root in roots:
        directory = Path(root) / db_id
        for name in (f"{db_id}.sqlite", f"{db_id}.db"):
            candidate = directory / name
            if candidate.is_file() and candidate.stat().st_size > 4096:
                return candidate
        if directory.is_dir():
            for candidate in sorted(directory.iterdir()):
                if candidate.suffix.lower() in {".sqlite", ".db"} and candidate.stat().st_size > 4096:
                    return candidate
    raise FileNotFoundError(f"no materialized SQLite database for {db_id!r} under {[str(r) for r in roots]}")


@lru_cache(maxsize=256)
def introspect_schema(db_path: str) -> dict[str, Any]:
    from .sql_dialects import dialect, introspect

    if dialect(db_path) != "sqlite":
        return introspect(db_path)
    items: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for (table,) in tables:
            quoted = table.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{quoted}")').fetchall()
            items.append(
                {
                    "table_name": table,
                    "column_names": [row[1] for row in columns],
                    "column_types": [row[2] for row in columns],
                }
            )
    return {"schema_items": items, "foreign_keys": []}


TEXT2SQL_DATASETS = ("spider", "bird", "spider2", "beaver", "sciencebenchmark", "entsql", "livesqlbench")
# Benchmarks with a labeled split usable as Stage 1 calibration data.
HAS_TRAIN_SPLIT = {"spider", "bird", "sciencebenchmark"}


def load_text2sql(
    dataset: str,
    split: str = "target",
    limit: int | None = None,
    **layout: Any,
) -> list[Text2SQLItem]:
    if dataset not in {"spider", "bird"}:
        from .benchmarks import load_benchmark

        items = load_benchmark(dataset, split, layout.get("text2sql_root", "datasets/text2sql"), **layout)
        if not items and split == "target":
            raise RuntimeError(f"no usable {dataset}/{split} items found")
        return items[:limit] if limit is not None else items
    roots_only = {k: v for k, v in layout.items() if k in {"text2sql_root", "bird_metadata_root", "bird_database_root"}}
    metadata, roots = dataset_layout(dataset, split, **roots_only)
    if not metadata.is_file():
        raise FileNotFoundError(f"missing metadata: {metadata}")
    with metadata.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    items: list[Text2SQLItem] = []
    for index, row in enumerate(rows):
        try:
            db_path = resolve_database(str(row["db_id"]), roots)
        except FileNotFoundError:
            continue
        items.append(
            Text2SQLItem(
                id=f"{dataset}-{split}-{index}",
                dataset=dataset,
                db_id=str(row["db_id"]),
                question=str(row["question"]),
                gold_sql=str(row["sql"]),
                db_path=db_path,
                schema=row.get("schema") or introspect_schema(str(db_path)),
                evidence=str(row.get("evidence") or ""),
            )
        )
        if limit is not None and len(items) >= limit:
            break
    if not items:
        raise RuntimeError(f"no usable {dataset}/{split} items found")
    return items


def schema_prompt(schema: dict[str, Any]) -> str:
    lines: list[str] = []
    for table in schema.get("schema_items", []):
        names = table.get("column_names", [])
        types = table.get("column_types", [])
        columns = ", ".join(
            f"{name} {types[i] if i < len(types) else ''}".strip() for i, name in enumerate(names)
        )
        lines.append(f"{table.get('table_name', '?')}({columns})")
    return "\n".join(lines)
