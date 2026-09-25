"""Loaders for Spider 2.0, BEAVER, ScienceBenchmark, EntSQL, and LiveSQLBench.

Each loader reads the benchmark's official release layout under
``<text2sql_root>/<benchmark>/`` (see the README) and returns ``Text2SQLItem`` objects.

* spider2           Spider 2.0-lite local (SQLite) instances; gold SQL, or the official
                    gold result tables scored with the Spider 2.0 matcher.
* beaver            dw, dw_real, nova, neutron; MySQL databases named as in ``db``.
* sciencebenchmark  cordis, oncomx, sdss; PostgreSQL databases named after the domain.
                    ``train`` = the seed and synthetic splits, ``target`` = dev.
* entsql            five enterprise domains, one SQLite file per subdomain; questions in
                    English (or Chinese) with their long document as evidence.  Gold:
                    ``gold/<id>.sql`` or ``gold/<id>.csv`` (EntSQL matcher).
* livesqlbench      LiveSQLBench-Base-Lite, SQLite edition; SELECT-only ("Query") tasks
                    with gold ``sol_sql``.

``database_names`` maps a benchmark database id to the server database name when they
differ.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .data import Text2SQLItem, introspect_schema
from .sql_dialects import locator


def _first(paths: Iterable[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _clip(text: str, limit: int | None) -> str:
    return text if limit is None or len(text) <= limit else text[:limit]


def _spider_schema(tables: Mapping[str, Any]) -> dict[str, Any]:
    """Spider ``tables.json`` entry -> the ``schema_items`` prompt layout."""
    names = tables.get("table_names_original") or tables.get("table_names") or []
    columns = tables.get("column_names_original") or tables.get("column_names") or []
    types = tables.get("column_types") or []
    items = [{"table_name": name, "column_names": [], "column_types": []} for name in names]
    for index, (table, column) in enumerate(columns):
        if table < 0 or table >= len(items):
            continue
        items[table]["column_names"].append(column)
        items[table]["column_types"].append(types[index] if index < len(types) else "")
    return {"schema_items": items, "foreign_keys": tables.get("foreign_keys", [])}


# ---------------------------------------------------------------------------- Spider 2.0


def load_spider2(root: Path, split: str, max_evidence_chars: int | None = None, **_: Any) -> list[Text2SQLItem]:
    if split != "target":
        return []
    base = root / "spider2"
    if (base / "spider2-lite").is_dir():
        base = base / "spider2-lite"
    questions = _first([base / "spider2-lite.jsonl", base / "spider2-lite" / "spider2-lite.jsonl"])
    if questions is None:
        raise FileNotFoundError(f"spider2-lite.jsonl not found under {base}")
    databases = _first([base / "resource" / "databases" / "spider2-localdb", base / "spider2-localdb"])
    if databases is None:
        raise FileNotFoundError(f"spider2-localdb not found under {base}")
    gold = base / "evaluation_suite" / "gold"
    specs = {row["instance_id"]: row for row in _read_jsonl(gold / "spider2lite_eval.jsonl")} if (
        gold / "spider2lite_eval.jsonl"
    ).exists() else {}
    documents = base / "resource" / "documents"
    results = sorted((gold / "exec_result").glob("*.csv")) if (gold / "exec_result").is_dir() else []
    items: list[Text2SQLItem] = []
    for row in _read_jsonl(questions):
        instance = row["instance_id"]
        if not instance.startswith("local"):
            continue  # the local SQLite instances
        db_file = _first([databases / f"{row['db']}.sqlite", databases / f"{row['db']}.db"])
        if db_file is None:
            continue
        sql_path = gold / "sql" / f"{instance}.sql"
        gold_sql = sql_path.read_text(encoding="utf-8").strip() if sql_path.exists() else None
        pattern = re.compile(rf"^{re.escape(instance)}(_[a-z])?\.csv$")
        gold_results = tuple(str(p) for p in results if pattern.match(p.name))
        spec = specs.get(instance, {})
        evidence = ""
        if row.get("external_knowledge"):
            document = documents / row["external_knowledge"]
            if document.exists():
                evidence = _clip(document.read_text(encoding="utf-8"), max_evidence_chars)
        items.append(
            Text2SQLItem(
                id=f"spider2-{instance}",
                dataset="spider2",
                db_id=row["db"],
                question=row["question"],
                gold_sql=gold_sql,
                db_path=db_file,
                schema=introspect_schema(str(db_file)),
                evidence=evidence,
                gold_results=() if gold_sql else gold_results,
                gold_matcher=None if gold_sql else "spider2",
                gold_spec=json.dumps(
                    {"condition_cols": spec.get("condition_cols"), "ignore_order": spec.get("ignore_order", False)}
                ),
            )
        )
    return items


# -------------------------------------------------------------------------------- BEAVER


BEAVER_SPLITS = ("dw", "dw_real", "nova", "neutron")


def load_beaver(
    root: Path, split: str, database_names: Mapping[str, str] | None = None, **_: Any
) -> list[Text2SQLItem]:
    if split != "target":
        return []
    base = root / "beaver"
    names = dict(database_names or {})
    items: list[Text2SQLItem] = []
    schemas: dict[str, dict[str, Any]] = {}
    for part in BEAVER_SPLITS:
        questions = _first([base / part / "dev.json", base / "data" / part / "dev.json"])
        if questions is None:
            continue
        for index, row in enumerate(_read_json(questions)):
            db = str(row.get("db") or ("dw" if part == "dw_real" else part))
            target = locator("mysql", names.get(db, db))
            if target not in schemas:
                schemas[target] = introspect_schema(target)
            knowledge = row.get("domain_knowledge") or []
            evidence = "\n".join(str(k) for k in knowledge) if isinstance(knowledge, list) else str(knowledge)
            items.append(
                Text2SQLItem(
                    id=f"beaver-{part}-{row.get('id', index)}",
                    dataset="beaver",
                    db_id=db,
                    question=row["question"],
                    gold_sql=row.get("sql") or None,
                    db_path=target,
                    schema=schemas[target],
                    evidence=evidence,
                    dialect="mysql",
                )
            )
    return items


# ---------------------------------------------------------------------- ScienceBenchmark


SCIENCE_DOMAINS = ("cordis", "oncomx", "sdss")


def load_sciencebenchmark(
    root: Path, split: str, database_names: Mapping[str, str] | None = None, **_: Any
) -> list[Text2SQLItem]:
    base = root / "sciencebenchmark"
    names = dict(database_names or {})
    files = ("dev.json",) if split == "target" else ("seed.json", "synth.json")
    items: list[Text2SQLItem] = []
    for domain in SCIENCE_DOMAINS:
        directory = base / domain
        if not directory.is_dir():
            continue
        target = locator("postgresql", names.get(domain, domain))
        tables_path = directory / "tables.json"
        schema = None
        if tables_path.exists():
            tables = _read_json(tables_path)
            schema = _spider_schema(tables[0] if isinstance(tables, list) else tables)
        for filename in files:
            path = directory / filename
            if not path.exists():
                continue
            for index, row in enumerate(_read_json(path)):
                items.append(
                    Text2SQLItem(
                        id=f"sciencebenchmark-{domain}-{Path(filename).stem}-{index}",
                        dataset="sciencebenchmark",
                        db_id=domain,
                        question=row["question"],
                        gold_sql=row.get("query") or None,
                        db_path=target,
                        schema=schema or introspect_schema(target),
                        dialect="postgresql",
                    )
                )
    return items


# -------------------------------------------------------------------------------- EntSQL


ENTSQL_DOMAINS = ("Finance", "HR", "Management", "Treasury", "Union")


def load_entsql(
    root: Path, split: str, language: str = "en", max_evidence_chars: int | None = None, **_: Any
) -> list[Text2SQLItem]:
    if split != "target":
        return []
    base = root / "entsql"
    gold = base / "gold"
    items: list[Text2SQLItem] = []
    for domain in ENTSQL_DOMAINS:
        questions_path = base / "JSON" / domain / f"{domain}_{language}.json"
        chinese_path = base / "JSON" / domain / f"{domain}_zh.json"
        if not questions_path.exists():
            continue
        # The Chinese subdomain names are the database file names; question ids are shared.
        database_of: dict[str, str] = {}
        if chinese_path.exists():
            for dom in _read_json(chinese_path)["domains"]:
                for sub in dom["subdomains"]:
                    for question in sub["questions"]:
                        database_of[question["id"]] = sub["subdomain_name"]
        for dom in _read_json(questions_path)["domains"]:
            for sub in dom["subdomains"]:
                for question in sub["questions"]:
                    name = database_of.get(question["id"], sub["subdomain_name"])
                    db_file = _first(
                        [base / "DB" / domain / f"{name}.db", base / "DB" / domain / f"{name}.sqlite",
                         base / domain / f"{name}.db"]
                    )
                    if db_file is None:
                        continue
                    sql_path = gold / f"{question['id']}.sql"
                    csv_path = gold / f"{question['id']}.csv"
                    gold_sql = sql_path.read_text(encoding="utf-8").strip() if sql_path.exists() else None
                    items.append(
                        Text2SQLItem(
                            id=f"entsql-{question['id']}",
                            dataset="entsql",
                            db_id=f"{domain}/{name}",
                            question=question["question"],
                            gold_sql=gold_sql,
                            db_path=db_file,
                            schema=introspect_schema(str(db_file)),
                            evidence=_clip(question.get("long_doc") or "", max_evidence_chars),
                            gold_results=(str(csv_path),) if not gold_sql and csv_path.exists() else (),
                            gold_matcher="entsql" if not gold_sql and csv_path.exists() else None,
                        )
                    )
    return items


# -------------------------------------------------------------------------- LiveSQLBench


def _livesql_knowledge(directory: Path, db: str, ids: Iterable[Any]) -> str:
    path = directory / f"{db}_kb.jsonl"
    if not path.exists():
        return ""
    wanted = {str(i) for i in ids}
    parts = []
    for entry in _read_jsonl(path):
        if str(entry.get("id")) in wanted:
            fields = [entry.get("knowledge"), entry.get("description"), entry.get("definition")]
            parts.append(" ".join(str(f) for f in fields if f))
    return "\n".join(parts)


def load_livesqlbench(root: Path, split: str, max_evidence_chars: int | None = None, **_: Any) -> list[Text2SQLItem]:
    if split != "target":
        return []
    base = root / "livesqlbench"
    data = _first([base / "livesqlbench_data_sqlite.jsonl", base / "livesqlbench_data.jsonl"])
    if data is None:
        raise FileNotFoundError(f"LiveSQLBench task file not found under {base}")
    items: list[Text2SQLItem] = []
    for row in _read_jsonl(data):
        if row.get("category", "Query") != "Query":
            continue  # read-only query tasks
        db = row["selected_database"]
        directory = base / db
        db_file = _first([directory / f"{db}_template.sqlite", directory / f"{db}.sqlite"])
        if db_file is None:
            continue
        solution = row.get("sol_sql") or []
        solution = [solution] if isinstance(solution, str) else list(solution)
        gold_sql = next((sql for sql in solution if sql and str(sql).strip()), None)
        items.append(
            Text2SQLItem(
                id=f"livesqlbench-{row['instance_id']}",
                dataset="livesqlbench",
                db_id=db,
                question=row["query"],
                gold_sql=gold_sql,
                db_path=db_file,
                schema=introspect_schema(str(db_file)),
                evidence=_clip(_livesql_knowledge(directory, db, row.get("external_knowledge") or []), max_evidence_chars),
            )
        )
    return items


LOADERS = {
    "spider2": load_spider2,
    "beaver": load_beaver,
    "sciencebenchmark": load_sciencebenchmark,
    "entsql": load_entsql,
    "livesqlbench": load_livesqlbench,
}


def load_benchmark(dataset: str, split: str, root: str | Path, **options: Any) -> list[Text2SQLItem]:
    if dataset not in LOADERS:
        raise KeyError(f"unknown Text2SQL dataset: {dataset}")
    return LOADERS[dataset](Path(root), split, **options)
