import json
import sqlite3

import numpy as np
import pytest

from pooleval.result_match import entsql_match, spider2_match


def _sqlite(path, rows):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
        connection.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
    return path


def test_spider2_matcher_follows_official_rules(tmp_path):
    gold = tmp_path / "local001.csv"
    gold.write_text("name,score\nann,1.0\nbob,2.004\n")
    assert spider2_match(["n", "s"], [("bob", 2.0), ("ann", 1.0)], [gold], ignore_order=True)
    assert not spider2_match(["n", "s"], [("bob", 2.0), ("ann", 1.0)], [gold], ignore_order=False)
    # condition_cols: only the second gold column must be matched; extra predicted columns are fine.
    assert spider2_match(["x", "s", "y"], [(9, 1.0, "z"), (8, 2.0, "w")], [gold], condition_cols=[1])
    other = tmp_path / "local001_b.csv"
    other.write_text("name\ncarl\n")
    assert spider2_match(["n"], [("carl",)], [gold, other])


def test_entsql_matcher_requires_each_gold_row_in_a_distinct_result_row(tmp_path):
    gold = tmp_path / "HR-1.csv"
    gold.write_text("dept,ratio\nNorth,12.5%\nSouth,\"1,000\"\n")
    assert entsql_match([("North", 12.5, "x"), ("South", 1000)], gold)
    assert not entsql_match([("North", 12.5, "South", 1000)], gold)
    assert entsql_match([("North", 12.51), ("South", 1000.0)], gold)


def test_spider2_loader_reads_local_instances(tmp_path):
    from pooleval.benchmarks import load_benchmark

    base = tmp_path / "spider2" / "spider2-lite"
    (base / "resource" / "databases" / "spider2-localdb").mkdir(parents=True)
    (base / "resource" / "documents").mkdir(parents=True)
    (base / "evaluation_suite" / "gold" / "sql").mkdir(parents=True)
    (base / "evaluation_suite" / "gold" / "exec_result").mkdir(parents=True)
    _sqlite(base / "resource" / "databases" / "spider2-localdb" / "shop.sqlite", [(1, "a", 1.0)])
    (base / "resource" / "documents" / "rules.md").write_text("Score means rating.")
    rows = [
        {"instance_id": "local001", "db": "shop", "question": "q1", "external_knowledge": "rules.md"},
        {"instance_id": "local002", "db": "shop", "question": "q2", "external_knowledge": None},
        {"instance_id": "bq001", "db": "cloud", "question": "q3", "external_knowledge": None},
    ]
    (base / "spider2-lite.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (base / "evaluation_suite" / "gold" / "sql" / "local001.sql").write_text("SELECT name FROM t")
    for name in ("local002_a.csv", "local002_b.csv"):
        (base / "evaluation_suite" / "gold" / "exec_result" / name).write_text("name\na\n")
    (base / "evaluation_suite" / "gold" / "spider2lite_eval.jsonl").write_text(
        json.dumps({"instance_id": "local002", "condition_cols": [0], "ignore_order": True})
    )
    items = load_benchmark("spider2", "target", tmp_path)
    assert [item.id for item in items] == ["spider2-local001", "spider2-local002"]
    assert items[0].gold_sql == "SELECT name FROM t" and items[0].evidence == "Score means rating."
    assert items[1].gold_sql is None and items[1].gold_matcher == "spider2" and len(items[1].gold_results) == 2
    assert json.loads(items[1].gold_spec) == {"condition_cols": [0], "ignore_order": True}


def test_entsql_loader_maps_questions_to_subdomain_databases(tmp_path):
    from pooleval.benchmarks import load_benchmark

    base = tmp_path / "entsql"
    (base / "JSON" / "HR").mkdir(parents=True)
    (base / "DB" / "HR").mkdir(parents=True)
    (base / "gold").mkdir()
    _sqlite(base / "DB" / "HR" / "基础人事.db", [(1, "a", 1.0)])
    questions = [{"id": "HR-BasicHR-001", "question": "ratio?", "long_doc": "doc"}]
    for language, name in (("en", "BasicHR"), ("zh", "基础人事")):
        document = {"domains": [{"subdomains": [{"subdomain_name": name, "questions": questions}]}]}
        (base / "JSON" / "HR" / f"HR_{language}.json").write_text(json.dumps(document, ensure_ascii=False))
    (base / "gold" / "HR-BasicHR-001.csv").write_text("x\n1\n")
    items = load_benchmark("entsql", "target", tmp_path)
    assert len(items) == 1 and items[0].db_id == "HR/基础人事" and items[0].evidence == "doc"
    assert items[0].gold_matcher == "entsql" and items[0].has_gold


def test_livesqlbench_loader_keeps_query_tasks_with_knowledge(tmp_path):
    from pooleval.benchmarks import load_benchmark

    base = tmp_path / "livesqlbench"
    (base / "alien").mkdir(parents=True)
    _sqlite(base / "alien" / "alien_template.sqlite", [(1, "a", 1.0)])
    (base / "alien" / "alien_kb.jsonl").write_text(json.dumps({"id": 3, "knowledge": "SNQI", "definition": "ratio"}))
    rows = [
        {"instance_id": "alien_1", "selected_database": "alien", "query": "q", "category": "Query",
         "sol_sql": ["SELECT 1"], "external_knowledge": [3]},
        {"instance_id": "alien_2", "selected_database": "alien", "query": "fix", "category": "Management",
         "sol_sql": ["UPDATE t SET name='x'"], "external_knowledge": []},
    ]
    (base / "livesqlbench_data_sqlite.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    items = load_benchmark("livesqlbench", "target", tmp_path)
    assert [item.id for item in items] == ["livesqlbench-alien_1"]
    assert items[0].gold_sql == "SELECT 1" and items[0].evidence == "SNQI ratio"


def test_sciencebenchmark_loader_uses_postgres_locators_and_tables_json(tmp_path):
    from pooleval.benchmarks import load_benchmark

    base = tmp_path / "sciencebenchmark" / "cordis"
    base.mkdir(parents=True)
    tables = [{"db_id": "cordis", "table_names_original": ["projects"], "column_names_original": [[-1, "*"], [0, "title"]],
               "column_types": ["text", "text"], "foreign_keys": []}]
    (base / "tables.json").write_text(json.dumps(tables))
    (base / "dev.json").write_text(json.dumps([{"db_id": "cordis", "question": "q", "query": "SELECT 1"}]))
    (base / "seed.json").write_text(json.dumps([{"db_id": "cordis", "question": "s", "query": "SELECT 2"}]))
    target = load_benchmark("sciencebenchmark", "target", tmp_path, database_names={"cordis": "cordis_db"})
    train = load_benchmark("sciencebenchmark", "train", tmp_path)
    assert target[0].db_path == "postgresql://cordis_db" and target[0].dialect == "postgresql"
    assert target[0].schema["schema_items"][0] == {"table_name": "projects", "column_names": ["title"], "column_types": ["text"]}
    assert [item.gold_sql for item in train] == ["SELECT 2"]


def test_server_mutation_plan_matches_the_sqlite_instances(tmp_path):
    from pooleval.databases import build_instances
    from pooleval.sql_dialects import plan_mutations

    rows = [(i, f"name{i}", float(i)) for i in range(1, 23)]
    source = _sqlite(tmp_path / "db.sqlite", rows)
    records = build_instances(source, tmp_path / "out", cap_rows_per_table=100)
    for record in records:
        expected = [list(r) for r in rows]
        for op in plan_mutations(rows, [True, False, False], record.variant):
            if op[0] == "delete":
                expected[op[1]] = None
            elif op[0] == "update":
                expected[op[1]][op[2]] = op[3]
            elif all(row is None or row[0] != op[1][0] for row in expected):
                expected.append(list(op[1]))  # insert-or-ignore on the primary key
        with sqlite3.connect(record.instance) as connection:
            actual = [list(r) for r in connection.execute("SELECT id, name, score FROM t ORDER BY id")]
        assert actual == sorted([r for r in expected if r is not None])


def test_cached_judge_reuses_votes_and_reports_original_time(tmp_path):
    from pooleval.judges import CachedJudge, JudgeEnsemble

    class Counting:
        name = "j"
        calls = 0

        def choose(self, item):
            Counting.calls += 1
            return 1

    judge = CachedJudge(JudgeEnsemble([Counting()]), lambda item, c: ("item", item, tuple(c)), "t", tmp_path / "c.json")
    assert judge(3, ["a", "b"], [0.1, 0.2]) == "b"
    first = judge.last_seconds
    assert judge(3, ["a", "b"], [0.9, 0.2]) == "b" and Counting.calls == 1 and judge.last_seconds == first
    reloaded = CachedJudge(JudgeEnsemble([Counting()]), lambda item, c: None, "t", tmp_path / "c.json")
    assert reloaded(3, ["a", "b"], [0.1, 0.2]) == "b" and Counting.calls == 1
    assert np.isfinite(first)
