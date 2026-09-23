import sqlite3

from pooleval.databases import build_instances


def test_builds_five_queryable_modified_instances(tmp_path):
    source = tmp_path / "source.sqlite"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE employee(id INTEGER PRIMARY KEY, name TEXT, salary REAL)")
        connection.executemany(
            "INSERT INTO employee VALUES (?, ?, ?)",
            [(i, f"person-{i}", 40_000 + i * 1_000) for i in range(1, 31)],
        )
    records = build_instances(source, tmp_path / "instances")
    assert len(records) == 5
    assert all(record.integrity == "ok" for record in records)
    assert all(record.changed_rows > 0 for record in records)
    for record in records:
        with sqlite3.connect(record.instance) as connection:
            assert connection.execute("SELECT count(*) FROM employee").fetchone()[0] > 0
