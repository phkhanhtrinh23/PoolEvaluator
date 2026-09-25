"""PostgreSQL and MySQL execution, schema introspection, and EX-Extended instances.

SQLite databases are addressed by file path.  Server databases are addressed by a
locator, ``postgresql://<database>`` or ``mysql://<database>``; the connection settings
come from the environment: the standard libpq variables (PGHOST, PGPORT, PGUSER,
PGPASSWORD) for PostgreSQL, and POOLEVAL_MYSQL_HOST, POOLEVAL_MYSQL_PORT,
POOLEVAL_MYSQL_USER, POOLEVAL_MYSQL_PASSWORD for MySQL.

The five EX-Extended instances apply the same deterministic edits as the SQLite
instances in ``pooleval.databases``: variants 2 and 5 delete every 10th / 5th row of the
first ``cap_rows`` rows of each table, variants 1, 3, 4 perturb one non-key column on
every 7th / 5th row, and variants 3 and 4 also insert a perturbed clone of the first row.
PostgreSQL instances are ``CREATE DATABASE ... TEMPLATE`` copies; MySQL instances are
table-by-table copies.  Rows are addressed by ``ctid`` (PostgreSQL) or by primary key,
falling back to all columns (MySQL).  A finished instance is marked, so reruns reuse it.
"""

from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import os
import uuid
from typing import Any, Sequence

POSTGRES = "postgresql://"
MYSQL = "mysql://"
_MARK = "pooleval EX-Extended instance {variant}"


def dialect(locator: Any) -> str:
    text = str(locator)
    if text.startswith(POSTGRES):
        return "postgresql"
    if text.startswith(MYSQL):
        return "mysql"
    return "sqlite"


def database_name(locator: Any) -> str:
    text = str(locator)
    for prefix in (POSTGRES, MYSQL):
        if text.startswith(prefix):
            return text[len(prefix):]
    raise ValueError(f"{locator!r} is not a server database locator")


def locator(kind: str, name: str) -> str:
    return (POSTGRES if kind == "postgresql" else MYSQL) + name


def instance_name(database: str, variant: int) -> str:
    """Instance database name, kept within the 63/64-byte identifier limits."""
    name = f"{database}__pe{variant}"
    if len(name.encode()) <= 60:
        return name
    digest = hashlib.sha1(database.encode()).hexdigest()[:8]
    return f"{database.encode()[:40].decode(errors='ignore')}_{digest}__pe{variant}"


# ----------------------------------------------------------------------- connections


def _pg_connect(database: str, autocommit: bool = False) -> Any:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("PostgreSQL benchmarks need `pip install psycopg[binary]`") from exc
    return psycopg.connect(dbname=database, autocommit=autocommit, connect_timeout=15)


def _mysql_connect(database: str | None = None) -> Any:
    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError("MySQL benchmarks need `pip install pymysql`") from exc
    return pymysql.connect(
        host=os.environ.get("POOLEVAL_MYSQL_HOST", "localhost"),
        port=int(os.environ.get("POOLEVAL_MYSQL_PORT", "3306")),
        user=os.environ.get("POOLEVAL_MYSQL_USER", "root"),
        password=os.environ.get("POOLEVAL_MYSQL_PASSWORD", ""),
        database=database,
        charset="utf8mb4",
        autocommit=True,
    )


def _pg_ident(*parts: str) -> str:
    return ".".join('"' + part.replace('"', '""') + '"' for part in parts)


def _my_ident(*parts: str) -> str:
    return ".".join("`" + part.replace("`", "``") + "`" for part in parts)


# ------------------------------------------------------------------------- execution


def execute_rows(target: Any, sql: str, timeout: float) -> tuple[list[str], list[tuple[Any, ...]]] | None:
    """Run one read-only query; return (column names, rows) or None on any failure."""
    kind = dialect(target)
    name = database_name(target)
    milliseconds = max(1, int(timeout * 1000))
    if kind == "postgresql":
        import psycopg

        try:
            with _pg_connect(name) as connection:
                connection.read_only = True
                with connection.cursor() as cursor:
                    cursor.execute(f"SET statement_timeout = {milliseconds}")
                    cursor.execute(sql)
                    if cursor.description is None:
                        return None
                    columns = [column.name for column in cursor.description]
                    rows = [tuple(row) for row in cursor.fetchall()]
                connection.rollback()
            return columns, rows
        except psycopg.Error:
            return None
    import pymysql

    try:
        connection = _mysql_connect(name)
    except pymysql.MySQLError:
        return None
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SET SESSION MAX_EXECUTION_TIME = {milliseconds}")
            cursor.execute("START TRANSACTION READ ONLY")
            cursor.execute(sql)
            if cursor.description is None:
                return None
            columns = [column[0] for column in cursor.description]
            rows = [tuple(row) for row in cursor.fetchall()]
            connection.rollback()
        return columns, rows
    except pymysql.MySQLError:
        return None
    finally:
        connection.close()


def introspect(target: Any) -> dict[str, Any]:
    """Tables and columns in the ``schema_items`` layout used for prompts."""
    kind = dialect(target)
    name = database_name(target)
    tables: dict[str, dict[str, list[str]]] = {}
    if kind == "postgresql":
        query = (
            "SELECT table_schema, table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
            "ORDER BY table_schema, table_name, ordinal_position"
        )
        with _pg_connect(name) as connection:
            rows = connection.execute(query).fetchall()
        for schema, table, column, kind_ in rows:
            key = table if schema == "public" else f"{schema}.{table}"
            entry = tables.setdefault(key, {"column_names": [], "column_types": []})
            entry["column_names"].append(column)
            entry["column_types"].append(kind_)
    else:
        connection = _mysql_connect(name)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT table_name, column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = DATABASE() ORDER BY table_name, ordinal_position"
                )
                rows = cursor.fetchall()
        finally:
            connection.close()
        for table, column, kind_ in rows:
            entry = tables.setdefault(table, {"column_names": [], "column_types": []})
            entry["column_names"].append(column)
            entry["column_types"].append(kind_)
    items = [{"table_name": table, **columns} for table, columns in tables.items()]
    return {"schema_items": items, "foreign_keys": []}


# ------------------------------------------------------------------------- mutations


def mutated(value: Any, salt: int) -> Any:
    """The SQLite instance perturbation, extended to the typed values servers return."""
    from .databases import _mutated

    if isinstance(value, bool):
        return not value
    if isinstance(value, decimal.Decimal):
        return value + decimal.Decimal(1 + salt % 7)
    if isinstance(value, dt.datetime):
        return value + dt.timedelta(days=1 + salt % 7, minutes=salt % 53)
    if isinstance(value, dt.date):
        return value + dt.timedelta(days=1 + salt % 7)
    if isinstance(value, dt.time):
        moved = dt.datetime.combine(dt.date(2000, 1, 1), value) + dt.timedelta(minutes=1 + salt % 53)
        return moved.time()
    if isinstance(value, dt.timedelta):
        return value + dt.timedelta(seconds=1 + salt % 53)
    if isinstance(value, uuid.UUID):
        return uuid.uuid5(value, str(salt))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value) + f"pe{salt % 997}".encode()
    if isinstance(value, (dict, list)):
        return value
    return _mutated(value, salt)


def plan_mutations(
    rows: Sequence[tuple[Any, ...]], key_columns: Sequence[bool], variant: int
) -> list[tuple[Any, ...]]:
    """Edits for one table, mirroring ``pooleval.databases._mutate_table``.

    ``rows`` are the table's first rows (column values only); ``key_columns`` flags the
    primary-key columns.  Returns ("delete", index), ("update", index, column, value)
    and ("insert", values) operations, applied in order.
    """
    if not rows:
        return []
    if variant in {2, 5}:
        step = 10 if variant == 2 else 5
        return [("delete", index) for index in range(len(rows)) if index % step == 0]
    columns = list(range(len(key_columns)))
    mutable = [c for c in columns if not key_columns[c]] or columns
    column = mutable[(variant - 1) % len(mutable)]
    stride = 7 if variant == 1 else 5
    operations: list[tuple[Any, ...]] = [
        ("update", index, column, mutated(rows[index][column], variant * 101 + index))
        for index in range(len(rows))
        if index % stride == 0
    ]
    if variant in {3, 4}:
        values = list(rows[0])
        for c in columns:
            if key_columns[c] or c == column:
                values[c] = mutated(values[c], variant * 313 + c)
        operations.append(("insert", tuple(values)))
    return operations


# ---------------------------------------------------------------------------- PostgreSQL


def _pg_tables(connection: Any) -> list[tuple[str, str]]:
    return connection.execute(
        "SELECT table_schema, table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE' "
        "AND table_schema NOT IN ('pg_catalog', 'information_schema') ORDER BY 1, 2"
    ).fetchall()


def _pg_mutate(database: str, variant: int, cap_rows: int) -> int:
    import psycopg

    changed = 0
    with _pg_connect(database, autocommit=True) as connection:
        try:
            connection.execute("SET session_replication_role = replica")  # FK checks off, as in SQLite
        except psycopg.Error:
            pass
        for schema, table in _pg_tables(connection):
            qualified = _pg_ident(schema, table)
            try:
                columns = connection.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema = %s "
                    "AND table_name = %s ORDER BY ordinal_position",
                    (schema, table),
                ).fetchall()
                keys = {
                    row[0]
                    for row in connection.execute(
                        "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid "
                        "AND a.attnum = ANY(i.indkey) WHERE i.indrelid = %s::regclass AND i.indisprimary",
                        (qualified,),
                    ).fetchall()
                }
                fetched = connection.execute(
                    f"SELECT ctid::text, * FROM {qualified} ORDER BY ctid LIMIT %s", (cap_rows,)
                ).fetchall()
            except psycopg.Error:
                continue
            names = [row[0] for row in columns]
            ctids = [row[0] for row in fetched]
            rows = [tuple(row[1:]) for row in fetched]
            for operation in plan_mutations(rows, [name in keys for name in names], variant):
                try:
                    if operation[0] == "delete":
                        cursor = connection.execute(f"DELETE FROM {qualified} WHERE ctid = %s::tid", (ctids[operation[1]],))
                    elif operation[0] == "update":
                        _, index, column, value = operation
                        cursor = connection.execute(
                            f"UPDATE {qualified} SET {_pg_ident(names[column])} = %s WHERE ctid = %s::tid",
                            (value, ctids[index]),
                        )
                    else:
                        marks = ", ".join(["%s"] * len(names))
                        cursor = connection.execute(
                            f"INSERT INTO {qualified} ({', '.join(_pg_ident(n) for n in names)}) "
                            f"VALUES ({marks}) ON CONFLICT DO NOTHING",
                            operation[1],
                        )
                    changed += max(cursor.rowcount, 0)
                except psycopg.Error:
                    continue
    return changed


def _pg_instances(database: str, count: int, cap_rows: int) -> list[tuple[str, int]]:
    records: list[tuple[str, int]] = []
    with _pg_connect(os.environ.get("POOLEVAL_PG_ADMIN_DB", "postgres"), autocommit=True) as admin:
        for variant in range(1, count + 1):
            name = instance_name(database, variant)
            mark = _MARK.format(variant=variant)
            row = admin.execute(
                "SELECT shobj_description(oid, 'pg_database') FROM pg_database WHERE datname = %s", (name,)
            ).fetchone()
            if row is not None and row[0] == mark:
                records.append((locator("postgresql", name), -1))
                continue
            if row is not None:
                admin.execute(f"DROP DATABASE {_pg_ident(name)}")
            admin.execute(f"CREATE DATABASE {_pg_ident(name)} TEMPLATE {_pg_ident(database)}")
            changed = _pg_mutate(name, variant, cap_rows)
            admin.execute(f"COMMENT ON DATABASE {_pg_ident(name)} IS '{mark}'")
            records.append((locator("postgresql", name), changed))
    return records


# --------------------------------------------------------------------------------- MySQL


def _my_mutate(connection: Any, database: str, variant: int, cap_rows: int) -> int:
    import pymysql

    changed = 0
    with connection.cursor() as cursor:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s "
            "AND table_type = 'BASE TABLE' ORDER BY table_name",
            (database,),
        )
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            qualified = _my_ident(database, table)
            try:
                cursor.execute(
                    "SELECT column_name, column_key FROM information_schema.columns WHERE table_schema = %s "
                    "AND table_name = %s ORDER BY ordinal_position",
                    (database, table),
                )
                meta = cursor.fetchall()
                names = [row[0] for row in meta]
                keys = [row[1] == "PRI" for row in meta]
                order = ", ".join(_my_ident(n) for n, k in zip(names, keys) if k)
                cursor.execute(
                    f"SELECT * FROM {qualified}" + (f" ORDER BY {order}" if order else "") + " LIMIT %s",
                    (cap_rows,),
                )
                rows = [tuple(row) for row in cursor.fetchall()]
            except pymysql.MySQLError:
                continue
            match = [i for i, k in enumerate(keys) if k] or list(range(len(names)))

            def where(row: tuple[Any, ...]) -> tuple[str, list[Any]]:
                clause = " AND ".join(f"{_my_ident(names[i])} <=> %s" for i in match)
                return clause, [row[i] for i in match]

            for operation in plan_mutations(rows, keys, variant):
                try:
                    if operation[0] == "delete":
                        clause, params = where(rows[operation[1]])
                        changed += cursor.execute(f"DELETE FROM {qualified} WHERE {clause} LIMIT 1", params)
                    elif operation[0] == "update":
                        _, index, column, value = operation
                        clause, params = where(rows[index])
                        changed += cursor.execute(
                            f"UPDATE {qualified} SET {_my_ident(names[column])} = %s WHERE {clause} LIMIT 1",
                            [value, *params],
                        )
                    else:
                        marks = ", ".join(["%s"] * len(names))
                        changed += cursor.execute(
                            f"INSERT IGNORE INTO {qualified} ({', '.join(_my_ident(n) for n in names)}) "
                            f"VALUES ({marks})",
                            list(operation[1]),
                        )
                except pymysql.MySQLError:
                    continue
    return changed


def _my_instances(database: str, count: int, cap_rows: int) -> list[tuple[str, int]]:
    records: list[tuple[str, int]] = []
    connection = _mysql_connect(None)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s "
                "AND table_type = 'BASE TABLE' ORDER BY table_name",
                (database,),
            )
            tables = [row[0] for row in cursor.fetchall()]
            for variant in range(1, count + 1):
                name = instance_name(database, variant)
                mark = _MARK.format(variant=variant)
                cursor.execute(
                    "SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s", (name,)
                )
                if cursor.fetchone() is not None:
                    try:
                        cursor.execute(f"SELECT note FROM {_my_ident(name, '__pooleval_instance')}")
                        row = cursor.fetchone()
                    except Exception:
                        row = None
                    if row is not None and row[0] == mark:
                        records.append((locator("mysql", name), -1))
                        continue
                    cursor.execute(f"DROP DATABASE {_my_ident(name)}")
                cursor.execute(f"CREATE DATABASE {_my_ident(name)}")
                cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                for table in tables:
                    cursor.execute(f"CREATE TABLE {_my_ident(name, table)} LIKE {_my_ident(database, table)}")
                    cursor.execute(f"INSERT INTO {_my_ident(name, table)} SELECT * FROM {_my_ident(database, table)}")
                changed = _my_mutate(connection, name, variant, cap_rows)
                cursor.execute(f"CREATE TABLE {_my_ident(name, '__pooleval_instance')} (note VARCHAR(64))")
                cursor.execute(f"INSERT INTO {_my_ident(name, '__pooleval_instance')} VALUES (%s)", (mark,))
                records.append((locator("mysql", name), changed))
    finally:
        connection.close()
    return records


def build_server_instances(target: Any, count: int = 5, cap_rows: int = 100) -> list[tuple[str, int]]:
    """Create (or reuse) the ``count`` EX-Extended instances of a server database.

    Returns (instance locator, changed rows) pairs; changed rows is -1 for a reused
    instance.
    """
    kind = dialect(target)
    name = database_name(target)
    if kind == "postgresql":
        return _pg_instances(name, count, cap_rows)
    return _my_instances(name, count, cap_rows)
