"""Load Spider (FusionSQL SFT format) and resolve local SQLite paths.

Each record has: db_id, question, sql (gold), schema (schema_items with names,
types, and sampled column_contents). We remap the record's stale db_path to the
actual database under DATA_ROOT, and take a deterministic subset for a first run.
"""
import json
import os
import random
from typing import List, Dict, Any

from .config import SPIDER_DEV, SPIDER_TRAIN, SPIDER_DB_DIRS


def resolve_db(db_id: str) -> str:
    for root in SPIDER_DB_DIRS:
        for ext in (".sqlite", ".db"):
            p = os.path.join(root, db_id, db_id + ext)
            if os.path.exists(p):
                return p
        # some dbs use a different file name inside the folder
        d = os.path.join(root, db_id)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith((".sqlite", ".db")):
                    return os.path.join(d, f)
    raise FileNotFoundError(f"no sqlite for db_id={db_id} under {SPIDER_DB_DIRS}")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def load_spider_split(split: str, n: int, seed: int = 0) -> List[Dict[str, Any]]:
    """Return n items {id, db_id, question, gold_sql, db_path, schema}.

    Deterministic subset. We stratify lightly by sampling across db_ids so a few
    databases don't dominate the pool run.
    """
    path = SPIDER_DEV if split == "dev" else SPIDER_TRAIN
    raw = _load(path)
    # group by db, round-robin pick to spread across schemas
    by_db: Dict[str, list] = {}
    for i, r in enumerate(raw):
        by_db.setdefault(r["db_id"], []).append((i, r))
    rng = random.Random(seed)
    for v in by_db.values():
        rng.shuffle(v)
    dbs = list(by_db.keys()); rng.shuffle(dbs)
    picked, di = [], 0
    while len(picked) < min(n, len(raw)):
        db = dbs[di % len(dbs)]; di += 1
        if by_db[db]:
            picked.append(by_db[db].pop())
        if all(not by_db[d] for d in dbs):
            break
    picked.sort(key=lambda x: x[0])
    out = []
    for gi, (idx, r) in enumerate(picked):
        out.append(dict(id=f"{split}-{idx}", db_id=r["db_id"], question=r["question"],
                        gold_sql=r["sql"], db_path=resolve_db(r["db_id"]),
                        schema=r["schema"]))
    return out


def schema_to_prompt(schema: Dict[str, Any], with_values: bool = True) -> str:
    """Render schema_items as CREATE TABLE-like text with a couple sample values."""
    lines = []
    for t in schema.get("schema_items", []):
        cols = t["column_names"]; types = t.get("column_types", [""] * len(cols))
        contents = t.get("column_contents", [[]] * len(cols))
        col_strs = []
        for j, c in enumerate(cols):
            ty = types[j] if j < len(types) else ""
            s = f"{c} {ty}".strip()
            if with_values and j < len(contents) and contents[j]:
                sample = ", ".join(str(x) for x in contents[j][:2])
                s += f"  -- e.g. {sample}"
            col_strs.append("    " + s)
        lines.append(f"CREATE TABLE {t['table_name']} (\n" + ",\n".join(col_strs) + "\n);")
    return "\n".join(lines)
