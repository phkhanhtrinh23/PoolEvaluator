"""Multi-dataset loaders for the real zoo.

Every loader returns a list of items in ONE canonical shape, so the rest of the
pipeline (prompt -> generate -> execute -> build PoolRun) is dataset-agnostic:

    {id, db_id, question, gold_sql, db_path, schema}

`schema` is a FusionSQL-style dict ({schema_items:[{table_name, column_names,
column_types, column_contents}], foreign_keys}). Datasets that already ship it
(Spider/BIRD FusionSQL SFT) pass it through; datasets that don't (HuggingFace sets)
get it by INTROSPECTING the live SQLite DB, so the same prompt path works everywhere.

Registry `DATASETS` maps a name -> config. `load_split(name, split, n, seed)` returns
a deterministic, schema-spread subset; `split="source"` supplies the labeled split
used for the seen prior (a real train file when available, else a disjoint slice of
the same pool).
"""
import json
import os
import random
import sqlite3
from functools import lru_cache
from typing import List, Dict, Any

from .config import SPIDER_DEV, SPIDER_TRAIN, SPIDER_DB_DIRS


# --------------------------------------------------------------------------- #
#  DB resolution + schema introspection                                        #
# --------------------------------------------------------------------------- #
def resolve_db(db_id: str, db_roots: List[str]) -> str:
    for root in db_roots:
        for ext in (".sqlite", ".db"):
            p = os.path.join(root, db_id, db_id + ext)
            if os.path.exists(p):
                return p
        d = os.path.join(root, db_id)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith((".sqlite", ".db")):
                    return os.path.join(d, f)
    raise FileNotFoundError(f"no sqlite for db_id={db_id} under {db_roots}")


@lru_cache(maxsize=256)
def introspect_schema(db_path: str, n_samples: int = 2) -> Dict[str, Any]:
    """Build a FusionSQL-style schema dict from a live SQLite database (tables,
    columns, types, and a couple of sample values per column)."""
    con = sqlite3.connect(db_path)
    con.text_factory = lambda b: b.decode("utf-8", "ignore")
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cur.fetchall()]
    items = []
    for t in tables:
        try:
            cur.execute(f'PRAGMA table_info("{t}")')
            info = cur.fetchall()             # (cid, name, type, notnull, dflt, pk)
        except sqlite3.Error:
            continue
        names = [c[1] for c in info]
        types = [c[2] for c in info]
        contents = []
        for name in names:
            try:
                cur.execute(f'SELECT DISTINCT "{name}" FROM "{t}" '
                            f'WHERE "{name}" IS NOT NULL LIMIT {n_samples}')
                contents.append([str(r[0])[:40] for r in cur.fetchall()])
            except sqlite3.Error:
                contents.append([])
        items.append(dict(table_name=t, column_names=names, column_types=types,
                          column_contents=contents))
    con.close()
    return dict(schema_items=items, foreign_keys=[])


# --------------------------------------------------------------------------- #
#  Deterministic schema-spread subset                                          #
# --------------------------------------------------------------------------- #
def _spread_subset(records, key_db, n, seed):
    by_db: Dict[str, list] = {}
    for i, r in enumerate(records):
        by_db.setdefault(key_db(r), []).append((i, r))
    rng = random.Random(seed)
    for v in by_db.values():
        rng.shuffle(v)
    dbs = list(by_db.keys()); rng.shuffle(dbs)
    picked, di = [], 0
    while len(picked) < min(n, len(records)):
        db = dbs[di % len(dbs)]; di += 1
        if by_db[db]:
            picked.append(by_db[db].pop())
        if all(not by_db[d] for d in dbs):
            break
    picked.sort(key=lambda x: x[0])
    return picked


# --------------------------------------------------------------------------- #
#  FusionSQL loader (Spider / BIRD SFT json)                                    #
# --------------------------------------------------------------------------- #
def _load_fusionsql(cfg, split, n, seed):
    path = cfg["dev"] if split in ("dev", "target") else cfg["train"]
    with open(path) as f:
        raw = json.load(f)
    picked = _spread_subset(raw, lambda r: r["db_id"], n, seed)
    out = []
    for idx, r in picked:
        db_path = resolve_db(r["db_id"], cfg["db_roots"])
        sch = r.get("schema") or introspect_schema(db_path)
        out.append(dict(id=f"{split}-{idx}", db_id=r["db_id"], question=r["question"],
                        gold_sql=r["sql"], db_path=db_path, schema=sch))
    return out


# --------------------------------------------------------------------------- #
#  HuggingFace-arrow loaders (introspect schema from the DB)                    #
# --------------------------------------------------------------------------- #
def _hf_records(cfg):
    """Return a flat list of normalized dicts {db_id, question, gold_sql} from the
    dataset's cached arrow file(s)."""
    from datasets import Dataset
    import glob
    files = sorted(glob.glob(cfg["arrow_glob"], recursive=True))
    if cfg.get("arrow_pick"):
        files = [f for f in files if cfg["arrow_pick"] in f] or files
    recs = []
    q, s, d = cfg["col_q"], cfg["col_sql"], cfg["col_db"]
    for f in files:
        ds = Dataset.from_file(f)
        for r in ds:
            if not r.get(q) or not r.get(s) or not r.get(d):
                continue
            recs.append(dict(db_id=str(r[d]), question=str(r[q]), sql=str(r[s])))
    return recs


def _partition(ok, split, n, seed):
    """Deterministic source/target split for datasets without a separate train file.
    Target = a held-out front slice; source = the disjoint remainder (labeled items
    used only for the seen prior). Schema is introspected from each DB."""
    rng = random.Random(seed + 991)
    order = list(range(len(ok))); rng.shuffle(order)
    cut = min(max(n, 1), int(len(ok) * 0.7)) if split in ("dev", "target") \
        else len(ok)
    if split in ("dev", "target"):
        pool = [ok[i] for i in order[:max(cut, n)]]
    else:
        pool = [ok[i] for i in order[int(len(ok) * 0.7):]] or \
               [ok[i] for i in order]      # tiny sets: fall back to whole pool
    picked = _spread_subset(pool, lambda r: r["db_id"], n, seed)
    out = []
    for idx, r in picked:
        out.append(dict(id=f"{split}-{idx}", db_id=r["db_id"], question=r["question"],
                        gold_sql=r["sql"], db_path=r["_db_path"],
                        schema=introspect_schema(r["_db_path"])))
    return out


def _load_hf(cfg, split, n, seed):
    recs = _hf_records(cfg)
    ok = []
    for r in recs:
        try:
            r["_db_path"] = resolve_db(r["db_id"], cfg["db_roots"])
            ok.append(r)
        except FileNotFoundError:
            continue
    return _partition(ok, split, n, seed)


# --------------------------------------------------------------------------- #
#  Spider 2.0-lite (local SQLite subset)                                        #
# --------------------------------------------------------------------------- #
_SPIDER2 = "/mnt/win_d/Multilingual_Text_to_SQL_code_backend/Spider2/spider2-lite"


def _load_spider2_local(cfg, split, n, seed):
    """Local (SQLite) Spider 2.0-lite instances that have a gold SQL and a resolvable
    database. External-knowledge docs (when referenced) are appended to the question,
    as the benchmark intends. bq*/sf* instances (cloud warehouses) are handled by a
    separate runner."""
    gold_dir = os.path.join(_SPIDER2, "evaluation_suite", "gold", "sql")
    db_dir = os.path.join(_SPIDER2, "resource", "databases", "spider2-localdb")
    docs = os.path.join(_SPIDER2, "resource", "documents")
    ok = []
    with open(os.path.join(_SPIDER2, "spider2-lite.jsonl")) as f:
        for line in f:
            it = json.loads(line)
            iid = it["instance_id"]
            if not iid.startswith("local"):
                continue
            gp = os.path.join(gold_dir, iid + ".sql")
            dbp = os.path.join(db_dir, it["db"] + ".sqlite")
            if not (os.path.exists(gp) and os.path.exists(dbp)):
                continue
            q = it["question"]
            ek = it.get("external_knowledge")
            if ek and os.path.exists(os.path.join(docs, ek)):
                with open(os.path.join(docs, ek)) as d:
                    q += "\n\n[External knowledge]\n" + d.read()[:2000]
            with open(gp) as g:
                ok.append(dict(db_id=it["db"], question=q, sql=g.read().strip(),
                              _db_path=dbp))
    return _partition(ok, split, n, seed)


# --------------------------------------------------------------------------- #
#  Registry                                                                     #
# --------------------------------------------------------------------------- #
# NOTE: the databases under data_FusionSQL/bird are git-LFS pointer stubs; the REAL
# BIRD SQLite files live under data/sft_data_collections (this is what the FusionSQL
# records' stale "./data/..." db_path actually resolves to).
_BIRD_DBS = ["/mnt/win_d/data/sft_data_collections/bird/dev/dev_databases",
             "/mnt/win_d/data/sft_data_collections/bird/train/train_databases"]
_LIVE_DBS = ["/mnt/win_d/skill_driven_agent_view/livesqlbench"]

DATASETS: Dict[str, Dict[str, Any]] = {
    "spider": dict(kind="fusionsql", dev=SPIDER_DEV, train=SPIDER_TRAIN,
                   db_roots=SPIDER_DB_DIRS),
    "bird": dict(kind="fusionsql",
                 dev="/mnt/win_d/data/sft_bird_dev_text2sql.json",
                 train="/mnt/win_d/data/sft_bird_train_text2sql.json",
                 db_roots=_BIRD_DBS),
    "bird_minidev": dict(
        kind="hf",
        arrow_glob="/mnt/win_d/datasets/birdsql___bird_mini_dev/**/*sqlite*.arrow",
        col_q="question", col_sql="SQL", col_db="db_id",
        db_roots=_BIRD_DBS),
    "livesqlbench": dict(
        kind="hf",
        arrow_glob="/mnt/win_d/datasets/birdsql___livesqlbench-base-lite-sqlite/**/*.arrow",
        col_q="query", col_sql="sol_sql", col_db="selected_database",
        db_roots=_LIVE_DBS),
    "sqlflow": dict(
        kind="hf",
        arrow_glob="/mnt/win_d/datasets/debugger123___sql_flow/**/*bird*.arrow",
        col_q="question", col_sql="SQL", col_db="db_id",
        db_roots=_BIRD_DBS),
    "spider2local": dict(kind="spider2local"),
}


def load_split(name: str, split: str, n: int, seed: int = 0):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name}; have {list(DATASETS)}")
    cfg = DATASETS[name]
    if cfg["kind"] == "fusionsql":
        return _load_fusionsql(cfg, split, n, seed)
    if cfg["kind"] == "spider2local":
        return _load_spider2_local(cfg, split, n, seed)
    return _load_hf(cfg, split, n, seed)
