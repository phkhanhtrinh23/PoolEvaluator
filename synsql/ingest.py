"""One streaming pass over the 9.3 GB SynSQL data.json -> a compact index.

data.json is a pretty-printed JSON array; because json.dumps escapes newlines, every
field sits on exactly one line, so we can extract the handful of fields we need with
line-prefix matching instead of parsing 9.3 GB of JSON (~40x faster). The huge `cot`
field is skipped entirely.

    python -m synsql.ingest
"""
import json
import os
import sys
import time

from .config import SYNSQL_DATA, SYNSQL_DB_ROOT, INDEX_JSONL, CACHE_ROOT

KEEP = ("db_id", "question", "sql", "sql_complexity", "question_style",
        "external_knowledge")
_PREFIX = {f'"{k}":': k for k in KEEP}


def _available_dbs():
    """db_ids with a real SQLite file on disk (only these can be executed)."""
    out = set()
    for d in os.listdir(SYNSQL_DB_ROOT):
        if os.path.exists(os.path.join(SYNSQL_DB_ROOT, d, d + ".sqlite")):
            out.add(d)
    return out


def build_index(path=SYNSQL_DATA, out_path=INDEX_JSONL, log=print):
    os.makedirs(CACHE_ROOT, exist_ok=True)
    dbs = _available_dbs()
    log(f"[ingest] {len(dbs)} executable databases on disk")
    t0, n_seen, n_kept, rec = time.time(), 0, 0, {}
    with open(path, "r", encoding="utf-8") as f, open(out_path, "w") as out:
        for line in f:
            s = line.lstrip()
            if s.startswith("}"):                      # end of a record
                n_seen += 1
                if rec.get("db_id") in dbs and rec.get("question") and rec.get("sql"):
                    out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    n_kept += 1
                rec = {}
                if n_seen % 500_000 == 0:
                    log(f"[ingest] {n_seen:,} seen / {n_kept:,} kept "
                        f"({time.time()-t0:.0f}s)")
                    sys.stdout.flush()
                continue
            head = s.split(" ", 1)[0]
            k = _PREFIX.get(head)
            if k is None:
                continue
            v = s[len(head) + 1:].rstrip().rstrip(",")
            try:
                rec[k] = json.loads(v)
            except json.JSONDecodeError:
                pass
    log(f"[ingest] done: {n_seen:,} records, {n_kept:,} kept -> {out_path} "
        f"({time.time()-t0:.0f}s)")
    return n_kept


if __name__ == "__main__":
    build_index()
