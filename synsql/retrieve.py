"""Retrieve the SynSQL subsets closest to an UNLABELED target set.

The target is unlabeled, so the representation may only use what a deployed operator
actually has: the natural-language question and the database schema. No gold SQL, no
model outputs, no labels.

    phi(item)  = TF-IDF over  question tokens  ++  schema tokens (table/column names)
    phi(set)   = L2-normalized mean of phi over the set          (kernel mean embedding)
    d(S, T)    = 1 - cos(phi(S), phi(T))                          [primary]
    MMD(S, T)  = || phi(S) - phi(T) ||_2                          [linear-kernel MMD]

Because phi(set) is a kernel mean embedding, ||phi(S)-phi(T)|| IS the linear-kernel
MMD -- so the primary distance is a proper two-sample distance, not an ad-hoc score.
"""
import json
import os
import re
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import SYNSQL_DB_ROOT, CACHE_ROOT

_TOK = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _split_ident(name):
    """table/column identifier -> words ('user_id', 'userID' -> 'user id')."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name)).replace("_", " ")
    return s.lower()


@lru_cache(maxsize=4096)
def synsql_schema_text(db_id):
    """Schema token bag for a SynSQL database (table + column names)."""
    import sqlite3
    p = os.path.join(SYNSQL_DB_ROOT, db_id, db_id + ".sqlite")
    if not os.path.exists(p):
        return _split_ident(db_id)
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'")
        parts = [_split_ident(db_id)]
        for (t,) in cur.fetchall():
            parts.append(_split_ident(t))
            try:
                cur.execute(f'PRAGMA table_info("{t}")')
                parts += [_split_ident(c[1]) for c in cur.fetchall()]
            except sqlite3.Error:
                pass
        con.close()
        return " ".join(parts)
    except sqlite3.Error:
        return _split_ident(db_id)


def schema_text_from_dict(schema):
    """Schema token bag from a FusionSQL-style schema dict (targets)."""
    parts = []
    for t in schema.get("schema_items", []):
        parts.append(_split_ident(t["table_name"]))
        parts += [_split_ident(c) for c in t["column_names"]]
    return " ".join(parts)


_SCHEMA_CACHE = os.path.join(CACHE_ROOT, "db_schema_text.json")


def schema_text_map(db_ids, log=print):
    """db_id -> schema token bag, built once for the whole corpus and cached to disk.
    (Per-item introspection would open 400k SQLite connections; this opens 16.5k.)"""
    cache = {}
    if os.path.exists(_SCHEMA_CACHE):
        with open(_SCHEMA_CACHE) as f:
            cache = json.load(f)
    missing = [d for d in set(db_ids) if d not in cache]
    if missing:
        log(f"[retrieve] introspecting {len(missing)} SynSQL schemas")
        for i, d in enumerate(missing):
            cache[d] = synsql_schema_text(d)
            if (i + 1) % 4000 == 0:
                log(f"  [schema] {i+1}/{len(missing)}")
        os.makedirs(CACHE_ROOT, exist_ok=True)
        with open(_SCHEMA_CACHE, "w") as f:
            json.dump(cache, f)
    return cache


def synsql_item_text(rec, smap=None):
    st = smap[rec["db_id"]] if smap else synsql_schema_text(rec["db_id"])
    return rec["question"] + " || " + st


def target_item_text(item):
    return item["question"] + " || " + schema_text_from_dict(item["schema"])


class SubsetRetriever:
    """Fit once on the SynSQL corpus, then score any number of unlabeled targets."""

    def __init__(self, recs, subsets, max_features=40000, log=print, vec=None,
                 X=None, smap=None):
        self.recs, self.subsets = recs, subsets
        if X is None:
            smap = smap or schema_text_map([r["db_id"] for r in recs], log=log)
            texts = [synsql_item_text(r, smap) for r in recs]
            self.vec = TfidfVectorizer(max_features=max_features, min_df=3,
                                       sublinear_tf=True, stop_words="english",
                                       token_pattern=r"[A-Za-z][A-Za-z0-9]+")
            X = normalize(self.vec.fit_transform(texts))
        else:
            self.vec = vec
        self.X = X
        log(f"[retrieve] fitted TF-IDF {self.X.shape} on {len(recs):,} SynSQL items")
        # kernel mean embedding per subset
        mus = np.vstack([np.asarray(self.X[idx].mean(axis=0)).ravel()
                         for idx in subsets])
        self.mu = normalize(mus)
        self.mu_raw = mus

    def embed_target(self, items):
        Xt = normalize(self.vec.transform([target_item_text(i) for i in items]))
        mu = np.asarray(Xt.mean(axis=0)).ravel()
        return normalize(mu.reshape(1, -1)).ravel(), mu

    def distances(self, items):
        """-> (cosine distance per subset, linear-MMD per subset)."""
        mu_n, mu_raw = self.embed_target(items)
        cos = 1.0 - self.mu @ mu_n
        mmd = np.linalg.norm(self.mu_raw - mu_raw[None, :], axis=1)
        return cos, mmd

    def topk(self, items, k, metric="cos"):
        cos, mmd = self.distances(items)
        d = cos if metric == "cos" else mmd
        return np.argsort(d)[:k], d


# --------------------------------------------------------------------------- #
#  STRUCTURAL retrieval: align on DIFFICULTY, not topic                        #
# --------------------------------------------------------------------------- #
# TF-IDF over question+schema tokens matches the *domain* of a set. But a pool's
# accuracy on a set is driven by how hard its queries are, not what they are about.
# These features are difficulty proxies computable WITHOUT labels -- they use only
# the question text and the schema, exactly what an unlabeled target provides.
_CUES = {
    "agg":    r"\b(average|mean|total|sum|count|number of|how many|maximum|minimum|"
              r"highest|lowest)\b",
    "group":  r"\b(each|per|every|for all|by category|group)\b",
    "sup":    r"\b(top|most|least|best|worst|rank|first|largest|smallest)\b",
    "cmp":    r"\b(more than|greater than|less than|at least|at most|between|above|"
              r"below|exceed)\b",
    "time":   r"\b(year|month|day|date|before|after|since|during|recent|latest)\b",
    "neg":    r"\b(not|without|exclude|never|no )\b",
    "join":   r"\b(and their|along with|together with|associated|related|belonging)\b",
}
_CUE_RE = {k: re.compile(v, re.I) for k, v in _CUES.items()}
_STATS_CACHE = os.path.join(CACHE_ROOT, "db_schema_stats.json")


def _db_stats(db_id):
    import sqlite3
    p = os.path.join(SYNSQL_DB_ROOT, db_id, db_id + ".sqlite")
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'")
        ts = [r[0] for r in cur.fetchall()]
        nc = 0
        for t in ts:
            try:
                cur.execute(f'PRAGMA table_info("{t}")')
                nc += len(cur.fetchall())
            except sqlite3.Error:
                pass
        con.close()
        return [len(ts), nc]
    except sqlite3.Error:
        return [0, 0]


def db_stats_map(db_ids, log=print):
    cache = {}
    if os.path.exists(_STATS_CACHE):
        with open(_STATS_CACHE) as f:
            cache = json.load(f)
    missing = [d for d in set(db_ids) if d not in cache]
    if missing:
        log(f"[struct] measuring {len(missing)} SynSQL schemas")
        for d in missing:
            cache[d] = _db_stats(d)
        os.makedirs(CACHE_ROOT, exist_ok=True)
        with open(_STATS_CACHE, "w") as f:
            json.dump(cache, f)
    return cache


def _feats(question, n_tables, n_cols):
    q = question.lower()
    f = [len(q.split()), n_tables, n_cols]
    f += [1.0 if _CUE_RE[k].search(q) else 0.0 for k in _CUES]
    return f


FEAT_NAMES = ["q_len", "n_tables", "n_cols"] + list(_CUES)


class StructuralRetriever:
    """Same interface as SubsetRetriever, but the representation is a normalized
    DIFFICULTY PROFILE instead of a topic bag-of-words."""

    def __init__(self, recs, subsets, log=print, stats=None):
        self.recs, self.subsets = recs, subsets
        stats = stats or db_stats_map([r["db_id"] for r in recs], log=log)
        F = np.array([_feats(r["question"], *stats.get(r["db_id"], [0, 0]))
                      for r in recs], dtype=float)
        self.mean_, self.std_ = F.mean(axis=0), F.std(axis=0) + 1e-9
        Z = (F - self.mean_) / self.std_
        self.mu = np.vstack([Z[idx].mean(axis=0) for idx in subsets])
        log(f"[struct] {Z.shape[1]} difficulty features over {len(recs):,} items")

    def _target_mu(self, items):
        F = np.array([_feats(i["question"],
                             len(i["schema"].get("schema_items", [])),
                             sum(len(t["column_names"])
                                 for t in i["schema"].get("schema_items", [])))
                      for i in items], dtype=float)
        return ((F - self.mean_) / self.std_).mean(axis=0)

    def distances(self, items):
        mu_t = self._target_mu(items)
        d = np.linalg.norm(self.mu - mu_t[None, :], axis=1)
        return d, d
