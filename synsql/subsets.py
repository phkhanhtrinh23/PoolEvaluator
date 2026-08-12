"""Partition SynSQL-2.5M into subsets of ~1K samples.

Three partitionings, because *how* you cut the corpus decides whether retrieval can
work at all:

  random    i.i.d. shuffle -> chunks of 1000.  This is the literal reading of "split
            randomly", and it is a DEGENERATE control: every subset is an unbiased
            sample of the same corpus, so all subsets have (up to O(1/sqrt(1000))
            noise) the SAME distribution and the SAME model accuracy. Retrieval then
            has nothing to select on.
  db        pack whole databases (db_id = a domain, e.g. "forestry") into subsets of
            ~1000. Subsets become domain-coherent, so they genuinely differ -- this
            is the partitioning retrieval needs.
  kmeans    MiniBatchKMeans on TF-IDF of the questions, balanced to ~1000 each. The
            maximally-separated partitioning; upper bound on retrievability.

    python -m synsql.subsets --sample 400000
"""
import argparse
import json
import os
import random

import numpy as np

from .config import INDEX_JSONL, SUBSETS_DIR, SUBSET_SIZE, SEED


def load_index(path=INDEX_JSONL, sample=None, seed=SEED):
    """Stream the 2.7 GB ingest index and return a uniform random `sample` of it via
    single-pass reservoir sampling (O(sample) memory, exact uniformity)."""
    rng = random.Random(seed)
    if not sample:
        with open(path) as f:
            return [json.loads(l) for l in f]
    res = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < sample:
                res.append(line)
            else:
                j = rng.randrange(i + 1)
                if j < sample:
                    res[j] = line
    return [json.loads(r) for r in res]


def _chunks(order, size):
    return [order[i:i + size] for i in range(0, len(order) - size + 1, size)]


def partition_random(recs, size=SUBSET_SIZE, seed=SEED):
    order = list(range(len(recs)))
    random.Random(seed).shuffle(order)
    return _chunks(order, size)


def partition_by_db(recs, size=SUBSET_SIZE, seed=SEED):
    """Greedily pack whole db_ids into ~size-item subsets (domain-coherent)."""
    by_db = {}
    for i, r in enumerate(recs):
        by_db.setdefault(r["db_id"], []).append(i)
    dbs = sorted(by_db, key=lambda d: -len(by_db[d]))
    random.Random(seed).shuffle(dbs)
    subs, cur = [], []
    for d in dbs:
        cur.extend(by_db[d])
        while len(cur) >= size:
            subs.append(cur[:size]); cur = cur[size:]
    return subs


def partition_kmeans(recs, size=SUBSET_SIZE, seed=SEED, log=print):
    """Cluster questions, then cut each cluster into size-item subsets."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import MiniBatchKMeans
    k = max(2, len(recs) // (size * 4))            # ~4 subsets per cluster
    texts = [r["question"] for r in recs]
    X = TfidfVectorizer(max_features=30000, min_df=3, sublinear_tf=True,
                        stop_words="english").fit_transform(texts)
    log(f"[kmeans] {X.shape} -> {k} clusters")
    lab = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                          batch_size=4096).fit_predict(X)
    subs = []
    for c in range(k):
        idx = np.flatnonzero(lab == c).tolist()
        random.Random(seed + c).shuffle(idx)
        subs.extend(_chunks(idx, size))
    return subs


PARTITIONERS = {"random": partition_random, "db": partition_by_db,
                "kmeans": partition_kmeans}


def build(sample=400_000, size=SUBSET_SIZE, seed=SEED, log=print):
    os.makedirs(SUBSETS_DIR, exist_ok=True)
    recs = load_index(sample=sample, seed=seed)
    log(f"[subsets] loaded {len(recs):,} records "
        f"over {len(set(r['db_id'] for r in recs)):,} databases")
    out = {}
    for name, fn in PARTITIONERS.items():
        subs = fn(recs, size=size, seed=seed) if name != "kmeans" else \
            fn(recs, size=size, seed=seed, log=log)
        log(f"[subsets] {name}: {len(subs)} subsets of {size}")
        out[name] = subs
    with open(os.path.join(SUBSETS_DIR, "records.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    with open(os.path.join(SUBSETS_DIR, "partitions.json"), "w") as f:
        json.dump({k: v for k, v in out.items()}, f)
    log(f"[saved] {SUBSETS_DIR}/records.jsonl + partitions.json")
    return recs, out


def load_built():
    with open(os.path.join(SUBSETS_DIR, "records.jsonl")) as f:
        recs = [json.loads(l) for l in f]
    with open(os.path.join(SUBSETS_DIR, "partitions.json")) as f:
        parts = json.load(f)
    return recs, parts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--size", type=int, default=SUBSET_SIZE)
    a = ap.parse_args()
    build(sample=a.sample, size=a.size)
