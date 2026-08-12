"""Probe SynSQL subsets with the model pool -> a reusable per-subset accuracy table.

Key economy: a subset is probed ONCE and reused by every target. Probing S subsets
with n items costs S*n*M generations; after that, any number of unlabeled targets get
a retrieval-aligned prior for FREE (just pick which probes to pool). That is what
makes the method cheap enough to deploy.
"""
import json
import os

import numpy as np

from zoo.config import ZooConfig
from zoo.datasets import introspect_schema
from zoo.generate import generate_predictions
from zoo.execute import result_key
from .config import SYNSQL_DB_ROOT, N_PROBE, ARTIFACTS


def probe_items(recs, subset_idx, n_probe=N_PROBE, seed=0, tag="s0"):
    """Turn a SynSQL subset into canonical zoo items (schema introspected from the
    live SQLite DB, external knowledge appended to the question as BIRD-style
    evidence). Spread across the subset's db_ids so one database can't dominate."""
    import random
    rng = random.Random(seed)
    by_db = {}
    for i in subset_idx:
        by_db.setdefault(recs[i]["db_id"], []).append(i)
    dbs = sorted(by_db)
    for d in dbs:
        rng.shuffle(by_db[d])
    rng.shuffle(dbs)
    picked, di = [], 0
    while len(picked) < min(n_probe, len(subset_idx)):
        d = dbs[di % len(dbs)]; di += 1
        if by_db[d]:
            picked.append(by_db[d].pop())
        if all(not by_db[d] for d in dbs):
            break
    items = []
    for i in picked:
        r = recs[i]
        p = os.path.join(SYNSQL_DB_ROOT, r["db_id"], r["db_id"] + ".sqlite")
        q = r["question"]
        if r.get("external_knowledge"):
            q += "\n[Evidence] " + r["external_knowledge"]
        try:
            sch = introspect_schema(p)
        except Exception:                                    # noqa: BLE001
            continue
        items.append(dict(id=f"{tag}-{i}", db_id=r["db_id"], question=q,
                          gold_sql=r["sql"], db_path=p, schema=sch))
    return items


def probe_correctness(items, member_names, preds, timeout=5.0):
    """[M, n] 0/1 correctness of each member on each probe item."""
    C = np.zeros((len(member_names), len(items)), dtype=np.int8)
    for j, it in enumerate(items):
        gk, _ = result_key(it["db_path"], it["gold_sql"], timeout)
        if gk is None:                                       # ungradable gold
            C[:, j] = -1
            continue
        for m, name in enumerate(member_names):
            k, _ = result_key(it["db_path"], preds[name].get(it["id"], ""), timeout)
            C[m, j] = int(k is not None and k == gk)
    return C


def probe_subsets(recs, subsets, cand_ids, zcfg: ZooConfig, n_probe=N_PROBE,
                  seed=0, log=print, workers=12):
    """Generate + execute the pool on each candidate subset. Cached on disk, so this
    is resumable and never re-pays for a subset already probed.

    Returns dict: subset_id -> {"C": [M,n] correctness (-1 = ungradable item),
                                "items": [item ids]}
    """
    names = [m.name for m in zcfg.manifest]
    cache_p = os.path.join(ARTIFACTS, "synsql_probes.npz")
    store = {}
    if os.path.exists(cache_p):
        z = np.load(cache_p, allow_pickle=True)
        store = {int(k): v for k, v in z["store"].item().items()}
        log(f"[probe] cache hit for {sorted(store)}")
    for s in cand_ids:
        if s in store:
            continue
        items = probe_items(recs, subsets[s], n_probe=n_probe, seed=seed, tag=f"syn{s}")
        log(f"[probe] subset {s}: {len(items)} items over "
            f"{len(set(i['db_id'] for i in items))} dbs")
        preds = generate_predictions(zcfg.manifest, items, zcfg, tag=f"synsql{s}",
                                     log=log, workers=workers)
        C = probe_correctness(items, names, preds, timeout=zcfg.exec_timeout)
        store[s] = dict(C=C, ids=[i["id"] for i in items])
        ok = (C[0] >= 0)
        log(f"[probe] subset {s}: gradable {ok.sum()}/{C.shape[1]}  "
            f"pool EX={np.mean([C[m][ok].mean() for m in range(C.shape[0])]):.3f}")
        np.savez(cache_p, store=store)
    return {s: store[s] for s in cand_ids}


def prior_from_probes(store, subset_ids):
    """Pool the probe items of the chosen subsets -> per-model EX + binomial sigma."""
    cols = [store[s]["C"] for s in subset_ids]
    C = np.concatenate(cols, axis=1)
    ok = C[0] >= 0
    C = C[:, ok].astype(float)
    n = C.shape[1]
    acc = C.mean(axis=1)
    sigma = np.sqrt(np.clip(acc * (1 - acc), 1e-4, None) / max(1, n))
    return acc, np.maximum(sigma, 1e-3), n


def prior_soft(store, subset_ids, dists, tau=0.02):
    """Distance-weighted prior: softmax(-d/tau) over ALL probed subsets instead of a
    hard top-k. Uses every probe (lower variance) but down-weights misaligned ones."""
    w = np.exp(-(np.asarray(dists) - np.min(dists)) / tau)
    w = w / w.sum()
    accs, ns = [], []
    for s in subset_ids:
        C = store[s]["C"]
        ok = C[0] >= 0
        accs.append(C[:, ok].astype(float).mean(axis=1))
        ns.append(int(ok.sum()))
    A = np.vstack(accs)                                    # [S, M]
    acc = (w[:, None] * A).sum(axis=0)
    n_eff = 1.0 / np.sum(w ** 2) * np.mean(ns)             # Kish effective sample size
    sigma = np.sqrt(np.clip(acc * (1 - acc), 1e-4, None) / max(1.0, n_eff))
    return acc, np.maximum(sigma, 1e-3), n_eff
