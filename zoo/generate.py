"""Run the zoo over a list of items -> {member_name: {item_id: pred_sql}}.

Cached at the API layer (zoo/clients.py), so this is fully resumable: re-running
only pays for members/items not already generated.
"""
import json
import os
import sys
import threading

from .clients import chat, extract_sql
from .prompt import build_messages
from .config import ARTIFACT_ROOT


def _one(mem, it, cfg, log):
    msgs = build_messages(it, style=mem.prompt)
    try:
        raw = chat(mem.provider, mem.model, msgs, temperature=mem.temperature,
                   max_tokens=cfg.max_tokens)
        return extract_sql(raw)
    except Exception as e:  # noqa
        log(f"  [gen] {mem.name} {it['id']} FAILED: {repr(e)[:80]}")
        return ""


def generate_predictions(members, items, cfg, tag="dev", log=print, workers=1):
    """workers>1 issues the (member, item) calls concurrently. The on-disk cache is
    written atomically, so threading changes throughput only, never results."""
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    preds = {m.name: {} for m in members}
    jobs = [(mem, it) for mem in members for it in items]
    total, done = len(jobs), 0
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_one, mem, it, cfg, log): (mem, it)
                    for mem, it in jobs}
            for fut, (mem, it) in futs.items():
                preds[mem.name][it["id"]] = fut.result()
                with lock:
                    done += 1
                    if done % 200 == 0 or done == total:
                        log(f"  [gen:{tag}] {done}/{total}")
                        sys.stdout.flush()
    else:
        for mem, it in jobs:
            preds[mem.name][it["id"]] = _one(mem, it, cfg, log)
            done += 1
            if done % 100 == 0 or done == total:
                log(f"  [gen:{tag}] {done}/{total}")
                sys.stdout.flush()
    out = os.path.join(ARTIFACT_ROOT, f"preds_{tag}.json")
    with open(out, "w") as f:
        json.dump(preds, f)
    log(f"[saved] {out}")
    return preds
