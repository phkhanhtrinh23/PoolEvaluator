"""Run the zoo over a list of items -> {member_name: {item_id: pred_sql}}.

Cached at the API layer (zoo/clients.py), so this is fully resumable: re-running
only pays for members/items not already generated.
"""
import json
import os
import sys

from .clients import chat, extract_sql
from .prompt import build_messages
from .config import ARTIFACT_ROOT


def generate_predictions(members, items, cfg, tag="dev", log=print):
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    preds = {}
    total = len(members) * len(items)
    done = 0
    for mem in members:
        preds[mem.name] = {}
        for it in items:
            msgs = build_messages(it, style=mem.prompt)
            try:
                raw = chat(mem.provider, mem.model, msgs,
                           temperature=mem.temperature, max_tokens=cfg.max_tokens)
                sql = extract_sql(raw)
            except Exception as e:  # noqa
                log(f"  [gen] {mem.name} {it['id']} FAILED: {repr(e)[:80]}")
                sql = ""
            preds[mem.name][it["id"]] = sql
            done += 1
            if done % 100 == 0 or done == total:
                log(f"  [gen:{tag}] {done}/{total}")
                sys.stdout.flush()
    out = os.path.join(ARTIFACT_ROOT, f"preds_{tag}.json")
    with open(out, "w") as f:
        json.dump(preds, f)
    log(f"[saved] {out}")
    return preds
