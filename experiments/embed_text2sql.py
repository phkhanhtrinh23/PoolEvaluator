"""Cache OpenAI embeddings for the Spider/BIRD source and target questions.

Split out from the retrieval experiment so the paid call happens once and the analysis
can be re-run freely.  Each item is embedded as "[db_id] question": the database is part
of what makes a Text-to-SQL item hard, so dropping it would compare questions as though
they were schema-free.

  python experiments/embed_text2sql.py
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoo.datasets import load_split                                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "zoo_artifacts", "question_embeddings.npz")


def embed(texts, model, batch=128):
    from openai import OpenAI
    client, out = OpenAI(), []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        r = client.embeddings.create(model=model, input=chunk)
        out.extend(d.embedding for d in sorted(r.data, key=lambda d: d.index))
        print(f"    embedded {i + len(chunk)}/{len(texts)}", flush=True)
    return np.asarray(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--datasets", nargs="+", default=["spider", "bird"])
    ap.add_argument("--n-target", type=int, default=150)
    ap.add_argument("--n-source", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    payload = {}
    for ds in args.datasets:
        for split, n in [("dev", args.n_target), ("source", args.n_source)]:
            items = load_split(ds, split, n, seed=args.seed)
            texts = [f"[{x['db_id']}] {x['question']}" for x in items]
            print(f"  [{ds}/{split}] {len(texts)} items", flush=True)
            payload[f"{ds}_{split}"] = embed(texts, args.model)
            payload[f"{ds}_{split}_ids"] = np.array([x["id"] for x in items])
    np.savez_compressed(args.out, **payload)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
