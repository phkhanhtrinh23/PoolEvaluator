"""Does a Top-K cosine-similar slice of the labeled split predict target accuracy?

The deployment idea being tested: instead of labeling data from the target pool, retrieve
the K labeled items most similar to the unlabeled target data and use the models' accuracy
on that slice as the prior.  If retrieval works, the retrieved slice should look like the
target and the accuracies should match.

Each item gets a label-free embedding -- the only kind available at deployment:

  text2sql   OpenAI text-embedding-3-small over "[db_id] question"
  vision     the M models' class-probability vectors, concatenated (M x C dims)
  graph      same

Source items are scored by cosine similarity to the target, two ways: against the target
CENTROID (equivalently, mean similarity to every target item), and by mean similarity to
the item's ten nearest target neighbours, which rewards being close to some part of the
target rather than to its average.

Controls matter more than the headline here.  ``bottom-K`` takes the FARTHEST items: if
retrieval carries signal, nearest must beat farthest.  ``random-K`` is the same slice size
with no retrieval at all, and ``full`` uses the whole labeled split.  A Top-K that cannot
beat random-K of the same size is retrieving nothing.

Error is reported three ways, never as a single MAE:

  bias          mean signed error, the shared level offset -- one number for the whole pool
  MAE           mean absolute error, which CONTAINS that offset
  centered MAE  MAE after removing the offset: whether the SHAPE (relative model ability)
                is right even when the level is not

That split is the point.  A prior can rank every model correctly and still be uniformly
wrong about the level, and only the decomposition shows it -- a single MAE hides it.

  python experiments/run_retrieval_prior.py
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_CACHE = os.path.expanduser("~/.cache/pooleval_domains/pools")


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def similarity(src_emb, tgt_emb, how, knn=10, block=2048):
    """Cosine similarity of every source item to the target set."""
    S, T = unit(src_emb), unit(tgt_emb)
    if how == "centroid":
        return S @ unit(T.mean(axis=0))
    out = np.empty(len(S))
    k = min(knn, len(T))
    for i in range(0, len(S), block):                 # blocked: S x T can be huge
        sims = S[i:i + block] @ T.T
        part = np.partition(sims, -k, axis=1)[:, -k:]
        out[i:i + block] = part.mean(axis=1)
    return out


def scores(acc_hat, true_acc):
    """-> (bias, MAE, centered MAE, Spearman). See the module docstring."""
    d = np.asarray(acc_hat) - np.asarray(true_acc)
    bias = float(d.mean())
    rs = lambda v: np.argsort(np.argsort(v))
    a, b = rs(acc_hat), rs(true_acc)
    rho = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
    return bias, float(np.abs(d).mean()), float(np.abs(d - bias).mean()), rho


def evaluate(correct_src, true_acc, sim, K, rng, draws=20):
    """Per-model accuracy on each slice of size K. ``correct_src`` is [M, N_src] 0/1."""
    order = np.argsort(-sim)
    out = {"top-K": correct_src[:, order[:K]].mean(axis=1),
           "bottom-K": correct_src[:, order[-K:]].mean(axis=1),
           "full": correct_src.mean(axis=1)}
    rnd = [correct_src[:, rng.choice(correct_src.shape[1], K, replace=False)].mean(axis=1)
           for _ in range(draws)]
    out["random-K"] = np.mean(rnd, axis=0)
    out["_random_draws"] = rnd
    return {k: (v, true_acc) for k, v in out.items() if not k.startswith("_")}, rnd


def load_cases(args):
    cases = {}
    from zoo.new_formulation_real import load_run
    emb_path = os.path.join(ROOT, "zoo_artifacts", "question_embeddings.npz")
    if os.path.exists(emb_path):
        E = np.load(emb_path, allow_pickle=True)
        src_tc = np.load(os.path.join(ROOT, "zoo_artifacts", "source_true_class.npz"))
        for ds in ["spider", "bird"]:
            run, _ = load_run(ds)
            cases[f"text2sql/{ds}"] = dict(
                correct_src=(src_tc[ds] == 0).astype(float),
                true_acc=run.true_acc,
                src_emb=E[f"{ds}_source"], tgt_emb=E[f"{ds}_dev"])
    else:
        print("  [skip] text2sql -- run experiments/embed_text2sql.py first")

    for key in ["vision_mnist_usps", "vision_mnist_svhn", "graph_AC", "graph_DA"]:
        path = os.path.join(POOL_CACHE, key + ".npz")
        if not os.path.exists(path):
            continue
        z = np.load(path, allow_pickle=True)
        pred, gold = z["pred"], z["gold"]
        prob_s, prob_t = z["prob_s"], z["prob_t"]
        pred_s = prob_s.argmax(-1)
        # Label-free item embedding: every model's class-probability vector, concatenated.
        # M x C dims -- what a deployment can see without touching a single label.
        se = prob_s.transpose(1, 0, 2).reshape(prob_s.shape[1], -1)
        te = prob_t.transpose(1, 0, 2).reshape(prob_t.shape[1], -1)
        dom = "vision/" if key.startswith("vision") else "graph/"
        cases[dom + key.split("_", 1)[1]] = dict(
            correct_src=(pred_s == z["src_gold_val"][None, :]).astype(float),
            true_acc=(pred == gold[None, :]).mean(axis=1),
            src_emb=se, tgt_emb=te)
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    ap.add_argument("--how", nargs="+", default=["centroid", "knn"])
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "retrieval_prior.json"))
    args = ap.parse_args()

    payload = {}
    for name, c in load_cases(args).items():
        M, Ns = c["correct_src"].shape
        print(f"\n  [{name}]  M={M}  source={Ns}  target={len(c['true_acc']) and ''}"
              f"{c['tgt_emb'].shape[0]}  emb_dim={c['src_emb'].shape[1]}", flush=True)
        payload[name] = {}
        for how in args.how:
            sim = similarity(c["src_emb"], c["tgt_emb"], how, knn=args.knn)
            print(f"    similarity={how}  range [{sim.min():.3f}, {sim.max():.3f}]")
            print(f"      {'slice':12s}{'K':>6s}{'bias':>9s}{'MAE':>9s}"
                  f"{'cMAE':>9s}{'rho':>8s}   (accuracy points)")
            for fr in args.fracs:
                K = max(2, int(round(fr * Ns)))
                rng = np.random.default_rng(args.seed)
                res, rnd = evaluate(c["correct_src"], c["true_acc"], sim, K, rng,
                                    draws=args.draws)
                for slice_name in ["top-K", "random-K", "bottom-K"]:
                    ah, ta = res[slice_name]
                    b, m, cm, rho = scores(ah, ta)
                    payload[name].setdefault(how, {}).setdefault(
                        f"K={K}", {})[slice_name] = dict(bias=b, mae=m, cmae=cm, rho=rho)
                    print(f"      {slice_name:12s}{K:>6d}{100*b:>9.2f}{100*m:>9.2f}"
                          f"{100*cm:>9.2f}{rho:>8.3f}")
                # Spread of the random control: top-K only "beats random" if it clears this.
                sd = np.std([np.abs(np.asarray(r) - c["true_acc"]).mean() for r in rnd])
                print(f"      {'  (rand sd)':12s}{'':>6s}{'':>9s}{100*sd:>9.2f}")
            ah, ta = evaluate(c["correct_src"], c["true_acc"], sim, 2,
                              np.random.default_rng(0))[0]["full"]
            b, m, cm, rho = scores(ah, ta)
            payload[name].setdefault(how, {})["full"] = dict(bias=b, mae=m, cmae=cm, rho=rho)
            print(f"      {'full split':12s}{Ns:>6d}{100*b:>9.2f}{100*m:>9.2f}"
                  f"{100*cm:>9.2f}{rho:>8.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
