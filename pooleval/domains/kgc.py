"""Knowledge-graph completion: the LARGE-K port.

Task: given (head, relation, ?), name the tail entity. The answer is one of
`num_entities` classes -- 14,541 for FB15k-237, 40,943 for WN18RR -- so the label
space is finite but enormous. That is the point of adding this task.

WHY THIS TASK. Section 7 of `docs/multiclass_ds.md` measured how estimators
degrade as K grows, but only in simulation, because no real dataset in the repo
went past K=10. The simulation predicted full Dawid--Skene falls apart between
K=50 and K=100 once it drops below ~1 observation per free parameter. Knowledge-
graph completion tests that at K=14541, where full DS would need

    M * K^2 * 8 bytes  =  12 * 14541^2 * 8  ~=  20 GB

of confusion matrices. It is not slow, it is inapplicable -- `multiclass_ds.full_ds`
raises `MemoryError` with that arithmetic rather than pretending. One-coin DS
survives because its parameter count does not depend on K at all, but even it
needs the sparse E-step in `one_coin_ds_sparse`, since an [M, N, K] indicator is
also out of reach.

POOL DESIGN. Four genuinely different scoring functions -- TransE (translational),
DistMult (diagonal bilinear), ComplEx (complex bilinear), RotatE (rotation) --
each at several seeds. Same-family models are the near-clones.

ANCHORS.
  prior    : filtered Hits@1 on the labelled VALIDATION split.
  verifier : NOT a pool member and not an embedding model at all -- a frequency
             baseline that answers with the most common tail seen for that
             relation in training. Pure co-occurrence statistics, which the
             embedding models only see indirectly.
"""
import os
import numpy as np
import torch

CACHE = os.path.expanduser("~/.cache/pooleval_domains/kgc")
FAMILIES = ["TransE", "DistMult", "ComplEx", "RotatE"]
DATASETS = {"fb15k237": "FB15k_237", "wn18rr": "WordNet18RR"}


def load_kg(name):
    """-> dict of train/val/test triples [2,E] + types, plus counts.

    The two datasets expose splits differently: FB15k_237 takes a `split=`
    argument and returns one graph per split, WordNet18RR returns a single graph
    carrying train/val/test boolean masks.
    """
    import torch_geometric.datasets as D
    root = os.path.join(CACHE, name)
    cls = getattr(D, DATASETS[name])
    splits = {}
    try:
        for split in ("train", "val", "test"):
            d = cls(root, split=split)[0]
            splits[split] = (d.edge_index, d.edge_type)
    except TypeError:                                  # mask-style dataset
        d = cls(root)[0]
        for split, mask in (("train", d.train_mask), ("val", d.val_mask),
                            ("test", d.test_mask)):
            splits[split] = (d.edge_index[:, mask], d.edge_type[mask])
    n_ent = int(max(ei.max() for ei, _ in splits.values())) + 1
    n_rel = int(max(et.max() for _, et in splits.values())) + 1
    return dict(splits=splits, n_entities=n_ent, n_relations=n_rel)


def build_model(family, n_ent, n_rel, dim=128):
    import torch_geometric.nn.kge as kge
    return getattr(kge, family)(num_nodes=n_ent, num_relations=n_rel,
                                hidden_channels=dim)


def _train(model, edge_index, edge_type, device, epochs=60, batch=4096, lr=0.01,
           seed=0, verbose=False):
    torch.manual_seed(seed)
    model = model.to(device)
    loader = model.loader(head_index=edge_index[0].to(device),
                          rel_type=edge_type.to(device),
                          tail_index=edge_index[1].to(device),
                          batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        for h, r, t in loader:
            opt.zero_grad()
            loss = model.loss(h, r, t)
            loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def _predict_tails(model, h, r, n_ent, device, known=None, gold=None, chunk=32):
    """argmax_t score(h, r, t) over ALL entities, with filtered ranking.

    `known` maps (head, relation) -> set of true tails. The standard filtered
    protocol masks the OTHER true tails so a model is not punished for naming a
    different correct answer -- but never the gold tail of the query itself,
    which must stay reachable or accuracy is identically zero.
    """
    preds = np.empty(len(h), dtype=np.int64)
    all_t = torch.arange(n_ent, device=device)
    for s in range(0, len(h), chunk):
        hs = h[s:s + chunk].to(device)
        rs = r[s:s + chunk].to(device)
        b = len(hs)
        hh = hs.view(-1, 1).expand(b, n_ent).reshape(-1)
        rr = rs.view(-1, 1).expand(b, n_ent).reshape(-1)
        tt = all_t.view(1, -1).expand(b, n_ent).reshape(-1)
        sc = model(hh, rr, tt).view(b, n_ent)
        if known is not None:
            for k in range(b):
                other = known.get((int(hs[k]), int(rs[k])))
                if not other:
                    continue
                if gold is not None:
                    other = other - {int(gold[s + k])}      # never mask the answer
                if other:
                    sc[k, torch.tensor(sorted(other), device=device)] = -1e9
        preds[s:s + b] = sc.argmax(1).cpu().numpy()
    return preds


def _known_tails(splits):
    known = {}
    for ei, et in splits.values():
        for hh, rr, tt in zip(ei[0].tolist(), et.tolist(), ei[1].tolist()):
            known.setdefault((hh, rr), set()).add(tt)
    return known


def _frequency_verifier(train_ei, train_et, n_rel, n_ent, h, r):
    """NOT a pool member: answer with the most frequent tail for that relation."""
    best = np.zeros(n_rel, dtype=np.int64)
    for rel in range(n_rel):
        m = (train_et == rel).numpy()
        if m.any():
            best[rel] = np.bincount(train_ei[1].numpy()[m], minlength=n_ent).argmax()
    return best[r.numpy()]


def build_pool(name="fb15k237", seeds=(0, 1, 2), families=FAMILIES, dim=128,
               epochs=60, n_target=None, device="cuda", verbose=True):
    """Train {family x seed} on the train split; predict tails on val and test."""
    device = device if torch.cuda.is_available() else "cpu"
    kg = load_kg(name)
    n_ent, n_rel = kg["n_entities"], kg["n_relations"]
    (tr_ei, tr_et) = kg["splits"]["train"]
    (va_ei, va_et) = kg["splits"]["val"]
    (te_ei, te_et) = kg["splits"]["test"]

    if n_target is not None and n_target < te_ei.size(1):
        g = torch.Generator().manual_seed(7)
        idx = torch.randperm(te_ei.size(1), generator=g)[:n_target]
        te_ei, te_et = te_ei[:, idx], te_et[idx]
    if n_target is not None and n_target < va_ei.size(1):
        g = torch.Generator().manual_seed(8)
        idx = torch.randperm(va_ei.size(1), generator=g)[:n_target]
        va_ei, va_et = va_ei[:, idx], va_et[idx]

    known = _known_tails(kg["splits"])
    gold_t = te_ei[1].numpy().astype(np.int64)
    gold_v = va_ei[1].numpy().astype(np.int64)

    preds, preds_s, priors, group, names = [], [], [], [], []
    for gi, fam in enumerate(families):
        for sd in seeds:
            m = _train(build_model(fam, n_ent, n_rel, dim), tr_ei, tr_et, device,
                       epochs=epochs, seed=1000 * gi + sd)
            pv = _predict_tails(m, va_ei[0], va_et, n_ent, device, known, gold_v)
            pt = _predict_tails(m, te_ei[0], te_et, n_ent, device, known, gold_t)
            priors.append(float((pv == gold_v).mean()))
            preds.append(pt); preds_s.append(pv)
            group.append(gi); names.append(f"{fam}-s{sd}")
            if verbose:
                print(f"  {name} {fam}-s{sd}: val H@1 {priors[-1]:.3f}  "
                      f"test H@1 {float((pt == gold_t).mean()):.3f}", flush=True)
            del m; torch.cuda.empty_cache()

    vg = _frequency_verifier(tr_ei, tr_et, n_rel, n_ent, te_ei[0], te_et)
    if verbose:
        print(f"  verifier (most-frequent tail per relation): "
              f"test H@1 {float((vg == gold_t).mean()):.3f}")

    # prob_s / prob_t are one-hot stand-ins: the confidence baselines need a
    # distribution, and a full [N, 14541] softmax is neither affordable nor
    # meaningful here. DoC/ATC results on this task should be read with that
    # caveat -- they are reported for completeness, not as a fair comparison.
    return dict(pred=np.stack(preds), gold=gold_t, group=np.array(group),
                prior=np.array(priors), verifier_guess=vg, names=names,
                n_classes=n_ent, src_gold_val=gold_v,
                pred_s=np.stack(preds_s), dataset=name)
