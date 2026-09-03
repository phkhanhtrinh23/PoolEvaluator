"""Link-prediction port: a pool of GNN encoders scored on an UNLABELLED target graph.

Same three citation networks as `graph.py` (ACMv9 / Citationv1 / DBLPv7), but the
task is now "does an edge exist between these two nodes?" rather than "what class
is this node?". Train the encoder on the source graph's edges, then score node
pairs of the target graph with no target supervision.

WHY THIS TASK IS WORTH ADDING. It is the K=2 boundary case of the whole
collision argument. `Trinh_proof.tex` observes that the pre-collision assumption
`P(C=1 | Z=0) = 1-beta` -- two wrong answers always coincide -- is *forced* when
there are only two possible answers, and false otherwise. Node and image
classification (K=5, K=10) sit on the "false" side; link prediction sits exactly
on the "forced" side. So here gamma = 1 is not an approximation, it is correct,
and the collision-aware machinery should provably buy nothing. That is a
falsifiable prediction, which is why the task earns its place.

It also stresses the pool in a way classification does not: link prediction is
class-imbalanced by construction (most pairs are non-edges), so we score a
balanced sample of positives and sampled negatives, and the "accuracy" being
estimated is balanced accuracy over that sample.

POOL DESIGN. Same five encoders as `graph.py` (GCN, SAGE, GAT, GIN, MLP) x seeds,
each with a dot-product decoder. Same-architecture models are the near-clones.

ANCHORS.
  prior    : link-prediction accuracy on the held-out SOURCE edge split.
  verifier : a structural heuristic that is NOT a pool member and uses no learned
             embedding -- Adamic-Adar on the target graph, thresholded. It sees
             pure topology, which the encoders only see through their features.
"""
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .graph import ARCHS, Net, load_graph

CACHE = os.path.expanduser("~/.cache/pooleval_domains/graph")


def _split_edges(A, n_nodes, val_frac=0.1, seed=0):
    """Undirected edge split -> (train_pos, eval_pos, eval_neg) as [2,E] arrays."""
    rng = np.random.default_rng(seed)
    Au = sp.triu(sp.csr_matrix(A), k=1).tocoo()
    e = np.vstack([Au.row, Au.col])
    perm = rng.permutation(e.shape[1])
    e = e[:, perm]
    n_eval = max(1, int(val_frac * e.shape[1]))
    eval_pos, train_pos = e[:, :n_eval], e[:, n_eval:]

    # negatives: sample non-adjacent pairs, same count as eval positives
    Acsr = sp.csr_matrix(A)
    neg = []
    while len(neg) < n_eval:
        u = rng.integers(0, n_nodes, n_eval)
        v = rng.integers(0, n_nodes, n_eval)
        ok = (u != v) & (np.asarray(Acsr[u, v]).ravel() == 0)
        neg.extend(zip(u[ok].tolist(), v[ok].tolist()))
    neg = np.array(neg[:n_eval]).T
    return train_pos, eval_pos, neg


def _message_graph(train_pos, device):
    """Edges visible to message passing: training positives, symmetrised."""
    ei = np.hstack([train_pos, train_pos[::-1]])
    return torch.tensor(ei, dtype=torch.long, device=device)


def _score(z, pairs):
    return (z[pairs[0]] * z[pairs[1]]).sum(-1)


def _train_encoder(arch, seed, x, ei, train_pos, n_nodes, epochs=200, lr=0.01,
                   wd=5e-4, dim=128):
    torch.manual_seed(seed)
    net = Net(arch, x.size(1), dim).to(x.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    tp = torch.tensor(train_pos, dtype=torch.long, device=x.device)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        z = net(x, ei)
        # fresh negatives each step -- standard link-prediction training
        nn_ = torch.randint(0, n_nodes, tp.shape, device=x.device)
        loss = (F.binary_cross_entropy_with_logits(_score(z, tp), torch.ones(tp.size(1), device=x.device))
                + F.binary_cross_entropy_with_logits(_score(z, nn_), torch.zeros(nn_.size(1), device=x.device)))
        loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def _predict(net, x, ei, pairs, thresh=0.0):
    z = net(x, ei)
    p = torch.tensor(pairs, dtype=torch.long, device=x.device)
    logit = _score(z, p)
    return (logit > thresh).long().cpu().numpy(), torch.sigmoid(logit).cpu().numpy()


def _adamic_adar(A, pairs):
    """Topology-only verifier: NOT a pool member, uses no learned embedding."""
    Acsr = sp.csr_matrix(A).astype(bool).astype(float)
    deg = np.asarray(Acsr.sum(1)).ravel()
    w = np.zeros_like(deg)
    nz = deg > 1
    w[nz] = 1.0 / np.log(deg[nz])
    scores = np.zeros(pairs.shape[1])
    for k in range(pairs.shape[1]):
        u, v = pairs[0, k], pairs[1, k]
        common = Acsr[u].multiply(Acsr[v]).indices
        scores[k] = w[common].sum()
    return scores


def build_pool(src, dst, seeds=(0, 1, 2), archs=ARCHS, device="cuda",
               epochs=200, val_frac=0.1, seed=0, verbose=True):
    """Train {arch x seed} link predictors on `src`, score `dst` node pairs."""
    device = device if torch.cuda.is_available() else "cpu"
    xs, _, _, As = load_graph(src, device)
    xt, _, _, At = load_graph(dst, device)
    ns, nt = xs.size(0), xt.size(0)

    s_train, s_pos, s_neg = _split_edges(As, ns, val_frac, seed)
    t_train, t_pos, t_neg = _split_edges(At, nt, val_frac, seed)
    ei_s, ei_t = _message_graph(s_train, device), _message_graph(t_train, device)

    # evaluation pairs and the withheld gold: 1 = edge, 0 = non-edge  (K = 2)
    s_pairs = np.hstack([s_pos, s_neg])
    s_gold = np.concatenate([np.ones(s_pos.shape[1]), np.zeros(s_neg.shape[1])]).astype(np.int64)
    t_pairs = np.hstack([t_pos, t_neg])
    t_gold = np.concatenate([np.ones(t_pos.shape[1]), np.zeros(t_neg.shape[1])]).astype(np.int64)

    preds, priors, group, names, prob_t, prob_s = [], [], [], [], [], []
    for gi, arch in enumerate(archs):
        for sd in seeds:
            net = _train_encoder(arch, 1000 * gi + sd, xs, ei_s, s_train, ns, epochs=epochs)
            ps, cs = _predict(net, xs, ei_s, s_pairs)
            pt, ct = _predict(net, xt, ei_t, t_pairs)
            priors.append(float((ps == s_gold).mean()))
            preds.append(pt)
            prob_s.append(np.stack([1 - cs, cs], 1))
            prob_t.append(np.stack([1 - ct, ct], 1))
            group.append(gi); names.append(f"{arch}-s{sd}")
            if verbose:
                print(f"  {src}->{dst} {arch}-s{sd}: src {priors[-1]:.3f}  "
                      f"tgt {float((pt == t_gold).mean()):.3f}", flush=True)
            del net; torch.cuda.empty_cache()

    aa = _adamic_adar(At, t_pairs)
    vg = (aa > np.median(aa[aa > 0]) if (aa > 0).any() else aa > 0).astype(np.int64)
    if verbose:
        print(f"  verifier (Adamic-Adar, topology only): tgt {float((vg == t_gold).mean()):.3f}")

    return dict(pred=np.stack(preds), gold=t_gold, group=np.array(group),
                prior=np.array(priors), verifier_guess=vg, names=names,
                n_classes=2, prob_t=np.stack(prob_t), prob_s=np.stack(prob_s),
                src_gold_val=s_gold, src=src, dst=dst)
