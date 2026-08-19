"""Node-classification port: a pool of GNNs evaluated on an UNLABELLED target graph.

Setup follows GNNEvaluator (Zheng et al., NeurIPS 2023): three citation networks
that share one 6775-dim feature space and one 5-class label space --

    ACMv9 (A, 9360 nodes) | Citationv1 (C, 8935) | DBLPv7 (D, 5484)

train on the source graph, predict every node of the target graph, and estimate
accuracy there with NO target labels. Six transfers: A->C, A->D, C->A, C->D,
D->A, D->C.

POOL DESIGN. GNNEvaluator scores one model at a time; PoolEval needs a pool with
*provenance structure*, so we instantiate each architecture (GCN, GraphSAGE, GAT,
GIN, MLP) with several random seeds. Same-architecture models are the near-clones:
they share an inductive bias and therefore make correlated errors -- exactly the
`n_groups` structure the Text2SQL pool got from shared pretrained bases.

ANCHORS.
  prior    : accuracy on the held-out SOURCE split (labels are free there). This is
             the "seen prior" -- biased by the shift, which is the point.
  verifier : an independent channel that is NOT a pool member -- a feature-only
             logistic regression (no message passing) whose posterior is smoothed
             over the TARGET adjacency. It sees information the pool's *outputs* do
             not carry, though it is not fully independent of them (every GNN also
             consumes features and structure); treat it as a weak channel.
"""
import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.utils import from_scipy_sparse_matrix

CACHE = os.path.expanduser("~/.cache/pooleval_domains/graph")
URL = "https://raw.githubusercontent.com/daiquanyu/AdaGCN_TKDE/main/input/{}.mat"
NAMES = {"A": "acmv9", "C": "citationv1", "D": "dblpv7"}
ARCHS = ["GCN", "SAGE", "GAT", "GIN", "MLP"]


def load_graph(key, device="cuda"):
    """key in {A, C, D} -> (x, edge_index, y) on `device`."""
    name = NAMES[key]
    path = os.path.join(CACHE, f"{name}.mat")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(URL.format(name), path)
    d = sio.loadmat(path)
    x = torch.tensor(np.asarray(d["attrb"].todense() if sp.issparse(d["attrb"])
                                else d["attrb"]), dtype=torch.float)
    y = torch.tensor(np.asarray(d["group"]).argmax(1), dtype=torch.long)
    A = sp.csr_matrix(d["network"])
    edge_index = from_scipy_sparse_matrix(A)[0]
    return x.to(device), edge_index.to(device), y.to(device), A


class Net(torch.nn.Module):
    """One backbone per architecture, identical width/depth so the pool differs by
    inductive bias and seed, not capacity."""

    def __init__(self, arch, d_in, d_out, hid=128, dropout=0.4):
        super().__init__()
        self.arch, self.dropout = arch, dropout
        if arch == "GCN":
            self.c1, self.c2 = GCNConv(d_in, hid), GCNConv(hid, d_out)
        elif arch == "SAGE":
            self.c1, self.c2 = SAGEConv(d_in, hid), SAGEConv(hid, d_out)
        elif arch == "GAT":
            self.c1 = GATConv(d_in, hid // 4, heads=4)
            self.c2 = GATConv(hid, d_out, heads=1)
        elif arch == "GIN":
            mlp = lambda i, o: torch.nn.Sequential(
                torch.nn.Linear(i, o), torch.nn.ReLU(), torch.nn.Linear(o, o))
            self.c1, self.c2 = GINConv(mlp(d_in, hid)), GINConv(mlp(hid, d_out))
        elif arch == "MLP":
            self.c1, self.c2 = torch.nn.Linear(d_in, hid), torch.nn.Linear(hid, d_out)
        else:
            raise ValueError(arch)

    def forward(self, x, ei):
        h = self.c1(x) if self.arch == "MLP" else self.c1(x, ei)
        h = F.dropout(F.relu(h), self.dropout, self.training)
        return self.c2(h) if self.arch == "MLP" else self.c2(h, ei)


def _train_one(arch, seed, xs, eis, ys, train_mask, epochs=200, lr=0.01, wd=5e-4):
    torch.manual_seed(seed)
    net = Net(arch, xs.size(1), int(ys.max()) + 1).to(xs.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = F.cross_entropy(net(xs, eis)[train_mask], ys[train_mask])
        loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def _probs(net, x, ei):
    return F.softmax(net(x, ei), dim=1).cpu().numpy()


def _predict(net, x, ei):
    return _probs(net, x, ei).argmax(1)


def _verifier(xs, ys, train_mask, xt, At, n_classes, hops=2, alpha=0.7):
    """Feature-only logistic regression (NOT a pool member) smoothed over the
    target adjacency -- the graph analogue of Text2SQL's execution verifier."""
    from sklearn.linear_model import LogisticRegression
    Xs = xs[train_mask].cpu().numpy(); Ys = ys[train_mask].cpu().numpy()
    lr = LogisticRegression(max_iter=300, n_jobs=-1).fit(Xs, Ys)
    P = lr.predict_proba(xt.cpu().numpy())
    if P.shape[1] < n_classes:                       # class missing from source split
        full = np.zeros((P.shape[0], n_classes)); full[:, lr.classes_] = P; P = full
    Anorm = sp.csr_matrix(At, dtype=float)
    deg = np.asarray(Anorm.sum(1)).ravel(); deg[deg == 0] = 1
    Anorm = sp.diags(1.0 / deg) @ Anorm
    S = P.copy()
    for _ in range(hops):                            # label propagation
        S = alpha * (Anorm @ S) + (1 - alpha) * P
    return S.argmax(1)


def build_pool(src, dst, seeds=(0, 1, 2), archs=ARCHS, device="cuda",
               train_frac=0.8, epochs=200, verbose=True):
    """Train {arch x seed} on `src`, predict all nodes of `dst`.

    Returns a dict with pred [M,N], gold [N], group [M] (= architecture id),
    prior [M] (source held-out accuracy), verifier_guess [N], and model names.
    """
    device = device if torch.cuda.is_available() else "cpu"
    xs, eis, ys, As = load_graph(src, device)
    xt, eit, yt, At = load_graph(dst, device)
    K = int(max(ys.max(), yt.max())) + 1

    g = torch.Generator().manual_seed(12345)
    perm = torch.randperm(xs.size(0), generator=g)
    ntr = int(train_frac * xs.size(0))
    tr = torch.zeros(xs.size(0), dtype=torch.bool); tr[perm[:ntr]] = True
    va = ~tr
    tr, va = tr.to(device), va.to(device)

    preds, priors, group, names = [], [], [], []
    prob_t, prob_s = [], []          # confidences for the ATC / DoC baselines
    for gi, arch in enumerate(archs):
        for s in seeds:
            net = _train_one(arch, s, xs, eis, ys, tr, epochs=epochs)
            vam = va.cpu().numpy()
            ps = _probs(net, xs, eis)
            src_acc = float((ps.argmax(1)[vam] == ys.cpu().numpy()[vam]).mean())
            pt = _probs(net, xt, eit)
            prob_s.append(ps[vam]); prob_t.append(pt)
            preds.append(pt.argmax(1))
            priors.append(src_acc); group.append(gi); names.append(f"{arch}-s{s}")
            if verbose:
                tgt = float((preds[-1] == yt.cpu().numpy()).mean())
                print(f"  {src}->{dst} {arch}-s{s}: src {src_acc:.3f}  tgt {tgt:.3f}")
    vg = _verifier(xs, ys, tr.cpu().numpy(), xt, At, K)
    return dict(pred=np.stack(preds), gold=yt.cpu().numpy(),
                group=np.array(group), prior=np.array(priors),
                verifier_guess=vg, names=names, n_classes=K,
                prob_t=np.stack(prob_t), prob_s=np.stack(prob_s),
                src_gold_val=ys.cpu().numpy()[va.cpu().numpy()],
                src=src, dst=dst)
