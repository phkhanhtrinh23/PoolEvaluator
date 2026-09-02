"""Image-classification port: a pool of CNN/ViT classifiers on an UNLABELLED
target set under distribution shift.

Setup follows MetaEvaluator (Pham et al., KDD 2026): train on a source digit
corpus and evaluate on a shifted target with the SAME 10-class label space --

    MNIST -> USPS   (mild shift: different pen strokes / scanner)
    MNIST -> SVHN   (strong shift: street-view photographs)

Architectures are MetaEvaluator's five families -- ResNeXt-50-32x4d,
RegNetY-8GF, ConvNeXt-Tiny, ViT-Tiny and DeiT-Small -- rebuilt at 32x32 (the ViT
family is instantiated directly from torchvision's VisionTransformer with a
patch size of 4, since torchvision ships no pretrained DeiT). Each family is
trained with several seeds, so a family IS a provenance group: same inductive
bias, correlated errors.

ANCHORS.
  prior    : held-out SOURCE accuracy -- the "seen prior", biased by the shift.
  verifier : a small CNN held OUT of the pool, averaged over test-time
             augmentations. Independent of every pool member's parameters, though
             it shares the source training distribution.
"""
import os
import ssl
import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms as T

DATA = os.path.expanduser("~/.cache/pooleval_domains/vision")
FAMILIES = ["ResNeXt50", "RegNetY8GF", "ConvNeXtT", "ViT-Tiny", "DeiT-Small"]
_MEAN, _STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def _tf(train=False):
    aug = [T.RandomAffine(8, translate=(0.08, 0.08), scale=(0.92, 1.08))] if train else []
    return T.Compose([T.Resize((32, 32)), T.Grayscale(3), *aug,
                      T.ToTensor(), T.Normalize(_MEAN, _STD)])


def load_dataset(name, train, tf=None):
    """name in {mnist, usps, svhn}. All returned as 32x32 RGB, 10 classes."""
    os.makedirs(DATA, exist_ok=True)
    tf = tf or _tf(train=False)
    _ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context   # USPS host
    try:
        if name == "mnist":
            return tv.datasets.MNIST(DATA, train=train, download=True, transform=tf)
        if name == "usps":
            return tv.datasets.USPS(DATA, train=train, download=True, transform=tf)
        if name == "svhn":
            return tv.datasets.SVHN(DATA, split="train" if train else "test",
                                    download=True, transform=tf)
        raise ValueError(name)
    finally:
        ssl._create_default_https_context = _ctx


def build_model(family, n_classes=10):
    """MetaEvaluator's five families, re-headed for 32x32 / 10 classes."""
    if family == "ResNeXt50":
        m = tv.models.resnext50_32x4d(num_classes=n_classes)
        m.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = torch.nn.Identity()
        return m
    if family == "RegNetY8GF":
        m = tv.models.regnet_y_8gf(num_classes=n_classes)
        m.stem[0] = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        return m
    if family == "ConvNeXtT":
        return tv.models.convnext_tiny(num_classes=n_classes)
    if family == "ViT-Tiny":
        return tv.models.VisionTransformer(image_size=32, patch_size=4, num_layers=6,
                                           num_heads=3, hidden_dim=192, mlp_dim=768,
                                           num_classes=n_classes)
    if family == "DeiT-Small":
        return tv.models.VisionTransformer(image_size=32, patch_size=4, num_layers=8,
                                           num_heads=6, hidden_dim=384, mlp_dim=1536,
                                           num_classes=n_classes)
    raise ValueError(family)


class _SmallCNN(torch.nn.Module):
    """Verifier backbone -- deliberately NOT one of the pool families."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(128, n_classes))

    def forward(self, x):
        return self.f(x)


def _train(model, loader, device, epochs, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    return model.eval()


@torch.no_grad()
def _probs(model, loader, device, tta=0):
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            p = F.softmax(model(x).float(), 1)
            for _ in range(tta):                       # test-time augmentation
                xa = torch.roll(x, shifts=(np.random.randint(-2, 3),
                                           np.random.randint(-2, 3)), dims=(2, 3))
                p = p + F.softmax(model(xa).float(), 1)
        out.append((p / (1 + tta)).cpu().numpy())
    return np.concatenate(out)


def build_pool(src="mnist", dst="usps", seeds=(0, 1, 2), families=FAMILIES,
               epochs=3, batch=256, n_target=None, device="cuda", verbose=True):
    """Train {family x seed} on `src`, predict the `dst` test split."""
    device = device if torch.cuda.is_available() else "cpu"
    tr_ds = load_dataset(src, True, _tf(train=True))
    va_ds = load_dataset(src, False)
    te_ds = load_dataset(dst, False)
    if n_target is not None and n_target < len(te_ds):
        g = torch.Generator().manual_seed(7)
        te_ds = torch.utils.data.Subset(
            te_ds, torch.randperm(len(te_ds), generator=g)[:n_target].tolist())

    mk = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=sh, num_workers=4, pin_memory=True, drop_last=False)
    tr, va, te = mk(tr_ds, True), mk(va_ds, False), mk(te_ds, False)

    gold = np.array([int(te_ds[i][1]) for i in range(len(te_ds))])
    gold_va = np.array([int(va_ds[i][1]) for i in range(len(va_ds))])

    preds, priors, group, names, prob_t, prob_s = [], [], [], [], [], []
    for gi, fam in enumerate(families):
        for s in seeds:
            m = _train(build_model(fam), tr, device, epochs, seed=1000 * gi + s)
            ps, pt = _probs(m, va, device), _probs(m, te, device)
            src_acc = float((ps.argmax(1) == gold_va).mean())
            preds.append(pt.argmax(1)); prob_s.append(ps); prob_t.append(pt)
            priors.append(src_acc); group.append(gi); names.append(f"{fam}-s{s}")
            if verbose:
                print(f"  {src}->{dst} {fam}-s{s}: src {src_acc:.3f}  "
                      f"tgt {float((preds[-1] == gold).mean()):.3f}", flush=True)
            del m; torch.cuda.empty_cache()

    ver = _train(_SmallCNN(), tr, device, epochs, seed=99)
    vg = _probs(ver, te, device, tta=4).argmax(1)
    if verbose:
        print(f"  verifier (held-out CNN + TTA): tgt {float((vg == gold).mean()):.3f}")

    return dict(pred=np.stack(preds), gold=gold, group=np.array(group),
                prior=np.array(priors), verifier_guess=vg, names=names, n_classes=10,
                prob_t=np.stack(prob_t), prob_s=np.stack(prob_s), src_gold_val=gold_va,
                src=src, dst=dst)


def build_pools_from_source(src, dsts, seeds=(0, 1, 2), families=FAMILIES,
                            epochs=3, batch=256, n_target=None, device="cuda",
                            verbose=True):
    """Train {family x seed} ONCE on `src`, then evaluate on every target in `dsts`.

    The per-pair `build_pool` retrains for each (src, dst), which is 3x wasteful
    when running a transfer matrix -- and worse, it means the pool evaluated on
    two different targets is not literally the same pool. This trains one set of
    models per source and reuses it, which is both cheaper and the correct
    controlled design: the only thing that changes across targets is the target.
    """
    device = device if torch.cuda.is_available() else "cpu"
    tr_ds, va_ds = load_dataset(src, True, _tf(train=True)), load_dataset(src, False)
    mk = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=sh, num_workers=4, pin_memory=True, drop_last=False)
    tr, va = mk(tr_ds, True), mk(va_ds, False)
    gold_va = np.array([int(va_ds[i][1]) for i in range(len(va_ds))])

    targets = {}
    for d in dsts:
        te_ds = load_dataset(d, False)
        if n_target is not None and n_target < len(te_ds):
            g = torch.Generator().manual_seed(7)
            te_ds = torch.utils.data.Subset(
                te_ds, torch.randperm(len(te_ds), generator=g)[:n_target].tolist())
        targets[d] = (mk(te_ds, False),
                      np.array([int(te_ds[i][1]) for i in range(len(te_ds))]))

    acc = {d: dict(pred=[], prob_t=[]) for d in dsts}
    priors, group, names, prob_s = [], [], [], []
    for gi, fam in enumerate(families):
        for sd in seeds:
            m = _train(build_model(fam), tr, device, epochs, seed=1000 * gi + sd)
            ps = _probs(m, va, device)
            priors.append(float((ps.argmax(1) == gold_va).mean()))
            prob_s.append(ps); group.append(gi); names.append(f"{fam}-s{sd}")
            for d, (te, gold) in targets.items():
                pt = _probs(m, te, device)
                acc[d]["pred"].append(pt.argmax(1)); acc[d]["prob_t"].append(pt)
                if verbose:
                    print(f"  {src}->{d} {fam}-s{sd}: src {priors[-1]:.3f}  "
                          f"tgt {float((pt.argmax(1) == gold).mean()):.3f}", flush=True)
            del m; torch.cuda.empty_cache()

    ver = _train(_SmallCNN(), tr, device, epochs, seed=99)
    out = {}
    for d, (te, gold) in targets.items():
        vg = _probs(ver, te, device, tta=4).argmax(1)
        if verbose:
            print(f"  {src}->{d} verifier: tgt {float((vg == gold).mean()):.3f}", flush=True)
        out[d] = dict(pred=np.stack(acc[d]["pred"]), gold=gold, group=np.array(group),
                      prior=np.array(priors), verifier_guess=vg, names=names,
                      n_classes=10, prob_t=np.stack(acc[d]["prob_t"]),
                      prob_s=np.stack(prob_s), src_gold_val=gold_va, src=src, dst=d)
    del ver; torch.cuda.empty_cache()
    return out
