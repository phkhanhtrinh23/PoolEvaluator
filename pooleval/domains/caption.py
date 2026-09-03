"""Image captioning: the UNBOUNDED-answer-space port.

Every other domain in `pooleval/domains/` has a closed label set -- 2 classes for
link prediction, 5 for citation graphs, 10 for digits, 14541 for knowledge-graph
completion. Captioning has none: the answer is a sentence, and the space of
sentences is not enumerable. That makes it the first non-SQL task in this repo
that sits in Text2SQL's own regime, which is exactly why it is worth running.

WHAT THIS BREAKS, AND WHAT SURVIVES.

  full Dawid--Skene   IMPOSSIBLE, and not for a memory reason. A confusion matrix
                      pi_j[c,k] needs class identity to persist ACROSS items, but
                      caption equivalence classes are per-image clusters --
                      "cluster 2" on image 5 has nothing to do with "cluster 2" on
                      image 6. There is no matrix to fill in.
  one-coin DS         Runs, but its (1-a)/(K-1) term needs a K. We use the per-item
                      candidate count, which is the honest reading: with an
                      unbounded space, two wrong captions colliding is rare, so the
                      implied collision rate should be small. Reported with that
                      caveat.
  PoolEval / latent   Runs unchanged. It only ever tests observations for EQUALITY,
                      so it never needs to enumerate the label space.
  binary + gamma      Its home regime -- the binary reduction is forced here, and
                      gamma is measurable rather than assumed.

THE EQUIVALENCE KERNEL. Text2SQL decides whether two answers agree by executing
both queries and comparing result tables, with graded strictness (LA0/LA1/LA2).
Captioning needs the same thing for sentences. We normalise, drop stopwords, and
declare two captions equivalent when their content-word Jaccard clears a
threshold; per image, the model captions and the reference captions are then
clustered by connected components of that relation. A model is correct on an
image when its caption lands in the same component as a reference.

This kernel is deliberately simple and deliberately reported at several
thresholds, because the threshold IS the strictness dial and the results should
be read as a function of it, not at one arbitrary setting.
"""
import os
import re
import numpy as np

CACHE = os.path.expanduser("~/.cache/pooleval_domains/caption")
FAMILIES = ["ViT-GPT2", "BLIP-base", "GIT-base", "BLIP-large", "GIT-large"]
HF_IDS = {
    "ViT-GPT2":   "nlpconnect/vit-gpt2-image-captioning",
    "BLIP-base":  "Salesforce/blip-image-captioning-base",
    "GIT-base":   "microsoft/git-base-coco",
    "BLIP-large": "Salesforce/blip-image-captioning-large",
    "GIT-large":  "microsoft/git-large-coco",
}
STOP = set("""a an the of in on at to for with and or is are was were be been being this that
these those there here its it his her their our your my as by from into over under near
some any all both each few more most other such no nor not only own same so than too very
s t can will just don should now up down out off above below then once he she they them him
""".split())


def normalize(text):
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split()


def content_words(text):
    """Normalised content-word set: the unit the kernel compares."""
    out = set()
    for w in normalize(text):
        if w in STOP or len(w) < 2:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                      # crude singularisation
        out.add(w)
    return out


def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def cluster_item(cands, threshold):
    """Connected components of `jaccard >= threshold` -> one cluster id per caption."""
    sets = [content_words(c) for c in cands]
    n = len(sets)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(sets[i], sets[j]) >= threshold:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [remap[r] for r in roots]


def encode_item(model_caps, ref_caps, threshold):
    """-> (pred_ids [M], gold_id) for one image.

    Model captions and reference captions are clustered TOGETHER, so a model is
    correct exactly when it shares a component with some reference. The gold id is
    the component holding the most references.
    """
    M = len(model_caps)
    ids = cluster_item(list(model_caps) + list(ref_caps), threshold)
    pred, ref_ids = ids[:M], ids[M:]
    gold = max(set(ref_ids), key=ref_ids.count)
    return np.array(pred, dtype=np.int64), int(gold)


def load_images(n_images, split="validation", seed=0):
    """-> (list of PIL images, list of reference-caption lists). Grouped by cocoid."""
    from datasets import load_dataset
    ds = load_dataset("jxie/coco_captions", split=split, streaming=True)
    by_id, order = {}, []
    for row in ds:
        cid = row["cocoid"]
        if cid not in by_id:
            if len(by_id) >= n_images:
                break
            by_id[cid] = dict(image=row["image"], refs=[])
            order.append(cid)
        by_id[cid]["refs"].append(row["caption"])
    return ([by_id[c]["image"].convert("RGB") for c in order],
            [by_id[c]["refs"] for c in order])


def _load_model(family, device):
    """-> (encode, decode, model).

    The five checkpoints do not share a processor convention: `AutoProcessor` on
    nlpconnect/vit-gpt2-image-captioning resolves to a *tokenizer*, which then
    rejects `images=`. So probe the processor on a dummy image and fall back to a
    separate image processor + tokenizer when it cannot handle pixels.
    """
    from PIL import Image
    from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer
    hf = HF_IDS[family]
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(hf)
    except Exception:
        from transformers import VisionEncoderDecoderModel
        model = VisionEncoderDecoderModel.from_pretrained(hf)
    model = model.to(device).eval()

    dummy = [Image.new("RGB", (32, 32))]
    try:
        proc = AutoProcessor.from_pretrained(hf)
        proc(images=dummy, return_tensors="pt")           # probe
        return (lambda im: proc(images=im, return_tensors="pt"),
                lambda ids: proc.batch_decode(ids, skip_special_tokens=True),
                model)
    except Exception:
        ip = AutoImageProcessor.from_pretrained(hf)
        tk = AutoTokenizer.from_pretrained(hf)
        return (lambda im: ip(images=im, return_tensors="pt"),
                lambda ids: tk.batch_decode(ids, skip_special_tokens=True),
                model)



def generate_captions(family, images, device="cuda", batch=16, max_new_tokens=24,
                      num_beams=3, seed=0, verbose=True):
    """Caption every image with one model family. `seed` varies decoding, not weights."""
    import torch
    encode, decode, model = _load_model(family, device)
    caps = []
    torch.manual_seed(seed)
    with torch.no_grad():
        for s in range(0, len(images), batch):
            chunk = images[s:s + batch]
            inputs = encode(chunk).to(device)
            kw = dict(max_new_tokens=max_new_tokens)
            if num_beams > 1 and seed == 0:
                kw.update(num_beams=num_beams)          # deterministic decode
            else:
                kw.update(do_sample=True, top_p=0.9, temperature=0.7 + 0.15 * seed)
            out = model.generate(**inputs, **kw)
            caps.extend(decode(out))
    del model
    import torch as _t; _t.cuda.empty_cache()
    if verbose:
        print(f"    {family} seed {seed}: e.g. {caps[0][:70]!r}", flush=True)
    return [c.strip() for c in caps]


def encode_pool(model_caps, verifier_caps, ref_caps, threshold):
    """Cluster every caption for an image together, then read off ids.

    model_caps    [M][N] strings
    verifier_caps [N]    strings from a model held OUT of the pool
    ref_caps      [N][R]  human references, withheld from the estimator
    -> pred [M,N], gold [N], verifier [N]   (all integer cluster ids, per item)
    """
    M, N = len(model_caps), len(ref_caps)
    pred = np.zeros((M, N), dtype=np.int64)
    gold = np.zeros(N, dtype=np.int64)
    vg = np.zeros(N, dtype=np.int64)
    for i in range(N):
        cands = [model_caps[m][i] for m in range(M)]
        cands.append(verifier_caps[i])
        cands.extend(ref_caps[i])
        ids = cluster_item(cands, threshold)
        pred[:, i] = ids[:M]
        vg[i] = ids[M]
        r = ids[M + 1:]
        gold[i] = max(set(r), key=r.count)
    return pred, gold, vg


def build_pool(n_source=300, n_target=700, families=None, seeds=(0, 1, 2),
               threshold=0.3, verifier_family="GIT-large", device="cuda",
               batch=16, verbose=True):
    """Caption a COCO split with a pool of captioners; score against held-out refs.

    Seeds vary DECODING, not weights: these are pretrained checkpoints, so
    same-family members share parameters exactly and differ only in beam vs
    sampled decoding. That is a stronger clone relationship than seeded training
    anywhere else in this repo, which makes it a clean test of the group discount.

    The source split keeps its references (that is what the prior is measured on);
    the target split has its references withheld and used only for scoring.
    """
    families = families or [f for f in FAMILIES if f != verifier_family]
    n = n_source + n_target
    images, refs = load_images(n)
    n = len(images)
    n_source = min(n_source, n - 1)
    if verbose:
        print(f"  {n} images ({n_source} source / {n - n_source} target), "
              f"pool = {len(families)} families x {len(seeds)} seeds, "
              f"kernel threshold {threshold}", flush=True)

    caps, group, names = [], [], []
    for gi, fam in enumerate(families):
        for sd in seeds:
            caps.append(generate_captions(fam, images, device=device, batch=batch,
                                          seed=sd, verbose=verbose))
            group.append(gi); names.append(f"{fam}-s{sd}")
    ver = generate_captions(verifier_family, images, device=device, batch=batch,
                            seed=0, verbose=verbose)

    pred, gold, vg = encode_pool(caps, ver, refs, threshold)
    sl, tl = slice(0, n_source), slice(n_source, n)
    prior = (pred[:, sl] == gold[None, sl]).mean(1)
    if verbose:
        acc = (pred[:, tl] == gold[None, tl]).mean(1)
        for k, nm in enumerate(names):
            print(f"    {nm:16s} source {prior[k]:.3f}  target {acc[k]:.3f}", flush=True)
        print(f"    verifier ({verifier_family}, held out): "
              f"target {float((vg[tl] == gold[tl]).mean()):.3f}", flush=True)

    # n_classes is the WIDEST per-item candidate count, not a real label-space size:
    # the space of sentences is unbounded and cluster ids are local to an image.
    K = int(max(pred[:, tl].max(), gold[tl].max(), vg[tl].max())) + 1
    return dict(pred=pred[:, tl], gold=gold[tl], group=np.array(group),
                prior=prior, verifier_guess=vg[tl], names=names, n_classes=K,
                pred_s=pred[:, sl], src_gold_val=gold[sl],
                threshold=threshold, captions=np.array(caps, dtype=object),
                verifier_captions=np.array(ver, dtype=object),
                n_source=n_source, unbounded=True)
