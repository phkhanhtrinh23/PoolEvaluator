"""Build ``node_text.json`` and ``class_names.json`` for graphs that ship without them.

GOFA and the graph language models read node text, and GOFA prompts name the
candidate labels.  ogbn-arxiv provides both; this module recovers them for the other
node targets from public raw releases, aligned to the node order the loaders use:

* GOOD-Cora (PyG ``CitationFull('Cora')`` = graph2gauss ``cora.npz``): each node's
  McCallum Cora paper id resolves, through the original ``cora-classify`` release,
  to the extracted title and abstract.  Class names are the 70 Cora topic paths.
* GOOD-WebKB (PyG ``WebKB`` = Geom-GCN files, concatenated Wisconsin, Cornell, Texas):
  each node is matched to its LINQS WebKB row by its 1703-word binary vector, which
  gives the page URL and class; the page text comes from the CMU WebKB archive.
  Geom-GCN numbered the classes separately for each university, so a label index
  names a different class in different universities.  ``class_names.json`` therefore
  stores one list per university plus each node's university.
* GOOD-Twitch (PyG ``Twitch`` DE, EN, ES, FR, PT, RU): Twitch has no text, so each
  node gets a description built from SNAP account metadata (language, account age,
  views, partner status, number of games).  The label ("mature") is never included.
* ACMv9, Citationv1, DBLPv7: only title bag-of-words vectors over an unpublished
  vocabulary were released, so no node text can be recovered.  The five research
  areas are documented, but not which label index is which; class names are written
  only when the order is supplied explicitly.

Every build also writes ``node_meta.json`` with the node count and a SHA-1 of the
label sequence.  The loaders refuse the text when the graph they load does not match.

    python -m pooleval.node_metadata --dataset all --node-root datasets/node
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CORA_NPZ_URL = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz"
CORA_RAW_URL = "https://people.cs.umass.edu/~mccallum/data/cora-classify.tar.gz"
LINQS_WEBKB_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/WebKB.tgz"
CMU_WEBKB_URL = "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz"
GEOM_GCN_URL = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/{}/out1_node_feature_label.txt"
SNAP_TWITCH_URL = "https://snap.stanford.edu/data/twitch.zip"

# Concatenation orders used by GOOD's process() methods.
WEBKB_UNIVERSITIES = ("wisconsin", "cornell", "texas")
TWITCH_LANGUAGES = (
    ("DE", "DE", "German"), ("EN", "ENGB", "English"), ("ES", "ES", "Spanish"),
    ("FR", "FR", "French"), ("PT", "PTBR", "Portuguese"), ("RU", "RU", "Russian"),
)
CITATION_AREAS = (
    "Database", "Artificial Intelligence", "Computer Vision", "Information Security", "Networking",
)
MAX_CHARS = 2000


def labels_sha1(labels: Iterable[int]) -> str:
    return hashlib.sha1(np.asarray(list(labels), dtype=np.int64).tobytes()).hexdigest()


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        from urllib.request import urlretrieve

        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {url}")
        urlretrieve(url, path)
    return path


def _write(
    directory: Path,
    labels: Sequence[int],
    source: str,
    notes: Sequence[str],
    text: Sequence[str] | None = None,
    class_names: Any = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if text is not None:
        if len(text) != len(labels):
            raise ValueError("node text and labels differ in length")
        (directory / "node_text.json").write_text(json.dumps(list(text)), encoding="utf-8")
    if class_names is not None:
        (directory / "class_names.json").write_text(json.dumps(class_names, indent=1), encoding="utf-8")
    meta = {
        "num_nodes": len(labels),
        "labels_sha1": labels_sha1(labels),
        "source": source,
        "notes": list(notes),
    }
    (directory / "node_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[written] {directory} ({len(labels)} nodes)")
    for note in notes:
        print(f"  note: {note}")
    return directory


def _clip(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 3].rstrip() + "..."


# --------------------------------------------------------------------- GOOD-Cora


def _cora_fields(document: str) -> dict[str, str]:
    """Title and Abstract from a McCallum extraction file ("Field: value" lines)."""
    fields: dict[str, list[str]] = {}
    current = None
    for line in document.splitlines():
        header = re.match(r"^([A-Z][A-Za-z-]+):\s?(.*)$", line)
        if header:
            current = header.group(1)
            fields.setdefault(current, []).append(header.group(2))
        elif current in {"Title", "Abstract"}:
            fields[current].append(line)
    return {key: " ".join(fields[key]).strip() for key in ("Title", "Abstract") if key in fields}


def build_good_cora(node_root: Path, raw_dir: Path) -> Path:
    npz = np.load(_download(CORA_NPZ_URL, raw_dir / "cora.npz"), allow_pickle=True)
    labels = npz["labels"].astype(int).tolist()
    idx_to_node = npz["idx_to_node"].item()
    paper_ids = [str(idx_to_node[i]).strip() for i in range(len(labels))]
    idx_to_class = npz["idx_to_class"].item()
    class_names = [
        " / ".join(part.replace("_", " ") for part in idx_to_class[i].split("/"))
        for i in range(len(idx_to_class))
    ]
    attr_names = npz["idx_to_attr"].item()

    archive = _download(CORA_RAW_URL, raw_dir / "cora-classify.tar.gz")
    papers: dict[str, list[tuple[str, str]]] = defaultdict(list)
    wanted: dict[str, str] = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.name == "cora/papers":
                for line in tar.extractfile(member).read().decode("latin-1").splitlines():
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        papers[parts[0]].append((parts[1], parts[2]))
                break
    needed = {paper for paper in paper_ids}
    for paper in needed:
        for filename, _ in papers.get(paper, []):
            wanted.setdefault("cora/extractions/" + filename, paper)
    extracted: dict[str, dict[str, str]] = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            paper = wanted.get(member.name)
            if paper is None or not member.isfile():
                continue
            fields = _cora_fields(tar.extractfile(member).read().decode("latin-1"))
            best = extracted.get(paper, {})
            if len(fields.get("Abstract", "")) > len(best.get("Abstract", "")) or not best:
                extracted[paper] = fields

    from scipy.sparse import csr_matrix

    attributes = csr_matrix(
        (npz["attr_data"], npz["attr_indices"], npz["attr_indptr"]), shape=tuple(npz["attr_shape"])
    )
    text: list[str] = []
    counts = Counter()
    for index, paper in enumerate(paper_ids):
        fields = extracted.get(paper, {})
        title = fields.get("Title", "")
        if not title:
            for _, citation in papers.get(paper, []):
                match = re.search(r"<title>(.*?)</title>", citation)
                if match:
                    title = match.group(1).strip(" \"',.")
                    break
        abstract = fields.get("Abstract", "")
        if title and abstract:
            counts["title+abstract"] += 1
            text.append(_clip(f"Title: {title}. Abstract: {abstract}"))
        elif title:
            counts["title"] += 1
            text.append(_clip(f"Title: {title}."))
        else:
            counts["keywords"] += 1
            row = attributes.getrow(index)
            top = row.indices[np.argsort(-row.data)][:30]
            text.append(_clip("Keywords: " + ", ".join(attr_names[int(i)] for i in top) + "."))
    notes = [
        "text from the original McCallum Cora extractions matched by paper id; "
        + ", ".join(f"{count} with {kind}" for kind, count in counts.most_common())
    ]
    return _write(
        node_root / "good" / "GOODCora", labels, "graph2gauss cora.npz + McCallum cora-classify",
        notes, text, class_names,
    )


# -------------------------------------------------------------------- GOOD-WebKB


class _PageText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title: list[str] = []
        self.body: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        (self.title if self._in_title else self.body).append(data)


def _page_text(document: str) -> tuple[str, str]:
    # CMU pages start with an HTTP response header block.
    parts = re.split(r"\r?\n\r?\n", document, maxsplit=1)
    body = parts[1] if len(parts) == 2 and parts[0].lstrip().upper().startswith(("MIME", "HTTP", "DATE", "SERVER")) else document
    parser = _PageText()
    try:
        parser.feed(body)
    except Exception:  # malformed 1997 HTML
        pass
    title = html.unescape(" ".join(parser.title))
    content = html.unescape(" ".join(parser.body))
    return re.sub(r"\s+", " ", title).strip(), re.sub(r"\s+", " ", content).strip()


def _url_key(url: str) -> str:
    return url.rstrip("/").casefold()


def build_good_webkb(node_root: Path, raw_dir: Path) -> Path:
    linqs = _download(LINQS_WEBKB_URL, raw_dir / "WebKB.tgz")
    cmu = _download(CMU_WEBKB_URL, raw_dir / "webkb-data.gtar.gz")
    pages: dict[str, dict[str, str]] = defaultdict(dict)  # url -> class -> archive member
    with tarfile.open(cmu) as tar:
        members = {}
        for member in tar:
            parts = member.name.split("/")
            if member.isfile() and len(parts) == 4:
                url = parts[3].replace("^", "/")
                pages[_url_key(url)][parts[1]] = member.name
                members[member.name] = member
        documents = {}
        wanted = {name for classes in pages.values() for name in classes.values()}
        for name in wanted:
            documents[name] = tar.extractfile(members[name]).read().decode("latin-1")

    labels: list[int] = []
    text: list[str] = []
    domain_of_node: list[str] = []
    by_domain: dict[str, list[str]] = {}
    notes: list[str] = []
    with tarfile.open(linqs) as tar:
        content = {
            Path(m.name).stem: tar.extractfile(m).read().decode("latin-1")
            for m in tar if m.isfile() and m.name.endswith(".content")
        }
    for university in WEBKB_UNIVERSITIES:
        rows = [line.split("\t") for line in content[university].splitlines() if line.strip()]
        index: dict[tuple[int, ...], list[tuple[str, str]]] = defaultdict(list)
        for row in rows:
            index[tuple(int(v) for v in row[1:-1])].append((row[0], row[-1]))
        geom_path = _download(GEOM_GCN_URL.format(university), raw_dir / f"geom_gcn_{university}.txt")
        geom = [line.split("\t") for line in geom_path.read_text().splitlines()[1:] if line.strip()]
        matches = [index.get(tuple(int(v) for v in row[1].split(",")), []) for row in geom]
        geom_labels = [int(row[2]) for row in geom]
        if any(not hits for hits in matches):
            raise ValueError(f"{university}: some Geom-GCN nodes have no LINQS feature match")
        votes: dict[int, Counter] = defaultdict(Counter)
        for label, hits in zip(geom_labels, matches):
            if len(hits) == 1:
                votes[label][hits[0][1]] += 1
        mapping = {label: counter.most_common(1)[0][0] for label, counter in votes.items()}
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(f"{university}: Geom-GCN labels do not map one-to-one onto classes")
        by_domain[university] = [mapping[label] for label in range(len(mapping))]
        used: set[str] = set()
        ambiguous = 0
        for label, hits in zip(geom_labels, matches):
            candidates = [h for h in hits if h[1] == mapping[label] and h[0] not in used] or [
                h for h in hits if h[0] not in used
            ] or hits
            ambiguous += len(hits) > 1
            url, page_class = candidates[0]
            used.add(url)
            stored = pages.get(_url_key(url), {})
            member = stored.get(page_class) or next(iter(stored.values()), None)
            title, body = _page_text(documents[member]) if member else ("", "")
            text.append(_clip(f"Web page {url}. Title: {title}. Content: {body}"))
            labels.append(label)
            domain_of_node.append(university)
        if ambiguous:
            notes.append(f"{university}: {ambiguous} nodes share their word vector with another page")
    if len({tuple(names) for names in by_domain.values()}) > 1:
        notes.append(
            "Geom-GCN numbers the WebKB classes separately per university, so a label index "
            "names a different class in different universities: "
            + "; ".join(f"{u}={names}" for u, names in by_domain.items())
        )
    return _write(
        node_root / "good" / "GOODWebKB", labels, "Geom-GCN WebKB + LINQS WebKB + CMU WebKB pages",
        notes, text, {"by_domain": by_domain, "domain_of_node": domain_of_node},
    )


# ------------------------------------------------------------------- GOOD-Twitch


def build_good_twitch(node_root: Path, raw_dir: Path) -> Path:
    archive = _download(SNAP_TWITCH_URL, raw_dir / "twitch.zip")
    labels: list[int] = []
    text: list[str] = []
    notes: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for _, folder, language in TWITCH_LANGUAGES:
            target = next(n for n in names if n.endswith(f"musae_{folder}_target.csv"))
            features = next(n for n in names if re.search(rf"/{folder}/musae_{folder}(_features)?\.json$", n))
            rows = list(csv.DictReader(io.TextIOWrapper(bundle.open(target), encoding="utf-8")))
            games = json.loads(bundle.read(features))
            by_id: dict[int, dict[str, str]] = {}
            for row in rows:  # SNAP FR repeats two accounts; keep the first row of each
                by_id.setdefault(int(row["new_id"]), row)
            if sorted(by_id) != list(range(len(by_id))):
                raise ValueError(f"{folder}: new_id does not cover 0..{len(by_id) - 1}")
            if len(by_id) != len(rows):
                notes.append(f"{folder}: {len(rows) - len(by_id)} duplicated account rows dropped")
            for node in range(len(by_id)):
                row = by_id[node]
                played = len(set(games.get(str(node), [])))
                partner = row["partner"] == "True"
                text.append(
                    f"A Twitch streamer broadcasting in {language}. The account is {int(row['days']):,} days old, "
                    f"its channel has {int(row['views']):,} views, and it is {'' if partner else 'not '}a Twitch partner. "
                    f"The streamer has played {played} distinct games."
                )
                labels.append(int(row["mature"] == "True"))
    notes += [
        "Twitch has no node text; descriptions are built from SNAP account metadata and exclude the label",
        "PyG node i is assumed to be MUSAE new_id i within each language; the label fingerprint checks this",
    ]
    return _write(
        node_root / "good" / "GOODTwitch", labels, "SNAP twitch (MUSAE)", notes, text,
        ["does not stream mature content", "streams mature content"],
    )


# ------------------------------------------------------------ citation networks


def build_citation(name: str, node_root: Path, label_order: Sequence[str] | None) -> Path | None:
    print(
        f"[{name}] no node text: the release has only title bag-of-words over an unpublished "
        "vocabulary. Add node_text.json yourself if you have the paper titles."
    )
    if not label_order:
        print(
            f"[{name}] class names not written: the documented areas {list(CITATION_AREAS)} are not "
            "published with their label indices; pass --citation-label-order to assert an order."
        )
        return None
    from .graph_data import _load_citation_graph

    graph, _, _, _ = _load_citation_graph(name, node_root)
    if len(label_order) != int(graph.y.max()) + 1:
        raise ValueError(f"{name} has {int(graph.y.max()) + 1} classes, got {len(label_order)} names")
    return _write(
        node_root / "gnnevaluator" / name, graph.y.tolist(), "user-specified label order",
        ["class order supplied with --citation-label-order, not verified against the data"],
        class_names=list(label_order),
    )


BUILDERS = {"good-cora": build_good_cora, "good-webkb": build_good_webkb, "good-twitch": build_good_twitch}
CITATION = ("acmv9", "citationv1", "dblpv7")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=[*BUILDERS, *CITATION, "all"], default="all")
    parser.add_argument("--node-root", default="datasets/node")
    parser.add_argument("--raw-dir", help="download cache (default: <node-root>/raw_text)")
    parser.add_argument(
        "--citation-label-order",
        help="comma-separated class names for label indices 0..4 of ACMv9/Citationv1/DBLPv7",
    )
    args = parser.parse_args(argv)
    node_root = Path(args.node_root)
    raw_dir = Path(args.raw_dir) if args.raw_dir else node_root / "raw_text"
    order = [part.strip() for part in args.citation_label_order.split(",")] if args.citation_label_order else None
    datasets = [*BUILDERS, *CITATION] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        if dataset in BUILDERS:
            BUILDERS[dataset](node_root, raw_dir)
        else:
            build_citation(dataset, node_root, order)


if __name__ == "__main__":
    main()
