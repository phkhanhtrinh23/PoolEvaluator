"""LLaGA runner (classification experts, HO and ND templates) on the official LLaGA code.

Run in LLaGA's environment; the job's ``runtime`` is the cloned LLaGA repository and
``checkpoint``/``base`` are the released linear projector and Vicuna-7B-v1.5-16k.  Graph
tokens are built as in LLaGA's ``eval/eval_pretrain.py``:

* node embeddings concatenate the three SimTeG text encoders (sbert 384 + roberta 1024 +
  e5 1024 = 2432 dimensions); ``--embedding-dir`` may hold precomputed
  ``<key>_simteg.pt`` tensors instead;
* ND (neighborhood detail): the fixed-shape 2-hop, 10-neighbor node sequence from
  ``get_fix_shape_subgraph_sequence_fast`` plus the 111-dimensional Laplacian encoding
  from ``build_laplacian_emb``;
* HO (hop-field overview): the center's embedding propagated 0, 1, 2 hops with the
  symmetric-normalized, self-looped adjacency (``generate_multi_hop_x``).

Answers are generated greedily and mapped to the class whose name they mention.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner_common import (  # noqa: E402
    load_graph, load_job, load_text, neighbor_lists, parse_args, parse_class, write_output,
)

SIMTEG_ENCODERS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-roberta-large-v1",
    "intfloat/e5-large",
)
USE_HOP = 2
SAMPLE_SIZE = 10


def simteg_embeddings(texts: list[str], nodes: list[int], device: str) -> dict[int, torch.Tensor]:
    from sentence_transformers import SentenceTransformer

    chosen = [texts[n] for n in nodes]
    parts = []
    for name in SIMTEG_ENCODERS:
        encoder = SentenceTransformer(name, device=device)
        parts.append(torch.as_tensor(encoder.encode(chosen, batch_size=64, convert_to_numpy=True)))
        del encoder
        torch.cuda.empty_cache()
    stacked = torch.cat(parts, dim=-1).float()
    return {node: stacked[i] for i, node in enumerate(nodes)}


def hop_embeddings(
    neighbors: list[list[int]], centers: list[int], base: dict[int, torch.Tensor]
) -> dict[int, list[torch.Tensor]]:
    """x, A x, A^2 x at each center, A = D^-1/2 (A+I) D^-1/2 with full-graph degrees."""
    degree = {v: len(neighbors[v]) + 1 for v in base}

    def propagate(values: dict[int, torch.Tensor], targets: set[int]) -> dict[int, torch.Tensor]:
        out = {}
        for v in targets:
            total = values[v] / degree[v]
            for u in neighbors[v]:
                if u in values:
                    total = total + values[u] / (degree[u] ** 0.5 * degree[v] ** 0.5)
            out[v] = total
        return out

    one_hop = set(centers) | {u for c in centers for u in neighbors[c]}
    first = propagate(base, one_hop)
    second = propagate(first, set(centers))
    return {c: [base[c], first[c], second[c]] for c in centers}


def main() -> None:
    args = parse_args(
        __doc__,
        [("--template", {"choices": ["HO", "ND"], "default": None}),
         ("--embedding-dir", {"default": None}),
         ("--max-new-tokens", {"type": int, "default": 64})],
    )
    job = load_job(args.job)
    template = args.template or ("HO" if "-HO-" in f"-{Path(job['checkpoint']).name}-".replace("_", "-") else "ND")
    runtime = Path(job["runtime"]).resolve()
    os.chdir(runtime)
    sys.path[:0] = [str(runtime), str(runtime / "utils")]
    from model.builder import load_pretrained_model
    from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX
    from utils.conversation import SeparatorStyle, conv_templates
    from utils.data_process import build_laplacian_emb, get_fix_shape_subgraph_sequence_fast
    from utils.utils import disable_torch_init, get_model_name_from_path, tokenizer_graph_token

    disable_torch_init()
    model_path = str(Path(job["checkpoint"]).resolve()) if Path(job["checkpoint"]).exists() else job["checkpoint"]
    tokenizer, model, _ = load_pretrained_model(model_path, job["base"], get_model_name_from_path(model_path))
    model = model.to(torch.float16).to(args.device).eval()
    structure = None
    if template == "ND":
        laplacian = runtime / "dataset" / f"laplacian_{USE_HOP}_{SAMPLE_SIZE}.pt"
        if not laplacian.exists():
            laplacian.parent.mkdir(parents=True, exist_ok=True)
            build_laplacian_emb(USE_HOP, SAMPLE_SIZE)
        structure = torch.load(laplacian)

    names = job["class_names"]
    question = (
        f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes come from a {job.get('domain', 'graph')}, "
        f"we need to classify the center node into {len(names)} classes: {', '.join(names)}, "
        "please tell me which class the center node belongs to?"
    )
    conversation = conv_templates["v1"].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    input_ids = tokenizer_graph_token(
        conversation.get_prompt(), tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(args.device)
    stop = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2

    output: dict[str, list[int]] = {}
    for entry in job["graphs"]:
        x, edges = load_graph(entry)
        texts = load_text(entry)
        if texts is None:
            raise SystemExit(f"LLaGA needs node text for graph {entry['key']}")
        neighbors = neighbor_lists(edges, x.shape[0])
        centers = [int(n) for n in entry["nodes"]]
        random.seed(args.seed)
        if template == "ND":
            sequences = {c: get_fix_shape_subgraph_sequence_fast(neighbors, c, USE_HOP, SAMPLE_SIZE) for c in centers}
            needed = sorted({n for seq in sequences.values() for n in seq if n != DEFAULT_GRAPH_PAD_ID})
        else:
            needed = sorted({v for c in centers for u in [c, *neighbors[c]] for v in [u, *neighbors[u]]})
        cached = Path(args.embedding_dir) / f"{entry['key']}_simteg.pt" if args.embedding_dir else None
        if cached is not None and cached.exists():
            table = torch.load(cached)
            base = {n: table[n].float() for n in needed}
        else:
            base = simteg_embeddings(texts, needed, args.device)
        hops = hop_embeddings(neighbors, centers, base) if template == "HO" else None
        labels: list[int] = []
        for center in centers:
            if template == "ND":
                graph = torch.tensor([sequences[center]], dtype=torch.long)
                dim = next(iter(base.values())).shape[0]
                graph_emb = torch.zeros((1, graph.shape[1], dim))
                for position, node in enumerate(sequences[center]):
                    if node != DEFAULT_GRAPH_PAD_ID:
                        graph_emb[0, position] = base[node]
                graph_emb = torch.cat([graph_emb, structure.unsqueeze(0)], dim=-1)
            else:
                graph = torch.tensor([[center] * (USE_HOP + 1)], dtype=torch.long)
                graph_emb = torch.stack(hops[center], dim=0).unsqueeze(0)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().to(args.device),
                    graph=graph.to(args.device),
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            text = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            if text.endswith(stop):
                text = text[: -len(stop)].strip()
            labels.append(parse_class(text, names))
        output[entry["key"]] = labels
        print(f"[llaga-{template}] {entry['key']}: {len(labels)} nodes", file=sys.stderr)
    write_output(args.output, output)


if __name__ == "__main__":
    main()
