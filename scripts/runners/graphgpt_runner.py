"""GraphGPT-7B runner on the official GraphGPT code (``graphgpt/eval/run_graphgpt.py``).

Run in GraphGPT's environment; the job's ``runtime`` is the cloned GraphGPT repository,
``checkpoint`` is GraphGPT-7B-mix-all, and ``auxiliary`` is the Arxiv-PubMed-GraphCLIP-GT
graph encoder.  Each target node is presented as a center-first 2-hop subgraph (at most
10 new neighbors per expanded node) whose node features feed the graph tower.  The
tower takes 128-dimensional features: ogbn-arxiv provides them natively; other graphs are
projected to 128 dimensions with PCA fitted on the graph's own features.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner_common import (  # noqa: E402
    load_graph, load_job, load_text, neighbor_lists, parse_args, parse_class, sample_subgraph, write_output,
)

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"
GRAPH_DIM = 128


def to_graph_features(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == GRAPH_DIM:
        return x
    if x.shape[1] < GRAPH_DIM:
        return torch.cat([x, torch.zeros(x.shape[0], GRAPH_DIM - x.shape[1])], dim=1)
    centered = x - x.mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(centered, q=GRAPH_DIM, center=False)
    return centered @ components[:, :GRAPH_DIM]


def main() -> None:
    args = parse_args(
        __doc__,
        [("--hops", {"type": int, "default": 2}),
         ("--per-hop", {"type": int, "default": 10}),
         ("--max-text-chars", {"type": int, "default": 1500}),
         ("--max-new-tokens", {"type": int, "default": 64})],
    )
    job = load_job(args.job)
    runtime = Path(job["runtime"]).resolve()
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    from transformers import AutoTokenizer

    from graphgpt.conversation import SeparatorStyle, conv_templates
    from graphgpt.model import (  # noqa: F401  (names re-exported by graphgpt.model)
        CLIP, GraphLlamaForCausalLM, graph_transformer, load_model_pretrained, transfer_param_tograph,
    )
    from graphgpt.model.utils import KeywordsStoppingCriteria
    from torch_geometric.data import Data

    checkpoint = job["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = GraphLlamaForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True
    ).to(args.device)
    use_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)
    clip_graph, graph_args = load_model_pretrained(CLIP, job["auxiliary"])
    graph_tower = transfer_param_tograph(clip_graph, graph_transformer(graph_args))
    model.get_model().graph_tower = graph_tower.to(args.device)
    graph_tower.to(device=args.device, dtype=torch.float16)
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_start_end
    if use_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN]
        )
    model.eval()

    names = job["class_names"]
    domain = job.get("domain", "graph")
    output: dict[str, list[int]] = {}
    for entry in job["graphs"]:
        x, edges = load_graph(entry)
        features = to_graph_features(x).to(torch.float16)
        texts = load_text(entry)
        neighbors = neighbor_lists(edges, x.shape[0])
        rng = random.Random(args.seed)
        labels: list[int] = []
        for center in entry["nodes"]:
            nodes, local_edges = sample_subgraph(neighbors, int(center), args.hops, args.per_hop, rng)
            graph = Data(graph_node=features[nodes], edge_index=local_edges, target_node=torch.tensor([0]))
            information = ""
            if texts is not None:
                information = f", with the following information: \n{texts[int(center)][: args.max_text_chars]}"
            question = (
                f"Given a {domain}: \n{DEFAULT_GRAPH_TOKEN}\nwhere the 0th node is the target node{information}\n"
                f"Question: Which of the following categories does the target node belong to: {', '.join(names)}? "
                "Directly give the category name."
            )
            patch = DEFAULT_GRAPH_PATCH_TOKEN * len(nodes)
            if use_start_end:
                patch = DEFAULT_G_START_TOKEN + patch + DEFAULT_G_END_TOKEN
            conversation = conv_templates["graphchat_v1"].copy()
            conversation.append_message(conversation.roles[0], question.replace(DEFAULT_GRAPH_TOKEN, patch))
            conversation.append_message(conversation.roles[1], None)
            input_ids = torch.as_tensor(tokenizer([conversation.get_prompt()]).input_ids).to(args.device)
            stop = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_data=graph.to(args.device),
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    stopping_criteria=[KeywordsStoppingCriteria([stop], tokenizer, input_ids)],
                )
            text = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            if text.endswith(stop):
                text = text[: -len(stop)].strip()
            labels.append(parse_class(text, names))
        output[entry["key"]] = labels
        print(f"[graphgpt] {entry['key']}: {len(labels)} nodes", file=sys.stderr)
    write_output(args.output, output)


if __name__ == "__main__":
    main()
