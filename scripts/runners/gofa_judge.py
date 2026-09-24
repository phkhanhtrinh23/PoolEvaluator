"""Serve GOFA judgments over stdin/stdout, one JSON object per line.

Run with the interpreter of GOFA's own environment (built from its environment.yml);
``pooleval.local_judges.GOFAJudge`` starts and drives this script.  Loading follows the
official ``chat_gofa.py``: GOFAMistral with decoder LoRA, the ICAE weights from
``--checkpoint-dir``, then the GOFA checkpoint.

Protocol: after loading, print {"ready": true}; then for every request line
{"graph": <GOFA JSON graph>} print {"text": <generated answer>}.  Library output is
redirected to stderr so it cannot corrupt the protocol stream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, help="clone of github.com/JiaruiFeng/GOFA")
    parser.add_argument("--checkpoint-dir", required=True, help="holds the GOFA and ICAE weights")
    parser.add_argument("--checkpoint", default="mistral_qamag03_best_ckpt.pth")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    os.chdir(args.runtime)
    sys.path.insert(0, args.runtime)
    import torch
    from modules.gofa import GOFAMistral, GOFAMistralConfig, ModelArguments, TrainingArguments
    from modules.utils import prepare_gofa_graph_input

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args, training_args, gofa_args = ModelArguments(), TrainingArguments(), GOFAMistralConfig()
    model_args.dec_lora = True
    model_args.checkpoint_dir = args.checkpoint_dir
    model_args.model_name_or_path = args.base_model
    training_args.bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    gofa = GOFAMistral((model_args, training_args, gofa_args))
    gofa.load_pretrained(os.path.join(args.checkpoint_dir, args.checkpoint))
    gofa.to(device)
    gofa.eval()
    protocol.write(json.dumps({"ready": True}) + "\n")

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        graph = prepare_gofa_graph_input(request["graph"], device=device)
        with torch.no_grad():
            generated = gofa.generate(graph, max_length=args.max_new_tokens)
        text = generated[0] if isinstance(generated, (list, tuple)) else generated
        protocol.write(json.dumps({"text": str(text)}) + "\n")


if __name__ == "__main__":
    main()
