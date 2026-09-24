"""Open-weight judges for the image and node pools (paper Appendices C and D).

* Image: Qwen2.5-VL-72B-Instruct, loaded in-process with Transformers.
* Node: GOFA, a generative graph foundation model on a Mistral-7B backbone.  GOFA pins
  its own dependencies, so it runs in its own Python environment as a persistent
  subprocess (``scripts/runners/gofa_judge.py``) that answers one JSON line per query.

Both judges choose among the pool's candidate answers or return -1 (NONE), exactly
like the Text2SQL API judges.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ROOT
from .judges import _parse


@dataclass(frozen=True)
class ImageJudgeItem:
    image: Any
    candidates: tuple[str, ...]
    domain: str = "image"


@dataclass(frozen=True)
class NodeJudgeItem:
    node_text: str
    neighbor_texts: tuple[str, ...]
    candidates: tuple[str, ...]
    domain: str = "graph"


def judge_source(model_id: str, cache_dir: str | Path | None) -> str:
    """Downloaded judge directory under ``<cache>/judges/`` when present, else the hub id."""
    if cache_dir is not None:
        path = Path(cache_dir).expanduser().resolve() / "judges" / model_id.rsplit("/", 1)[-1]
        if path.exists():
            return str(path)
    return model_id


def image_prompt(item: ImageJudgeItem) -> str:
    options = "\n".join(f"{index}: {name}" for index, name in enumerate(item.candidates))
    return (
        f"You are a strict, independent judge of {item.domain} classifiers. Look at the image "
        "and choose the candidate label that correctly describes it. Choose -1 when no "
        f"candidate is correct.\n\nCandidates:\n{options}\n\n"
        'Return exactly {"choice": INDEX}.'
    )


class QwenVLJudge:
    """Qwen2.5-VL judge; ``device_map='auto'`` shards the 72B checkpoint across GPUs."""

    def __init__(
        self,
        source: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        device_map: str | None = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 32,
    ):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("install model dependencies: pip install -e '.[models]'") from exc
        from .models import resolve_inference_dtype

        self.name = Path(source).name
        self.max_new_tokens = int(max_new_tokens)
        self.processor = AutoProcessor.from_pretrained(source)
        kwargs: dict[str, Any] = {"torch_dtype": resolve_inference_dtype(dtype, device_map)}
        if device_map:
            kwargs["device_map"] = device_map
        self.model = AutoModelForImageTextToText.from_pretrained(source, **kwargs).eval()

    def choose(self, item: ImageJudgeItem) -> int:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item.image.convert("RGB")},
                    {"type": "text", "text": image_prompt(item)},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return _parse(text, len(item.candidates))


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gofa_graph(item: NodeJudgeItem, max_chars: int = 600) -> dict[str, Any]:
    """GOFA's JSON graph: the target node (NODEID.AA), its neighbors, and a question node."""
    if len(item.candidates) + 1 > len(_LETTERS):
        raise ValueError("GOFA prompts support at most 25 candidate labels")
    texts = [item.node_text[:max_chars]] + [text[:max_chars] for text in item.neighbor_texts]
    options = "; ".join(f"{_LETTERS[k]}. {name}" for k, name in enumerate(item.candidates))
    none_letter = _LETTERS[len(item.candidates)]
    question = (
        f"This is a node classification task on a {item.domain}. Which category best describes "
        f"[NODEID.AA]? Options: {options}; {none_letter}. None of the above. "
        "Answer with the letter of one option."
    )
    nodes = {str(index): text for index, text in enumerate(texts)}
    prompt_id = len(texts)
    nodes[str(prompt_id)] = question
    edges: list[dict[str, Any]] = []
    for neighbor in range(1, len(texts)):
        for source, target in ((0, neighbor), (neighbor, 0)):
            edges.append(
                {"source": source, "target": target, "relation": {"content": "These two nodes are linked in the graph."}}
            )
    for node in range(len(texts)):
        edges.append(
            {"source": node, "target": prompt_id, "relation": {"content": "This edge connects the nodes in graph to a prompt node."}}
        )
        edges.append(
            {"source": prompt_id, "target": node, "relation": {"content": "This edge connects the prompt node to a node in the graph."}}
        )
    return {"question": [prompt_id], "complete": [], "node": nodes, "edge": edges}


def parse_option(text: str, candidates: Sequence[str]) -> int:
    """Map GOFA's free-text answer to a candidate index, or -1 for NONE/unparseable."""
    answer = (text or "").strip()
    allowed = _LETTERS[: len(candidates) + 1]

    def to_index(letter: str) -> int:
        index = allowed.index(letter)
        return index if index < len(candidates) else -1

    for pattern in (
        r"^\(?([A-Z])(?:[\.\):]|\s*$)",
        r"\b(?:answer|option|category)\s*(?:is|:)?\s*\(?([A-Z])\b",
    ):
        match = re.search(pattern, answer, re.IGNORECASE if "answer" in pattern else 0)
        if match and match.group(1).upper() in allowed:
            return to_index(match.group(1).upper())
    lowered = answer.casefold()
    if "none of the above" in lowered:
        return -1
    hits = [k for k, name in enumerate(candidates) if name.casefold() in lowered]
    if len(hits) == 1:
        return hits[0]
    letters = set(re.findall(r"\b([A-Z])\b", answer)) & set(allowed)
    return to_index(letters.pop()) if len(letters) == 1 else -1


class GOFAJudge:
    """Persistent GOFA subprocess speaking one JSON request/response per line."""

    def __init__(
        self,
        python: str,
        runtime_dir: str | Path,
        checkpoint_dir: str | Path,
        checkpoint: str,
        base_model: str,
        max_new_tokens: int = 32,
        runner: str | Path | None = None,
    ):
        self.name = "GOFA"
        runtime = Path(runtime_dir).expanduser().resolve()
        if not (runtime / "modules" / "gofa").is_dir():
            raise FileNotFoundError(
                f"GOFA runtime not found at {runtime}; run `python -m pooleval.models support --task node`"
            )
        self.command = [
            python,
            str(Path(runner) if runner else ROOT / "scripts" / "runners" / "gofa_judge.py"),
            "--runtime",
            str(runtime),
            "--checkpoint-dir",
            str(Path(checkpoint_dir).expanduser().resolve()),
            "--checkpoint",
            checkpoint,
            "--base-model",
            base_model,
            "--max-new-tokens",
            str(int(max_new_tokens)),
        ]
        self.cwd = runtime
        self.process: subprocess.Popen[str] | None = None
        self.last_text = ""

    def _start(self) -> subprocess.Popen[str]:
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            ready = json.loads(self.process.stdout.readline() or "{}")
            if not ready.get("ready"):
                raise RuntimeError("GOFA runner exited before becoming ready; see its stderr")
        return self.process

    def generate(self, graph: Mapping[str, Any]) -> str:
        process = self._start()
        process.stdin.write(json.dumps({"graph": graph}) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("GOFA runner terminated; see its stderr")
        return str(json.loads(line)["text"])

    def choose(self, item: NodeJudgeItem) -> int:
        self.last_text = self.generate(gofa_graph(item))
        return parse_option(self.last_text, item.candidates)

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=60)
        self.process = None


def load_image_judge(config: Mapping[str, Any], cache_dir: str | Path | None, dtype: str) -> QwenVLJudge:
    settings = config["judges"]["image"]
    return QwenVLJudge(
        judge_source(settings["model"], cache_dir),
        device_map=settings.get("device_map", "auto"),
        dtype=dtype,
        max_new_tokens=settings.get("max_new_tokens", 32),
    )


def load_node_judge(config: Mapping[str, Any], cache_dir: str | Path) -> GOFAJudge:
    settings = config["judges"]["node"]
    root = Path(cache_dir).expanduser().resolve()
    base = settings["base_model"]
    local_base = root / "shared" / base.replace("/", "__")
    return GOFAJudge(
        python=settings.get("python", "python"),
        runtime_dir=root / "runtimes" / "GOFA",
        checkpoint_dir=root / "judges" / "GOFA",
        checkpoint=settings["checkpoint"],
        base_model=str(local_base) if local_base.exists() else base,
        max_new_tokens=settings.get("max_new_tokens", 32),
    )
