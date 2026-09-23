"""Task adapters connecting loaded appendix models to prediction matrices."""

from __future__ import annotations

import re
from typing import Any, Sequence

from .models import LoadedModel


_SQL = re.compile(r"\b(?:SELECT|WITH)\b.*", re.IGNORECASE | re.DOTALL)


def text2sql_prompt(question: str, schema: str, evidence: str = "") -> str:
    evidence_block = f"\nEvidence: {evidence}" if evidence else ""
    return (
        "Translate the question to one SQLite query. Return SQL only.\n"
        f"Schema:\n{schema}{evidence_block}\nQuestion: {question}\nSQL:"
    )


def predict_text2sql(loaded: LoadedModel, prompts: Sequence[str], max_new_tokens: int = 512) -> list[str]:
    import torch

    tokenizer = loaded.processor
    output: list[str] = []
    for prompt in prompts:
        batch = tokenizer(prompt, return_tensors="pt")
        device = next(loaded.model.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            generated = loaded.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        if not getattr(loaded.model.config, "is_encoder_decoder", False):
            generated = generated[:, batch["input_ids"].shape[1] :]
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        match = _SQL.search(text)
        sql = match.group(0).strip() if match else text
        output.append(sql.split(";")[0].strip())
    return output


def predict_images(loaded: LoadedModel, images: Sequence[Any], class_names: Sequence[str]) -> list[int]:
    import torch

    if loaded.spec.backend == "timm":
        batch = torch.stack([loaded.processor(image) for image in images])
        device = next(loaded.model.parameters()).device
        with torch.inference_mode():
            return loaded.model(batch.to(device)).argmax(-1).cpu().tolist()
    processor = loaded.processor
    kwargs = {"images": list(images), "return_tensors": "pt"}
    if loaded.spec.backend == "zero_shot_image":
        kwargs["text"] = [f"a photo of {name}" for name in class_names]
        kwargs["padding"] = True
    batch = processor(**kwargs)
    device = next(loaded.model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        result = loaded.model(**batch)
    logits = getattr(result, "logits_per_image", None)
    if logits is None:
        logits = result.logits
    return logits.argmax(-1).cpu().tolist()


def predict_nodes(loaded: LoadedModel, graph: Any, class_names: Sequence[str] | None = None) -> list[int]:
    """Predict nodes for PyG models; graph-language models use prebuilt text prompts."""
    import torch

    if loaded.spec.backend == "pyg":
        loaded.model.eval()
        with torch.inference_mode():
            try:
                logits = loaded.model(graph.x, graph.edge_index)
            except TypeError:
                logits = loaded.model(graph.x)
        return logits.argmax(-1).cpu().tolist()
    if loaded.spec.backend == "external":
        runner = getattr(graph, "external_runner", None)
        if runner is None:
            raise ValueError(
                f"{loaded.spec.name} requires its official runtime ({loaded.spec.runtime_repo}); "
                "attach graph.external_runner(checkpoint, graph, class_names)"
            )
        return list(runner(loaded.model, graph, class_names))
    prompts = getattr(graph, "node_prompts", None)
    if prompts is None or not class_names:
        raise ValueError("graph-language models require graph.node_prompts and class_names")
    answers = predict_text2sql(loaded, prompts, max_new_tokens=32)
    lowered = [name.casefold() for name in class_names]
    return [next((i for i, name in enumerate(lowered) if name in answer.casefold()), -1) for answer in answers]
