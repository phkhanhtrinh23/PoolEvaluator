"""Fixed item encoders phi for Stage 1 subset retrieval (paper Eq. 9).

Every encoder maps a batch of items to one L2-normalized row per item.  Retrieval
averages these rows per dataset and compares datasets by cosine similarity.

* Text2SQL questions: ColBERTv2 (``colbert-ir/colbertv2.0``).  Each question is
  encoded exactly as ColBERT encodes a document -- ``[CLS] [D] tokens [SEP]``, BERT,
  the 128-d linear projection, per-token L2 normalization, punctuation masked -- and
  the item embedding is the mean of its token vectors.
* Images: DINOv2 (``facebook/dinov2-small``) CLS embeddings.  No pool member shares
  this backbone.
* Graph nodes: parameter-free propagated features, rows of D^-1/2 (A+I) D^-1/2 applied
  ``hops`` times to X.
* ``hashed-bow``: a dependency-free hashed bag of words, for tests only.
"""

from __future__ import annotations

import hashlib
import re
import string
from pathlib import Path
from typing import Any, Sequence

import numpy as np


COLBERT_ID = "colbert-ir/colbertv2.0"
DINOV2_ID = "facebook/dinov2-small"
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def _device(device: str | None) -> Any:
    import torch

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def local_source(model_id: str, cache_dir: str | Path | None, kind: str = "encoders") -> str:
    """Return the downloaded directory for ``model_id`` when present, else the hub id."""
    if cache_dir is not None:
        path = Path(cache_dir).expanduser().resolve() / kind / model_id.rsplit("/", 1)[-1]
        if path.exists():
            return str(path)
    return model_id


class HashedBagOfWords:
    """Deterministic hashed token counts; used by tests and offline smoke runs."""

    name = "hashed-bow"

    def __init__(self, dimensions: int = 2048):
        self.dimensions = int(dimensions)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=float)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.casefold()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                matrix[row, int.from_bytes(digest, "little") % self.dimensions] += 1.0
        return _normalize(matrix)


class ColBERTEncoder:
    """ColBERTv2 document-side encoder with mean pooling over token embeddings."""

    name = COLBERT_ID

    def __init__(
        self,
        source: str = COLBERT_ID,
        device: str | None = None,
        max_length: int = 180,
        batch_size: int = 64,
    ):
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoTokenizer, BertModel
        except ImportError as exc:
            raise RuntimeError("install model dependencies: pip install -e '.[models]'") from exc
        path = Path(source)
        if path.exists():
            weights = path / "model.safetensors"
        else:
            from huggingface_hub import hf_hub_download

            weights = Path(hf_hub_download(source, "model.safetensors"))
        self.device = _device(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.bert = BertModel.from_pretrained(source, add_pooling_layer=False).to(self.device).eval()
        state = load_file(str(weights))
        if "linear.weight" not in state:
            raise ValueError(f"{source} has no ColBERT projection 'linear.weight'")
        projection = state["linear.weight"]
        self.linear = torch.nn.Linear(projection.shape[1], projection.shape[0], bias=False)
        self.linear.weight.data.copy_(projection)
        self.linear = self.linear.to(self.device).eval()
        # ColBERT marks documents with [unused1] and masks punctuation tokens.
        self.doc_marker = self.tokenizer.convert_tokens_to_ids("[unused1]")
        self.skiplist = {
            self.tokenizer.encode(symbol, add_special_tokens=False)[0]
            for symbol in string.punctuation
        }

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        rows: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [". " + text for text in texts[start : start + self.batch_size]]
            tokens = self.tokenizer(
                batch,
                padding="longest",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            ids = tokens["input_ids"]
            ids[:, 1] = self.doc_marker
            attention = tokens["attention_mask"]
            keep = attention.bool() & ~torch.isin(ids, torch.tensor(sorted(self.skiplist)))
            with torch.inference_mode():
                hidden = self.bert(
                    input_ids=ids.to(self.device), attention_mask=attention.to(self.device)
                ).last_hidden_state
                vectors = torch.nn.functional.normalize(self.linear(hidden), p=2, dim=2)
                mask = keep.to(self.device).unsqueeze(-1).to(vectors.dtype)
                pooled = (vectors * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            rows.append(pooled.float().cpu().numpy())
        return _normalize(np.concatenate(rows, axis=0)) if rows else np.zeros((0, 128))


class DINOv2Encoder:
    """Frozen DINOv2 CLS embeddings for PIL images."""

    name = DINOV2_ID

    def __init__(self, source: str = DINOV2_ID, device: str | None = None, batch_size: int = 64):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("install model dependencies: pip install -e '.[models]'") from exc
        self.device = _device(device)
        self.batch_size = int(batch_size)
        self.processor = AutoImageProcessor.from_pretrained(source)
        self.model = AutoModel.from_pretrained(source).to(self.device).eval()

    def encode(self, images: Sequence[Any]) -> np.ndarray:
        import torch

        rows: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = self.processor(
                images=[image.convert("RGB") for image in images[start : start + self.batch_size]],
                return_tensors="pt",
            )
            with torch.inference_mode():
                output = self.model(pixel_values=batch["pixel_values"].to(self.device))
            rows.append(output.last_hidden_state[:, 0].float().cpu().numpy())
        return _normalize(np.concatenate(rows, axis=0)) if rows else np.zeros((0, 384))


def propagated_features(x: Any, edge_index: Any, hops: int = 2) -> np.ndarray:
    """Rows of (D^-1/2 (A+I) D^-1/2)^hops X, L2-normalized; no learned parameters."""
    import scipy.sparse as sp

    features = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=np.float32)
    edges = np.asarray(
        edge_index.detach().cpu().numpy() if hasattr(edge_index, "detach") else edge_index
    )
    n = features.shape[0]
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n)
    ).tocsr()
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32) + sp.eye(n, dtype=np.float32)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    scale = sp.diags(1.0 / np.sqrt(np.maximum(degree, 1.0)))
    operator = scale @ adjacency @ scale
    for _ in range(int(hops)):
        features = operator @ features
    return _normalize(np.asarray(features, dtype=float))


def text_encoder(name: str, cache_dir: str | Path | None = None, device: str | None = None) -> Any:
    if name == "hashed-bow":
        return HashedBagOfWords()
    return ColBERTEncoder(local_source(name, cache_dir), device=device)


def image_encoder(name: str, cache_dir: str | Path | None = None, device: str | None = None) -> Any:
    return DINOv2Encoder(local_source(name, cache_dir), device=device)


def dataset_embedding(item_embeddings: np.ndarray) -> np.ndarray:
    """h(D): the mean item embedding of a dataset (paper Eq. 9)."""
    return np.asarray(item_embeddings, dtype=float).mean(axis=0)


def rank_by_similarity(
    target: np.ndarray, subsets: dict[Any, np.ndarray]
) -> list[tuple[Any, float]]:
    """Cosine similarity between h(target) and every h(P_r), highest first."""
    target_vector = dataset_embedding(target)
    scored: list[tuple[Any, float]] = []
    for key, embeddings in subsets.items():
        vector = dataset_embedding(embeddings)
        denom = max(float(np.linalg.norm(target_vector) * np.linalg.norm(vector)), 1e-12)
        scored.append((key, float(np.dot(target_vector, vector) / denom)))
    scored.sort(key=lambda row: (-row[1], str(row[0])))
    return scored
