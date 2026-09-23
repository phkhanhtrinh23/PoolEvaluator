"""Target-to-meta-subset retrieval for Stage 1 (paper Eq. 9--10)."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Sequence

import numpy as np

from .data import Text2SQLItem


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _embedding(texts: Sequence[str], dimensions: int = 2048) -> np.ndarray:
    matrix = np.zeros((len(texts), dimensions), dtype=float)
    for row, text in enumerate(texts):
        for token in _TOKEN.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % dimensions
            matrix[row, index] += 1.0
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1.0)


def select_subsets(
    target_items: Sequence[Text2SQLItem],
    calibration_items: Sequence[Text2SQLItem],
    k: int = 15,
) -> tuple[list[Text2SQLItem], list[tuple[str, float]]]:
    """Retrieve top-K database subsets by cosine similarity of mean question embeddings."""
    groups: dict[str, list[Text2SQLItem]] = defaultdict(list)
    for item in calibration_items:
        groups[item.db_id].append(item)
    target_vector = _embedding([item.question for item in target_items]).mean(axis=0)
    scored: list[tuple[str, float]] = []
    for db_id, items in groups.items():
        vector = _embedding([item.question for item in items]).mean(axis=0)
        denom = max(float(np.linalg.norm(target_vector) * np.linalg.norm(vector)), 1e-12)
        scored.append((db_id, float(np.dot(target_vector, vector) / denom)))
    scored.sort(key=lambda row: (-row[1], row[0]))
    selected_ids = {db_id for db_id, _ in scored[: max(0, int(k))]}
    return [item for item in calibration_items if item.db_id in selected_ids], scored
