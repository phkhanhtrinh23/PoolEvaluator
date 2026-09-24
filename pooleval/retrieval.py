"""Target-to-meta-subset retrieval for Stage 1 (paper Eq. 9--10)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .data import Text2SQLItem
from .encoders import HashedBagOfWords, rank_by_similarity


def select_subsets(
    target_items: Sequence[Text2SQLItem],
    calibration_items: Sequence[Text2SQLItem],
    k: int = 15,
    encoder: Any = None,
) -> tuple[list[Text2SQLItem], list[tuple[str, float]]]:
    """Retrieve top-K database subsets by cosine similarity of mean question embeddings.

    ``encoder`` is the fixed phi (ColBERTv2 in the paper configuration); it defaults to
    the hashed bag of words so unit tests stay offline.
    """
    encoder = encoder or HashedBagOfWords()
    groups: dict[str, list[Text2SQLItem]] = defaultdict(list)
    for item in calibration_items:
        groups[item.db_id].append(item)
    target = encoder.encode([item.question for item in target_items])
    subsets = {
        db_id: encoder.encode([item.question for item in items]) for db_id, items in groups.items()
    }
    scored = rank_by_similarity(target, subsets)
    selected_ids = {db_id for db_id, _ in scored[: max(0, int(k))]}
    return [item for item in calibration_items if item.db_id in selected_ids], scored
