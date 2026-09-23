"""Difference-of-Confidence baseline."""

from typing import Sequence

import numpy as np


class DifferenceOfConfidence:
    name = "DoC"

    def evaluate(
        self,
        source_probabilities: Sequence[Sequence[float]],
        target_probabilities: Sequence[Sequence[float]],
        source_gold: Sequence[int],
    ) -> float:
        source = np.asarray(source_probabilities, dtype=float)
        target = np.asarray(target_probabilities, dtype=float)
        gold = np.asarray(source_gold, dtype=int)
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
            raise ValueError("source and target probabilities must have shape [items, classes]")
        if gold.shape != (source.shape[0],):
            raise ValueError("source_gold must contain one label per source item")
        source_accuracy = np.mean(source.argmax(axis=1) == gold)
        confidence_shift = source.max(axis=1).mean() - target.max(axis=1).mean()
        return float(np.clip(source_accuracy - confidence_shift, 0.0, 1.0))
