"""Average Thresholded Confidence baseline."""

from typing import Literal, Sequence

import numpy as np


class AverageThresholdedConfidence:
    name = "ATC"

    def __init__(self, score: Literal["max_confidence", "negative_entropy"] = "max_confidence"):
        self.score = score

    def _scores(self, probabilities: np.ndarray) -> np.ndarray:
        if self.score == "max_confidence":
            return probabilities.max(axis=1)
        if self.score == "negative_entropy":
            safe = np.clip(probabilities, 1e-12, 1.0)
            return (safe * np.log(safe)).sum(axis=1)
        raise ValueError(f"unknown ATC score: {self.score}")

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
        source_error = 1.0 - np.mean(source.argmax(axis=1) == gold)
        threshold = np.quantile(self._scores(source), np.clip(source_error, 0.0, 1.0))
        return float(np.mean(self._scores(target) >= threshold))
