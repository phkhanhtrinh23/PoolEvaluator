"""Generalized Disagreement Equality baseline."""

from typing import Any, Sequence

import numpy as np


class GeneralizedDisagreementEquality:
    name = "GDE"

    def evaluate(self, predictions_a: Sequence[Any], predictions_b: Sequence[Any]) -> float:
        first = np.asarray(predictions_a, dtype=object)
        second = np.asarray(predictions_b, dtype=object)
        if first.ndim != 1 or first.shape != second.shape:
            raise ValueError("replica predictions must be equally sized vectors")
        return float(1.0 - np.mean(first != second))
