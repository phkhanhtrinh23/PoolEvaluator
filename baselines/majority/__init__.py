"""Majority-vote agreement baseline."""

from typing import Any, Sequence

import numpy as np

from .._common import as_responses, majority_answers


class Majority:
    """Estimate accuracy by agreement with the per-item pool majority."""

    name = "Majority"

    def evaluate(
        self, responses: Sequence[Sequence[Any]], weights: Sequence[float] | None = None
    ) -> np.ndarray:
        array = as_responses(responses)
        voted = majority_answers(array, weights)
        return (array == voted[:, None]).mean(axis=0).astype(float)
