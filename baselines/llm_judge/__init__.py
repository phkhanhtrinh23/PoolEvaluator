"""Full-coverage LLM-as-judge baseline."""

from typing import Any, Callable, Sequence

import numpy as np

from .._common import as_responses


class LLMJudge:
    """Judge every target item and score models against the selected answer."""

    name = "LLM-as-judge"

    def evaluate(
        self,
        responses: Sequence[Sequence[Any]],
        judge: Callable[[int, list[Any]], Any],
    ) -> np.ndarray:
        array = as_responses(responses)
        judged = np.empty(array.shape[0], dtype=object)
        for item, row in enumerate(array):
            judged[item] = judge(item, list(dict.fromkeys(row.tolist())))
        return (array == judged[:, None]).mean(axis=0).astype(float)
