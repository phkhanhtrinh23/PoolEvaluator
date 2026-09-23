"""One-coin Dawid--Skene for heterogeneous candidate-answer sets."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .._common import as_responses, majority_answers


class DawidSkene:
    """Infer item truths and annotator/model accuracies without target labels.

    The one-coin form supports open-ended tasks because every item may have a
    different candidate-answer set. Wrong-answer mass is uniform over the other
    candidates for that item.
    """

    name = "Dawid--Skene"

    def __init__(self, max_iterations: int = 200, tolerance: float = 1e-8):
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    def evaluate(self, responses: Sequence[Sequence[Any]]) -> np.ndarray:
        array = as_responses(responses)
        voted = majority_answers(array)
        competence = np.clip((array == voted[:, None]).mean(axis=0), 1e-4, 1.0 - 1e-4)
        posterior: list[np.ndarray] = []
        candidates: list[list[Any]] = [list(dict.fromkeys(row.tolist())) for row in array]
        for _ in range(self.max_iterations):
            posterior = []
            for row, choices in zip(array, candidates):
                if len(choices) == 1:
                    posterior.append(np.ones(1, dtype=float))
                    continue
                log_probability = np.full(len(choices), -np.log(len(choices)), dtype=float)
                for choice_index, choice in enumerate(choices):
                    agrees = row == choice
                    log_probability[choice_index] += np.sum(
                        np.where(
                            agrees,
                            np.log(competence),
                            np.log((1.0 - competence) / (len(choices) - 1)),
                        )
                    )
                probability = np.exp(log_probability - np.max(log_probability))
                posterior.append(probability / probability.sum())
            updated = np.zeros(array.shape[1], dtype=float)
            for item, (row, choices, probability) in enumerate(zip(array, candidates, posterior)):
                choice_index = {choice: index for index, choice in enumerate(choices)}
                for model, answer in enumerate(row):
                    updated[model] += probability[choice_index[answer]]
            updated = np.clip(updated / array.shape[0], 1e-4, 1.0 - 1e-4)
            if np.max(np.abs(updated - competence)) < self.tolerance:
                competence = updated
                break
            competence = updated
        return competence
