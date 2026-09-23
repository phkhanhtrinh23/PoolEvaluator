"""Shared validation and agreement helpers for baseline estimators."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def as_responses(responses: Sequence[Sequence[Any]]) -> np.ndarray:
    array = np.asarray(responses, dtype=object)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("responses must have shape [items, models] with at least two models")
    return array


def as_gold(gold_answers: Sequence[Any], item_count: int) -> np.ndarray:
    gold = np.asarray(gold_answers, dtype=object)
    if gold.shape != (item_count,):
        raise ValueError("gold_answers must contain one answer per item")
    return gold


def measured_accuracy(responses: Sequence[Sequence[Any]], gold_answers: Sequence[Any]) -> np.ndarray:
    array = as_responses(responses)
    gold = as_gold(gold_answers, array.shape[0])
    return (array == gold[:, None]).mean(axis=0).astype(float)


def majority_answers(
    responses: Sequence[Sequence[Any]], weights: Sequence[float] | None = None
) -> np.ndarray:
    array = as_responses(responses)
    model_weights = np.ones(array.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    if model_weights.shape != (array.shape[1],):
        raise ValueError("weights must contain one value per model")
    answers = np.empty(array.shape[0], dtype=object)
    for item, row in enumerate(array):
        counts: dict[Any, int] = {}
        support: dict[Any, float] = {}
        first: dict[Any, int] = {}
        for model, (answer, weight) in enumerate(zip(row.tolist(), model_weights.tolist())):
            try:
                key = answer
                hash(key)
            except (TypeError, ValueError):
                key = repr(answer)
            counts[key] = counts.get(key, 0) + 1
            support[key] = support.get(key, 0.0) + float(weight)
            first.setdefault(key, model)
        winner = max(counts, key=lambda key: (counts[key], support[key], -first[key]))
        answers[item] = row[first[winner]]
    return answers


def mean_pairwise_agreement(responses: Sequence[Sequence[Any]]) -> np.ndarray:
    array = as_responses(responses)
    output = np.zeros(array.shape[1], dtype=float)
    for model in range(array.shape[1]):
        peers = [peer for peer in range(array.shape[1]) if peer != model]
        output[model] = np.mean([np.mean(array[:, model] == array[:, peer]) for peer in peers])
    return output
