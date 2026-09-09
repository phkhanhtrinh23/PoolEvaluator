"""Coverage-based prior-accuracy bounds and effective sample-size diagnostics.

This module implements the quantities derived in
``coverage_hoeffding_derivation.md`` and ``MTM08_analysis.md``.  The matching
ESS, ``1 / sum(w**2)``, controls concentration of a weighted calibration
average.  It is deliberately kept separate from the curvature-matched ESS of
a Beta prior, which is the sum of its two shape parameters.
"""
from itertools import combinations

import numpy as np


def shifted_cosine_similarity(target_embeddings, calibration_embeddings):
    """Return shifted cosine similarities in ``[0, 1]``.

    Rows are normalized internally.  Zero vectors are rejected because cosine
    similarity, and hence the coverage theorem, is undefined for them.
    """
    target = _unit_rows(target_embeddings, "target_embeddings")
    calibration = _unit_rows(calibration_embeddings, "calibration_embeddings")
    if target.shape[1] != calibration.shape[1]:
        raise ValueError("target and calibration embeddings must have equal width")
    return np.clip((1.0 + target @ calibration.T) / 2.0, 0.0, 1.0)


def greedy_dataset_coverage(similarity, dataset_ids, budget):
    """Greedily select calibration datasets by facility-location coverage.

    ``similarity[x, z]`` is the similarity between target item ``x`` and
    calibration item ``z``.  All calibration items with the same dataset ID are
    added together, so ``budget`` counts datasets rather than items.
    """
    similarity = _similarity_matrix(similarity)
    dataset_ids = np.asarray(dataset_ids)
    if dataset_ids.ndim != 1 or len(dataset_ids) != similarity.shape[1]:
        raise ValueError("dataset_ids must have one entry per calibration item")
    datasets = list(dict.fromkeys(dataset_ids.tolist()))
    if not 1 <= int(budget) <= len(datasets):
        raise ValueError("budget must be between one and the number of datasets")

    group_items = {dataset: np.flatnonzero(dataset_ids == dataset)
                   for dataset in datasets}
    selected = []
    best = np.zeros(similarity.shape[0], dtype=float)
    for _ in range(int(budget)):
        winner = None
        winner_best = None
        winner_gain = -np.inf
        for dataset in datasets:
            if dataset in selected:
                continue
            candidate_best = np.maximum(
                best, similarity[:, group_items[dataset]].max(axis=1)
            )
            gain = float((candidate_best - best).sum())
            if gain > winner_gain:
                winner = dataset
                winner_best = candidate_best
                winner_gain = gain
        selected.append(winner)
        best = winner_best

    selected_items = np.flatnonzero(np.isin(dataset_ids, selected))
    return {
        "selected_datasets": selected,
        "selected_items": selected_items,
        "coverage": float(best.sum()),
        "normalized_coverage": float(best.mean()),
        "best_similarity": best,
    }


def optimal_dataset_coverage(similarity, dataset_ids, budget):
    """Compute exact optimal coverage for small diagnostic problems."""
    similarity = _similarity_matrix(similarity)
    dataset_ids = np.asarray(dataset_ids)
    if dataset_ids.ndim != 1 or len(dataset_ids) != similarity.shape[1]:
        raise ValueError("dataset_ids must have one entry per calibration item")
    datasets = list(dict.fromkeys(dataset_ids.tolist()))
    if not 1 <= int(budget) <= len(datasets):
        raise ValueError("budget must be between one and the number of datasets")

    best_value = -np.inf
    best_datasets = None
    best_items = None
    for chosen in combinations(datasets, int(budget)):
        items = np.flatnonzero(np.isin(dataset_ids, chosen))
        value = float(similarity[:, items].max(axis=1).sum())
        if value > best_value:
            best_value = value
            best_datasets = list(chosen)
            best_items = items
    return {
        "selected_datasets": best_datasets,
        "selected_items": best_items,
        "coverage": best_value,
        "normalized_coverage": best_value / similarity.shape[0],
    }


def matching_diagnostics(similarity, selected_items):
    """Return nearest-match assignments, weights, coverage, and matching ESS."""
    similarity = _similarity_matrix(similarity)
    selected = np.asarray(selected_items, dtype=int)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("selected_items must be a non-empty one-dimensional array")
    if np.any((selected < 0) | (selected >= similarity.shape[1])):
        raise ValueError("selected_items contains an out-of-range index")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("selected_items must contain distinct indices")

    selected_similarity = similarity[:, selected]
    local_assignment = np.argmax(selected_similarity, axis=1)
    counts = np.bincount(local_assignment, minlength=len(selected)).astype(float)
    weights = counts / similarity.shape[0]
    best = selected_similarity[np.arange(similarity.shape[0]), local_assignment]
    return {
        "assignment": selected[local_assignment],
        "weights": weights,
        "effective_sample_size": float(1.0 / np.sum(weights ** 2)),
        "coverage": float(best.sum()),
        "normalized_coverage": float(best.mean()),
        "mean_match_distance": float(np.mean(2.0 * np.sqrt(1.0 - best))),
    }


def prior_accuracy_diagnostics(similarity, selected_items, outcomes,
                               lipschitz, delta=0.05):
    """Compute matched/uniform priors and their conditional Hoeffding bounds.

    ``outcomes`` may be ``[calibration_items]`` or
    ``[models, calibration_items]``.  Returned estimator and bound arrays have
    one entry per model.
    """
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be strictly between zero and one")
    similarity = _similarity_matrix(similarity)
    selected = np.asarray(selected_items, dtype=int)
    match = matching_diagnostics(similarity, selected)

    values = np.asarray(outcomes, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != similarity.shape[1]:
        raise ValueError("outcomes must have one column per calibration item")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("outcomes must lie in [0, 1]")

    model_count = values.shape[0]
    constants = np.broadcast_to(np.asarray(lipschitz, dtype=float), (model_count,))
    if np.any(constants < 0.0):
        raise ValueError("lipschitz constants must be non-negative")

    selected_values = values[:, selected]
    matched = selected_values @ match["weights"]
    uniform = selected_values.mean(axis=1)
    uniform_weights = np.full(len(selected), 1.0 / len(selected))
    mismatch = 0.5 * float(np.abs(uniform_weights - match["weights"]).sum())
    transfer = (2.0 * constants
                * np.sqrt(1.0 - match["normalized_coverage"]))
    exact_transfer = constants * match["mean_match_distance"]
    log_term = np.log(2.0 / float(delta)) / 2.0
    matched_sampling = np.sqrt(log_term / match["effective_sample_size"])
    uniform_sampling = np.sqrt(log_term / len(selected))

    return {
        **match,
        "matched_accuracy": matched,
        "uniform_accuracy": uniform,
        "weight_mismatch": mismatch,
        "transfer_bound": transfer,
        "exact_distance_transfer_bound": exact_transfer,
        "matched_sampling_bound": float(matched_sampling),
        "uniform_sampling_bound": float(uniform_sampling),
        "matched_bound": transfer + matched_sampling,
        "uniform_bound": transfer + mismatch + uniform_sampling,
    }


def beta_effective_sample_size(alpha, beta):
    """Return curvature-matched ESS for a Beta-Bernoulli prior."""
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    try:
        alpha, beta = np.broadcast_arrays(alpha, beta)
    except ValueError as error:
        raise ValueError("alpha and beta must be broadcast-compatible") from error
    if np.any(alpha <= 0.0) or np.any(beta <= 0.0):
        raise ValueError("alpha and beta must be positive")
    return alpha + beta


def beta_power_prior_diagnostics(prior_accuracy, calibration_count, discount,
                                 target_count):
    """Describe a discounted Beta(1,1) calibration prior.

    The added MAP strength is ``discount * calibration_count``.  Total MTM08
    Beta ESS includes the uniform baseline and is therefore two larger.
    """
    accuracy = np.asarray(prior_accuracy, dtype=float)
    if np.any((accuracy < 0.0) | (accuracy > 1.0)):
        raise ValueError("prior_accuracy must lie in [0, 1]")
    if int(calibration_count) <= 0 or int(target_count) <= 0:
        raise ValueError("calibration_count and target_count must be positive")
    if not 0.0 <= float(discount) <= 1.0:
        raise ValueError("discount must lie in [0, 1]")

    strength = float(discount) * int(calibration_count)
    alpha = 1.0 + strength * accuracy
    beta = 1.0 + strength * (1.0 - accuracy)
    return {
        "alpha": alpha,
        "beta": beta,
        "discount": float(discount),
        "calibration_count": int(calibration_count),
        "added_strength": strength,
        "beta_effective_sample_size": beta_effective_sample_size(alpha, beta),
        "map_prior_weight": strength / (int(target_count) + strength),
    }


def bound_feasibility(normalized_coverage, effective_sample_size, lipschitz,
                      delta=0.05, tolerance=1.0):
    """Check the exact coverage/ESS condition for a requested error tolerance."""
    coverage = float(normalized_coverage)
    size = float(effective_sample_size)
    constant = float(lipschitz)
    tolerance = float(tolerance)
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("normalized_coverage must lie in [0, 1]")
    if size <= 0.0 or constant < 0.0:
        raise ValueError("effective_sample_size must be positive and lipschitz non-negative")
    if not 0.0 < float(delta) < 1.0 or not 0.0 < tolerance <= 1.0:
        raise ValueError("delta and tolerance must lie in (0, 1]")

    transfer = 2.0 * constant * np.sqrt(1.0 - coverage)
    if transfer >= tolerance:
        required = np.inf
    else:
        required = (np.log(2.0 / float(delta))
                    / (2.0 * (tolerance - transfer) ** 2))
    bound = transfer + np.sqrt(np.log(2.0 / float(delta)) / (2.0 * size))
    return {
        "bound": float(bound),
        "transfer_bound": float(transfer),
        "required_effective_sample_size": float(required),
        "feasible": bool(size >= required),
    }


def _unit_rows(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"{name} contains a zero vector")
    return array / norms


def _similarity_matrix(values):
    similarity = np.asarray(values, dtype=float)
    if similarity.ndim != 2 or similarity.shape[0] == 0 or similarity.shape[1] == 0:
        raise ValueError("similarity must be a non-empty two-dimensional array")
    if np.any(~np.isfinite(similarity)) or np.any(similarity < -1e-12) \
            or np.any(similarity > 1.0 + 1e-12):
        raise ValueError("similarity must contain finite values in [0, 1]")
    return np.clip(similarity, 0.0, 1.0)
