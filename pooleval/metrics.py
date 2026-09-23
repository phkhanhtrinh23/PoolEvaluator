from __future__ import annotations

import numpy as np


def evaluate(estimated: np.ndarray, truth: np.ndarray, top_k: int = 5) -> dict[str, float]:
    estimated = np.asarray(estimated, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimated.shape != truth.shape:
        raise ValueError("estimated and truth must have the same shape")
    mae = float(np.mean(np.abs(estimated - truth)))
    concordant = discordant = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            sign_true = np.sign(truth[i] - truth[j])
            sign_est = np.sign(estimated[i] - estimated[j])
            if not sign_true or not sign_est:
                continue
            concordant += sign_true == sign_est
            discordant += sign_true != sign_est
    pairs = concordant + discordant
    kendall = float((concordant - discordant) / pairs) if pairs else 0.0
    best = int(np.argmax(truth))
    ranking = np.argsort(-estimated)
    return {
        "mae": mae,
        "kendall_tau": kendall,
        "pairwise_accuracy": float(concordant / pairs) if pairs else 0.0,
        "top_1": float(ranking[0] == best),
        f"top_{top_k}": float(best in ranking[: min(top_k, len(ranking))]),
    }
