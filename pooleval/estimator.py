"""The three-stage PoolEvaluator estimator from Algorithm 1 of the paper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np


EPS = 1.0e-6


def _clip(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def _as_responses(responses: Sequence[Sequence[Any]]) -> np.ndarray:
    array = np.asarray(responses, dtype=object)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("responses must have shape [items, models] with at least two models")
    return array


def _weighted_mode(values: np.ndarray, weights: np.ndarray) -> Any:
    """Mode by vote count; summed initial accuracy breaks count ties (paper Eq. 4)."""
    counts: dict[Any, int] = {}
    support: dict[Any, float] = {}
    first: dict[Any, int] = {}
    for index, (value, weight) in enumerate(zip(values.tolist(), weights.tolist())):
        try:
            key = value
            hash(key)
        except (TypeError, ValueError):
            key = repr(value)
        counts[key] = counts.get(key, 0) + 1
        support[key] = support.get(key, 0.0) + float(weight)
        first.setdefault(key, index)
    winner = max(counts, key=lambda key: (counts[key], support[key], -first[key]))
    return values[first[winner]]


def leave_one_out_agreement(responses: Sequence[Sequence[Any]], weights: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(C, consensus)`` from paper Eq. 4--5.

    ``C[i,j]`` is one when model ``j`` agrees with the consensus of every *other*
    model on item ``i``.  Accuracy priors break ties, as specified in the paper.
    """
    r = _as_responses(responses)
    n_items, n_models = r.shape
    w = np.ones(n_models, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n_models,):
        raise ValueError(f"weights must have shape {(n_models,)}")
    agreement = np.zeros((n_items, n_models), dtype=float)
    consensus = np.empty((n_items, n_models), dtype=object)
    all_models = np.arange(n_models)
    for i in range(n_items):
        for j in range(n_models):
            keep = all_models != j
            answer = _weighted_mode(r[i, keep], w[keep])
            consensus[i, j] = answer
            agreement[i, j] = float(r[i, j] == answer)
    return agreement, consensus


def calibration_parameters(
    responses: Sequence[Sequence[Any]],
    gold_answers: Sequence[Any],
    smoothing: float = 1.0,
    subset_ids: Sequence[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize alpha, beta, gamma from a labeled retrieved meta-dataset (Eq. 12)."""
    r = _as_responses(responses)
    gold = np.asarray(gold_answers, dtype=object)
    if gold.shape != (r.shape[0],):
        raise ValueError("gold_answers must contain one answer per item")
    correct = r == gold[:, None]
    if subset_ids is None:
        alpha = correct.mean(axis=0)
    else:
        groups = np.asarray(subset_ids, dtype=object)
        if groups.shape != (r.shape[0],):
            raise ValueError("subset_ids must contain one identifier per calibration item")
        unique = list(dict.fromkeys(groups.tolist()))
        alpha = np.mean([correct[groups == group].mean(axis=0) for group in unique], axis=0)
    agreement, _ = leave_one_out_agreement(r, alpha)
    correct_n = correct.sum(axis=0)
    wrong_n = (~correct).sum(axis=0)
    beta = ((agreement * correct).sum(axis=0) + smoothing) / (correct_n + 2.0 * smoothing)
    gamma = ((agreement * ~correct).sum(axis=0) + smoothing) / (wrong_n + 2.0 * smoothing)
    return _clip(alpha), _clip(beta), _clip(gamma)


@dataclass
class Estimate:
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    posterior: np.ndarray
    agreement: np.ndarray
    ranking: np.ndarray
    iterations: int
    converged: bool
    validated: dict[int, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)


class PoolEvaluator:
    """Closed-form EM with optional exact judge validation (paper Eq. 13--19)."""

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        smoothing: float = 1.0,
        seed: int = 0,
    ):
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.smoothing = float(smoothing)
        self.seed = int(seed)

    @staticmethod
    def _e_step(c: np.ndarray, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        like_correct = alpha * np.where(c == 1, beta, 1.0 - beta)
        like_wrong = (1.0 - alpha) * np.where(c == 1, gamma, 1.0 - gamma)
        return _clip(like_correct / np.maximum(like_correct + like_wrong, EPS))

    def fit(
        self,
        responses: Sequence[Sequence[Any]],
        initial: tuple[Sequence[float], Sequence[float], Sequence[float]] | None = None,
        validated: Mapping[int, Any] | None = None,
    ) -> Estimate:
        r = _as_responses(responses)
        n_items, n_models = r.shape
        if initial is None:
            rng = np.random.default_rng(self.seed)
            alpha = rng.uniform(EPS, 1.0 - EPS, size=n_models)
            beta = rng.uniform(EPS, 1.0 - EPS, size=n_models)
            gamma = rng.uniform(EPS, 1.0 - EPS, size=n_models)
        else:
            alpha, beta, gamma = map(_clip, initial)
            if any(x.shape != (n_models,) for x in (alpha, beta, gamma)):
                raise ValueError("each initial parameter must contain one value per model")
        judged = dict(validated or {})
        agreement, _ = leave_one_out_agreement(r, alpha)
        history: list[dict[str, Any]] = []
        converged = False
        q = np.empty_like(agreement)
        for iteration in range(1, self.max_iterations + 1):
            old = np.concatenate([alpha, beta, gamma])
            q = self._e_step(agreement, alpha, beta, gamma)
            for item, answer in judged.items():
                if not 0 <= item < n_items:
                    raise IndexError(f"validated item {item} is outside [0, {n_items})")
                q[item, :] = (r[item, :] == answer).astype(float)

            alpha = _clip(q.mean(axis=0))
            beta = _clip((q * agreement).sum(axis=0) / np.maximum(q.sum(axis=0), EPS))
            wrong = 1.0 - q
            gamma = _clip(
                (wrong * agreement).sum(axis=0) / np.maximum(wrong.sum(axis=0), EPS)
            )
            delta = float(np.max(np.abs(np.concatenate([alpha, beta, gamma]) - old)))
            history.append({"iteration": iteration, "delta": delta})
            if delta < self.tolerance:
                converged = True
                break
        q = self._e_step(agreement, alpha, beta, gamma)
        for item, answer in judged.items():
            q[item, :] = (r[item, :] == answer).astype(float)
        return Estimate(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            posterior=q,
            agreement=agreement,
            ranking=np.argsort(-alpha),
            iterations=iteration,
            converged=converged,
            validated=judged,
            history=history,
        )

    @staticmethod
    def uncertainty(posterior: np.ndarray) -> np.ndarray:
        p = _clip(posterior)
        return (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))).sum(axis=1)

    def information_gain(
        self,
        responses: Sequence[Sequence[Any]],
        estimate: Estimate,
        validated: Mapping[int, Any] | None = None,
    ) -> np.ndarray:
        """Exact judge look-ahead from paper Eq. 15--17.

        For each unresolved item, enumerate every distinct pool answer plus NONE,
        normalize their feasible correctness-vector probabilities, warm-start a
        hypothetical EM fit, and measure expected remaining posterior entropy.
        """
        r = _as_responses(responses)
        judged = dict(validated or estimate.validated)
        unresolved = [i for i in range(r.shape[0]) if i not in judged]
        scores = np.full(r.shape[0], -np.inf, dtype=float)
        if not unresolved:
            return scores
        current_entropy = float(self.uncertainty(estimate.posterior[unresolved]).sum())
        initial = (estimate.alpha, estimate.beta, estimate.gamma)
        for item in unresolved:
            answers = list(dict.fromkeys(r[item].tolist()))
            none_answer = object()
            outcomes: list[tuple[Any, np.ndarray]] = [
                (answer, (r[item] == answer).astype(float)) for answer in answers
            ]
            outcomes.append((none_answer, np.zeros(r.shape[1], dtype=float)))
            log_mass: list[float] = []
            for _, z in outcomes:
                q = _clip(estimate.posterior[item])
                log_mass.append(float((z * np.log(q) + (1.0 - z) * np.log(1.0 - q)).sum()))
            mass = np.exp(np.asarray(log_mass) - max(log_mass))
            probability = mass / mass.sum()
            remaining = [i for i in unresolved if i != item]
            expected_entropy = 0.0
            for (answer, _), weight in zip(outcomes, probability):
                hypothetical = dict(judged)
                hypothetical[item] = answer
                fitted = self.fit(r, initial, hypothetical)
                entropy = float(self.uncertainty(fitted.posterior[remaining]).sum()) if remaining else 0.0
                expected_entropy += float(weight) * entropy
            scores[item] = current_entropy - expected_entropy
        return scores

    def refine(
        self,
        responses: Sequence[Sequence[Any]],
        initial: tuple[Sequence[float], Sequence[float], Sequence[float]],
        judge: Callable[[int, list[Any]], Any],
        rounds: int = 10,
        batch_size: int = 1,
    ) -> Estimate:
        """Select ambiguous items, call a judge, and warm-start EM after each round."""
        r = _as_responses(responses)
        validated: dict[int, Any] = {}
        current_initial = initial
        estimate = self.fit(r, current_initial, validated)
        judge_history: list[dict[str, Any]] = []
        for round_index in range(int(rounds)):
            selected = 0
            for _ in range(max(1, int(batch_size))):
                gains = self.information_gain(r, estimate, validated)
                item = int(np.argmax(gains))
                if not np.isfinite(gains[item]) or gains[item] <= 0.0:
                    break
                unique = list(dict.fromkeys(r[item].tolist()))
                answer = judge(int(item), unique)
                validated[int(item)] = answer
                judge_history.append(
                    {
                        "round": round_index + 1,
                        "item": int(item),
                        "information_gain": float(gains[item]),
                        "answer": answer,
                    }
                )
                current_initial = (estimate.alpha, estimate.beta, estimate.gamma)
                estimate = self.fit(r, current_initial, validated)
                selected += 1
            if not selected:
                break
        estimate.history = judge_history + estimate.history
        return estimate
