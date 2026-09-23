"""Baseline estimators used in the PoolEvaluator experiments."""

from .agreement_line import AgreementLine
from .atc import AverageThresholdedConfidence
from .dawid_skene import DawidSkene
from .doc import DifferenceOfConfidence
from .gde import GeneralizedDisagreementEquality
from .independent import Independent
from .llm_judge import LLMJudge
from .majority import Majority


def baseline_registry() -> dict[str, type]:
    """Return the available baseline classes without constructing task inputs."""
    return {
        "independent": Independent,
        "majority": Majority,
        "dawid_skene": DawidSkene,
        "agreement_line": AgreementLine,
        "llm_judge": LLMJudge,
        "doc": DifferenceOfConfidence,
        "atc": AverageThresholdedConfidence,
        "gde": GeneralizedDisagreementEquality,
    }


__all__ = [
    "AgreementLine",
    "AverageThresholdedConfidence",
    "DawidSkene",
    "DifferenceOfConfidence",
    "GeneralizedDisagreementEquality",
    "Independent",
    "LLMJudge",
    "Majority",
    "baseline_registry",
]
