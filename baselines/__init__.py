"""Baseline registry for pool evaluation (one folder per method).

B1 Independent (Garg'22), B2 Majority/self-consistency (Wang'23),
B3 Dawid--Skene ('79), B4 Agreement-on-the-line (Baek'22),
B5 LLM-as-judge (paper proxy in the simulator).
"""
from .independent import Independent
from .majority import Majority
from .dawid_skene import DawidSkene
from .agreement_line import AgreementLine
from .llm_judge import LLMJudge


def all_baselines():
    return [Independent(), Majority(), DawidSkene(), AgreementLine(), LLMJudge()]


__all__ = ["Independent", "Majority", "DawidSkene", "AgreementLine", "LLMJudge",
           "all_baselines"]
