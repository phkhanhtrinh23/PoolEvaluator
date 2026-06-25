"""Baseline registry for pool evaluation (one folder per method).

B1 Independent (Garg'22), B2 Majority/self-consistency (Wang'23),
B3 Dawid--Skene ('79), B4 Agreement-on-the-line (Baek'22).
"""
from .independent import Independent
from .majority import Majority
from .dawid_skene import DawidSkene
from .agreement_line import AgreementLine


def all_baselines():
    return [Independent(), Majority(), DawidSkene(), AgreementLine()]


__all__ = ["Independent", "Majority", "DawidSkene", "AgreementLine",
           "all_baselines"]
