"""Configuration for PoolEval (label-free joint pool evaluation for Text-to-SQL).

Defaults mirror the paper: a diverse M=12 pool over seven provenance groups,
graded kernel L0-L2, IRT-style provenance-grouped errors, seen-prior + verifier
anchors, inverse-variance fusion, EM to convergence, 90% conformal intervals.
"""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- pool / workload ----
    M: int = 12               # pool size (models)
    N: int = 400              # unlabeled items (questions)
    n_groups: int = 7         # provenance groups (shared pretrained base)
    seed: int = 0

    # ---- latent-correctness generative model ----
    acc_lo: float = 0.30      # true model accuracy range (moderate-shift regime)
    acc_hi: float = 0.70
    diff_std: float = 1.6     # item difficulty b_i ~ N(0, diff_std)
    irt_scale: float = 4.0    # logistic scale (kept for reference)

    # ---- correlated (provenance) errors ----
    collusion: float = 0.55   # prob a within-group error is the GROUP-shared wrong answer
    universal_error: float = 0.30  # prob an error is a CROSS-pool shared wrong answer
                                   # (the gauge trap: "everyone makes the same mistake"
                                   #  looks like consensus -> only prior/verifier fix it)
    n_wrong_classes: int = 6  # idiosyncratic wrong-answer classes

    # ---- anchors ----
    prior_bias: float = 0.06       # shift-induced bias of the seen prior (signed per model)
    prior_noise: float = 0.07      # seen-prior noise std
    verifier_acc: float = 0.72     # P(verifier points at the true class) -- independent channel
    verifier_strength: float = 2.0 # log-bonus the verifier adds to its guessed class

    # ---- graded equivalence kernel (simulated precision/recall vs gold) ----
    recall_L0: float = 0.74   # exact match: misses canonical-equal results
    recall_L1: float = 0.95   # canonicalized
    recall_L2: float = 0.97   # multi-instance
    precision_L0: float = 0.98
    precision_L1: float = 0.95
    precision_L2: float = 0.99

    # ---- inference ----
    em_iters: int = 40
    fusion: str = "precision"   # precision | prior_only | agreement_only
    use_prior: bool = True
    use_verifier: bool = True
    use_correlation: bool = True
    kernel_level: str = "L2"    # L0 | L1 | L2  (L0 = exact-match ablation)

    def to_dict(self):
        return asdict(self)
