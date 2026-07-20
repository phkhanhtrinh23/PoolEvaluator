"""Configuration for PoolEval-SQL (label-free joint pool evaluation for Text-to-SQL).

Defaults mirror the paper: a diverse M=12 pool over seven provenance groups,
graded kernel LA0-LA2, IRT-style provenance-grouped errors, seen-prior + verifier
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

    # ---- LLM-as-judge baseline (B5): a PREFERENCE judge (no execution) ----
    judge_hit: float = 0.90        # P(judge labels a truly-correct answer correct)
    judge_fa: float = 0.18         # P(judge is fooled by a plausible-but-wrong query)
    judge_style_bias: float = 0.06 # per-model judge style/verbosity bias std (mis-ranking)

    # ---- graded equivalence kernel (simulated precision/recall vs gold) ----
    recall_LA0: float = 0.74   # LA0 exact match: misses canonical-equal results
    recall_LA1: float = 0.95   # LA1 canonicalized
    recall_LA2: float = 0.97   # LA2 multi-instance
    precision_LA0: float = 0.98
    precision_LA1: float = 0.95
    precision_LA2: float = 0.99

    # ---- inference ----
    em_iters: int = 40
    fusion: str = "precision"   # precision | prior_only | agreement_only
    use_prior: bool = True
    use_verifier: bool = True
    use_correlation: bool = True
    kernel_level: str = "LA2"   # LA0 | LA1 | LA2  (LA0 = exact-match ablation)

    # ---- real-data mode ----
    # When True, PoolRun.true_class already holds REAL result-equivalence classes
    # (from executing model SQL on the live DB, not the synthetic simulator), so the
    # kernel is the identity: estimator and baselines consume the observed classes
    # directly instead of re-perturbing them. Default False keeps simulator behavior.
    real_data: bool = False

    def to_dict(self):
        return asdict(self)
