"""PoolEval-SQL: label-free joint evaluation and ranking of a Text-to-SQL model
pool via anchored, correlation-aware latent-correctness inference.

The paper names the framework PoolEval-SQL; this package is imported as `pooleval`
and its estimator is the `PoolEval` class."""
from .config import Config
from .inference import PoolEval
from .new_formulation import (NewFormulationPoolEval, agreement_em,
                              CollisionAwareNewFormulationPoolEval,
                              collision_agreement_em)
from .prior_bounds import (beta_effective_sample_size,
                           beta_power_prior_diagnostics, bound_feasibility,
                           greedy_dataset_coverage, matching_diagnostics,
                           optimal_dataset_coverage,
                           prior_accuracy_diagnostics,
                           shifted_cosine_similarity)
from .validated_em import (LabeledStatistics, correctness_em, JudgeExpert,
                          NoisyExpert, OracleExpert,
                          excess_collision, gamma_counts, gamma_from_counts,
                          information_gain, latent_posterior, run_validation,
                          set_entropy, validated_em, vote_discount,
                          wrong_collision_matrix)
from .active import ActivePoolEval, ActiveConfig, SimulatedJudge
from .data import simulate, PoolRun
from . import metrics, kernel, theory, validated_em, partitions, answer_matching

__all__ = ["Config", "PoolEval", "NewFormulationPoolEval", "agreement_em",
           "CollisionAwareNewFormulationPoolEval", "collision_agreement_em",
           "shifted_cosine_similarity", "greedy_dataset_coverage",
           "optimal_dataset_coverage", "matching_diagnostics",
           "prior_accuracy_diagnostics", "beta_power_prior_diagnostics",
           "beta_effective_sample_size",
           "bound_feasibility",
           "ActivePoolEval", "ActiveConfig", "SimulatedJudge", "simulate", "PoolRun",
           "metrics", "kernel", "theory", "validated_em", "partitions",
           "answer_matching",
           "LabeledStatistics", "OracleExpert", "NoisyExpert",
           "wrong_collision_matrix", "excess_collision", "gamma_counts",
           "gamma_from_counts", "vote_discount", "latent_posterior",
           "set_entropy", "validated_em", "correctness_em", "information_gain", "JudgeExpert",
           "run_validation"]
__version__ = "0.1.0"
