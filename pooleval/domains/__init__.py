"""Domain ports of PoolEval: the estimator core is task-agnostic, only the
*observation kernel*, the *prior*, and the *verifier* are domain-specific.

  text2sql : obs = execution-result equivalence class   (graded LA0/LA1/LA2 kernel)
  vision   : obs = predicted class label                (exact match -- kernel is identity)
  graph    : obs = predicted node label                 (exact match -- kernel is identity)
"""
from .adapter import from_predictions, pool_run_from_predictions
from .collision import collision_estimates, source_gamma, gamma_from_run   # noqa: F401
