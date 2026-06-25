"""B3 -- Dawid--Skene (1979): the classic latent-truth crowd model with NO prior
and errors assumed independent. Good on relative order but, lacking an anchor, its
absolute level is unidentifiable (gauge ambiguity) and a correlated pool inflates it.
Implemented as PoolEval's EM with the anchor/verifier/correlation components off.
"""
import copy
from .._common import obs_at
from pooleval.latent import run_em


class DawidSkene:
    name = "B3 Dawid--Skene"

    def evaluate(self, run, cfg):
        c = copy.copy(cfg)
        c.use_prior = False
        c.use_verifier = False
        c.use_correlation = False
        c.fusion = "agreement_only"
        obs = obs_at(run, c, level="L1")
        out = run_em(obs, run, c)
        return out["acc"]
