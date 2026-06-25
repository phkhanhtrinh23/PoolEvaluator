"""PoolEval estimator: kernel -> correlation-aware latent EM -> outputs.

Returns per-model accuracy, the ranking with pairwise win-probabilities,
conformal-style 90% intervals widened by the measured kernel-equivalence error,
the effective-independent-model count M_eff, and a collusion flag.
"""
import numpy as np
from . import kernel as kernelmod
from .latent import run_em


class PoolEval:
    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(self, run, verbose=False):
        cfg = self.cfg
        # graded equivalence kernel (L0 exact / L1 canonical / L2 multi-instance)
        obs = kernelmod.apply(run.true_class, cfg, level=cfg.kernel_level,
                              rng=np.random.default_rng(cfg.seed + 7))
        prec, rec = kernelmod.measure(run.true_class, obs)
        out = run_em(obs, run, cfg, verbose=verbose)

        acc = out["acc"]
        order = np.argsort(-acc)                       # best first
        # interval half-width: posterior sigma widened by kernel error (1-precision)
        z = 1.645                                      # 90%
        half = z * out["a_sigma"] + 0.5 * (1 - prec)
        out.update(ranking=order, kernel_precision=prec, kernel_recall=rec,
                   lo=np.clip(acc - half, 0, 1), hi=np.clip(acc + half, 0, 1),
                   obs=obs)
        return out

    def win_prob(self, out, a, b):
        """P(acc_a > acc_b) from posterior (normal approx)."""
        from scipy.stats import norm
        d = out["acc"][a] - out["acc"][b]
        s = np.sqrt(out["a_sigma"][a] ** 2 + out["a_sigma"][b] ** 2) + 1e-9
        return float(norm.cdf(d / s))
