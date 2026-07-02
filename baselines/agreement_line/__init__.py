"""B4 -- Agreement-on-the-line (Baek et al., 2022): the agreement--accuracy linear
trend, fit across the pool as a tuned agreement baseline. We regress each model's
mean pairwise agreement rate onto the seen-prior accuracies and predict from the
fitted line. A single shared line cannot model provenance-correlated agreement.
"""
import numpy as np
from .._common import obs_at


class AgreementLine:
    name = "B4 Agreement-on-line"

    def evaluate(self, run, cfg):
        obs = obs_at(run, cfg, level="LA1")
        M, N = obs.shape
        r = np.zeros(M)
        for m in range(M):
            agree = 0.0
            for mp in range(M):
                if mp == m:
                    continue
                agree += np.mean(obs[m] == obs[mp])
            r[m] = agree / (M - 1)
        # fit prior ~ slope*r + intercept (least squares), predict from the line
        A = np.vstack([r, np.ones(M)]).T
        slope, intercept = np.linalg.lstsq(A, run.prior, rcond=None)[0]
        return np.clip(slope * r + intercept, 0.02, 0.98)
