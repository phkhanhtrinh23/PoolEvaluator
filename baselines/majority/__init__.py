"""B2 -- Majority / self-consistency (Wang et al., 2023).

Each model's accuracy is read off its agreement with the per-item execution-majority
answer. It has no notion of latent difficulty or correlated voters, so a colluding
within-group majority inflates the estimate.
"""
from .._common import obs_at, majority_accuracy


class Majority:
    name = "B2 Majority/self-cons."

    def evaluate(self, run, cfg):
        obs = obs_at(run, cfg, level="LA1")
        acc, _ = majority_accuracy(obs)
        return acc
