"""B5 -- LLM-as-judge (paper proxy).

The paper's judge baseline asks a strong model to score each candidate answer
against the question and schema. In this simulator we approximate that signal
with the independent execution verifier plus extra judge noise so it stays a
strong-but-imperfect single-model baseline instead of collapsing into an oracle.
"""
import numpy as np

from .._common import obs_at


class LLMJudge:
    name = "B5 LLM-as-judge"

    def evaluate(self, run, cfg):
        obs = obs_at(run, cfg, level="L1")
        rng = np.random.default_rng(cfg.seed + 313)
        judge = run.verifier_guess.copy()
        noisy = rng.random(len(judge)) < 0.22
        if np.any(noisy):
            idx = np.where(noisy)[0]
            to_wrong = judge[idx] == 0
            if np.any(to_wrong):
                judge[idx[to_wrong]] = rng.integers(1, 1000, size=int(to_wrong.sum()))
            if np.any(~to_wrong):
                judge[idx[~to_wrong]] = 0
        return (obs == judge[None, :]).mean(axis=1)
