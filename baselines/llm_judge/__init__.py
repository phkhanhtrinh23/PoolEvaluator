"""B5 -- LLM-as-judge (paper proxy).

The paper's judge baseline asks a strong LLM to score each candidate query's
correctness against the question and schema -- a PREFERENCE judgment, without
executing the query -- and reads each model's accuracy off its mean judged-correct
rate. We simulate that signal directly: the judge labels a truly-correct answer
correct w.p. `judge_hit`, and is fooled into accepting a plausible-but-wrong query
w.p. `judge_fa` (it never executes, so it cannot catch a query that looks right but
returns the wrong rows). A small per-model style bias (verbosity/format preference)
makes its ranking imperfect. Being preference- rather than execution-grounded, it
trails the execution-grounded estimators -- the comparison the paper draws ("does
execution-grounded agreement beat preference judging?").
"""
import numpy as np


class LLMJudge:
    name = "B5 LLM-as-judge"

    def evaluate(self, run, cfg):
        rng = np.random.default_rng(cfg.seed + 313)
        correct = (run.true_class == 0)                 # [M,N] latent correctness
        M, N = correct.shape
        # per-model style bias: the judge systematically over/under-rates some models
        # (verbosity/format preference), the source of preference-based mis-ranking.
        style = rng.normal(0.0, cfg.judge_style_bias, size=M)
        p_hit = np.clip(cfg.judge_hit + style, 0.5, 0.99)[:, None]
        p_fa = np.clip(cfg.judge_fa + style, 0.02, 0.6)[:, None]
        u = rng.random((M, N))
        judged = np.where(correct, u < p_hit, u < p_fa)  # per-answer accept/reject
        return judged.mean(axis=1)                       # mean judged-correct rate
