"""B1 -- Independent single-model label-free evaluation (Garg et al., 2022).

The parallel-but-separate strategy a pool operator reaches for first: each model's
accuracy is its own single-model label-free calibration (the seen prior). It never
sees cross-model agreement, so its ranking variance does not shrink with the pool.
"""


class Independent:
    name = "B1 Independent"

    def evaluate(self, run, cfg):
        return run.prior.copy()
