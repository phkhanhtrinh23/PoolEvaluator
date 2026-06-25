"""Synthetic Text-to-SQL pool simulator.

The paper evaluates real model zoos on Spider/BIRD/Spider 2.0; those checkpoints
and databases are not redistributable, so this module simulates the *evaluation
problem* PoolEval is designed for, faithfully reproducing the phenomena the method
exploits or guards against:

  * Latent correctness: each item i has a latent correct result class (0) and a
    difficulty b_i; model m answers correctly w.p. sigma(s*(a_m - b_i)) (IRT).
  * Provenance-correlated errors: models share a pretrained base in `n_groups`
    groups; when a within-group model errs, with prob ~collusion*phi_i it emits the
    GROUP's shared wrong answer -> near-clones agree on wrong results (false
    agreement that fools independence-assuming estimators).
  * Seen priors: a single-model label-free calibrator gives each model a noisy,
    shift-biased prior accuracy (the per-model evaluator's output).
  * Execution verifier: an independent channel that points at the true class with
    prob `verifier_acc` (errors uncorrelated with the model pool).

`true_class[m,i] == 0` means model m is correct on item i (gold withheld from the
estimator; used only to score it).
"""
from dataclasses import dataclass
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class PoolRun:
    true_class: np.ndarray    # [M,N] int; 0 = correct, >0 = wrong class id
    true_acc: np.ndarray      # [M] empirical accuracy (fraction correct)
    group: np.ndarray         # [M] provenance group id
    prior: np.ndarray         # [M] seen single-model prior accuracy
    prior_sigma: np.ndarray   # [M] prior uncertainty
    b: np.ndarray             # [N] item difficulty
    phi: np.ndarray           # [N] shared-error exposure
    verifier_guess: np.ndarray  # [N] class the verifier points at
    verifier_correct: np.ndarray  # [N] bool: verifier guess == true class (0)
    M: int
    N: int
    n_groups: int


def default_groups(M, G, n_big=3):
    """Paper-like skewed provenance: a few LARGE near-clone groups + singletons.
    For M=12, G=7 this yields sizes [3,3,2,1,1,1,1] (3 Qwen, 3 GPT, 2 DeepSeek,
    and one each of StarCoder2/CodeLlama/Llama-3/Gemini), so the true independent
    count is ~G < M -- the regime where correlated errors fool naive majority."""
    sizes = [1] * G
    extra = max(0, M - G)
    g = 0
    while extra > 0:
        sizes[g % min(n_big, G)] += 1
        g += 1
        extra -= 1
    group = []
    for gi, s in enumerate(sizes):
        group += [gi] * s
    return np.array(group[:M])


def simulate(cfg, group_assignment=None, acc=None, rng=None) -> PoolRun:
    rng = rng or np.random.default_rng(cfg.seed)
    M, N, G = cfg.M, cfg.N, cfg.n_groups

    # provenance group per model (paper-like skewed default unless given)
    if group_assignment is None:
        group = default_groups(M, G)
    else:
        group = np.asarray(group_assignment)[:M]
        G = int(group.max()) + 1

    # true accuracies (abilities), spread so ranking is non-trivial
    if acc is None:
        acc = rng.uniform(cfg.acc_lo, cfg.acc_hi, size=M)
    a_logit = np.log(acc / (1 - acc))

    b = rng.normal(0, cfg.diff_std, size=N)            # item difficulty
    phi = rng.uniform(0.2, 1.0, size=N)                # shared-error exposure
    # group shared wrong answer id per (group, item)
    shared_wrong = 1000 + np.arange(G)[:, None] * N + np.arange(N)[None, :]
    # universal (cross-pool) shared wrong answer per item -- the gauge trap
    universal_wrong = 800000 + np.arange(N)
    # gauge-trap items: where the universal error is active AND can fool the verifier
    gauge_item = rng.random(N) < cfg.universal_error

    true_class = np.zeros((M, N), dtype=np.int64)
    idio = 100000
    for m in range(M):
        g = group[m]
        # IRT: P(correct on i) = sigma(ability_m - difficulty_i)
        p = sigmoid(a_logit[m] - b)
        u = rng.random(N)
        correct = u < p
        for i in range(N):
            if correct[i]:
                true_class[m, i] = 0
            else:
                r = rng.random()
                if gauge_item[i] and r < 0.85:
                    true_class[m, i] = universal_wrong[i]      # cross-pool shared (gauge)
                elif r < cfg.collusion * phi[i]:
                    true_class[m, i] = shared_wrong[g, i]      # group-shared wrong
                else:
                    true_class[m, i] = idio + rng.integers(cfg.n_wrong_classes)
                    idio += cfg.n_wrong_classes
    true_acc = (true_class == 0).mean(axis=1)

    # seen prior: shift-biased + noisy single-model calibration
    bias = rng.normal(0, cfg.prior_bias, size=M)
    prior = np.clip(true_acc + bias + rng.normal(0, cfg.prior_noise, size=M), 0.01, 0.99)
    prior_sigma = np.full(M, max(cfg.prior_noise, 1e-3))

    # execution verifier: independent channel pointing at the true class (0). On
    # gauge-trap items the plausible-but-wrong universal answer can fool it too, so
    # the verifier alone cannot pin the absolute level -- the seen prior is needed.
    vc = rng.random(N) < cfg.verifier_acc
    verifier_guess = np.where(vc, 0, rng.integers(1, 1000, size=N))
    fooled = gauge_item & (rng.random(N) < 0.5) & (~vc.astype(bool) | (rng.random(N) < 0.3))
    verifier_guess = np.where(fooled, universal_wrong, verifier_guess)
    vc = verifier_guess == 0
    return PoolRun(true_class, true_acc, group, prior, prior_sigma, b, phi,
                   verifier_guess, vc, M, N, G)
