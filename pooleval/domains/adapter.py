"""Build a PoolRun from REAL model predictions in a closed-label-space task.

PoolEval's EM (`pooleval.latent.run_em`) only ever tests observations for
EQUALITY, so any task whose models emit a discrete answer per item can be fed to
it unchanged. For Text2SQL the answer is a result-set equivalence class produced
by the graded kernel; for image / node classification it is simply the predicted
label, and the kernel degenerates to the identity (exact match, precision =
recall = 1). Set `cfg.real_data = True` so `kernel.apply` passes observations
through.

CLASS ENCODING. The repo convention is `true_class[m, i] == 0  <=>  model m is
correct on item i`. For a closed label space we preserve the *partition* exactly
by mapping

    true_class[m, i] = 0            if pred[m, i] == gold[i]
                     = 1 + pred[m, i]   otherwise

Two models that emit the same wrong label collide on the same wrong class (which
is what makes shared error visible to the estimator), no wrong label can alias
class 0, and the mapping is a bijection on each item's induced partition -- so
the estimator sees exactly the agreement structure of the raw predictions.

NOTE ON A STRUCTURAL DIFFERENCE FROM TEXT2SQL. Here the candidate set
`np.unique(obs[:, i])` is drawn from a *global, closed* label space of K classes,
so the correct answer is at worst one of K options and an external channel
(verifier / judge) can always name it. In Text2SQL the candidate set is the set
of result tables the pool happened to produce, and the truth can be absent from
it entirely -- the candidate-coverage failure. Porting to classification removes
that failure mode; it does NOT remove the gauge trap (all models wrong on the
same label).
"""
import numpy as np

from ..data.simulator import PoolRun


def encode_classes(pred, gold):
    """[M,N] predicted labels + [N] gold -> repo-convention class ids (0 = correct)."""
    pred = np.asarray(pred, dtype=np.int64)
    gold = np.asarray(gold, dtype=np.int64)[None, :]
    return np.where(pred == gold, 0, 1 + pred).astype(np.int64)


def from_predictions(pred, gold, group, prior=None, prior_sigma=None,
                     verifier_guess=None, n_classes=None, prior_noise=0.07):
    """Assemble a PoolRun the estimator and every baseline can consume.

    pred            [M,N] int   predicted label of each model on each target item
    gold            [N]   int   ground truth -- WITHHELD from the estimator, used
                                only to score it afterwards
    group           [M]   int   provenance group (architecture family / shared backbone)
    prior           [M]   float source-domain accuracy ("seen prior"); None -> no prior
    verifier_guess  [N]   int   label an independent channel points at; None -> none
    """
    pred = np.asarray(pred, dtype=np.int64)
    gold = np.asarray(gold, dtype=np.int64)
    M, N = pred.shape
    group = np.asarray(group, dtype=np.int64)
    G = int(group.max()) + 1

    true_class = encode_classes(pred, gold)
    true_acc = (true_class == 0).mean(axis=1)

    if prior is None:                       # uninformative anchor; use_prior=False
        prior = np.full(M, 0.5)
        prior_sigma = np.full(M, 1.0)
    else:
        prior = np.clip(np.asarray(prior, dtype=float), 0.01, 0.99)
        prior_sigma = (np.full(M, max(prior_noise, 1e-3)) if prior_sigma is None
                       else np.asarray(prior_sigma, dtype=float))

    if verifier_guess is None:
        vg = np.full(N, -1, dtype=np.int64)          # never matches a candidate
    else:
        vg = encode_classes(np.asarray(verifier_guess)[None, :], gold)[0]
    verifier_correct = vg == 0

    # b / phi are diagnostics the estimator re-fits; seed them from the data.
    item_agree = (true_class == 0).mean(axis=0)
    return PoolRun(true_class=true_class, true_acc=true_acc, group=group,
                   prior=prior, prior_sigma=prior_sigma,
                   b=-np.log(np.clip(item_agree, .05, .95) /
                             (1 - np.clip(item_agree, .05, .95))) * 0.5,
                   phi=np.full(N, 1.0),
                   verifier_guess=vg, verifier_correct=verifier_correct,
                   M=M, N=N, n_groups=G)


pool_run_from_predictions = from_predictions      # alias
