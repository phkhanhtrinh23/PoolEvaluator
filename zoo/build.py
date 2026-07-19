"""Execute the zoo's predictions and assemble a real PoolRun for the estimator.

Pipeline: predicted SQL --exec--> canonical result key --compare-to-gold-->
equivalence class -> true_class[M,N]; source-split accuracy -> seen prior;
execution-plausibility -> verifier. The resulting PoolRun plugs into the SHIPPED
estimator/baselines unchanged (Config.real_data=True makes the kernel the identity).
"""
import numpy as np

from pooleval.data.simulator import PoolRun
from .execute import result_key, assign_classes, is_nonempty


def _exec_item(item, member_names, preds, timeout):
    """For one item: gold key + per-model (key, class)."""
    gold_key, _ = result_key(item["db_path"], item["gold_sql"], timeout)
    pred_keys = []
    for name in member_names:
        sql = preds[name].get(item["id"], "")
        k, _ = result_key(item["db_path"], sql, timeout)
        pred_keys.append(k)
    classes = assign_classes(gold_key, pred_keys)
    return gold_key, pred_keys, classes


def true_class_matrix(items, member_names, preds, timeout, log=print):
    """[M,N] equivalence classes (0=correct) + per-item (gold_key, pred_keys)."""
    M, N = len(member_names), len(items)
    tc = np.zeros((M, N), dtype=np.int64)
    keys = []
    for i, it in enumerate(items):
        gold_key, pred_keys, classes = _exec_item(it, member_names, preds, timeout)
        tc[:, i] = classes
        keys.append((gold_key, pred_keys))
        if (i + 1) % 50 == 0 or i + 1 == N:
            log(f"  [exec] {i+1}/{N}")
    return tc, keys


def execution_verifier(keys, tc):
    """Real, execution-grounded verifier guess per item.

    Points at the most-supported NON-EMPTY, executable result class; abstains
    (sentinel) if every result is empty or failed. NOTE: this is partly
    pool-correlated (it cannot certify a query that runs but returns wrong rows) --
    the realistic, non-idealized verifier, unlike the simulator's independent oracle.
    """
    N = len(keys)
    vg = np.full(N, 10 ** 9, dtype=np.int64)   # sentinel: no class matches -> no effect
    for i, (gold_key, pred_keys) in enumerate(keys):
        # tally support per class among non-empty executable results
        support = {}
        for m, k in enumerate(pred_keys):
            if is_nonempty(k):
                support[int(tc[m, i])] = support.get(int(tc[m, i]), 0) + 1
        if support:
            vg[i] = max(support, key=support.get)
    vc = (vg == 0)
    return vg, vc


def seen_prior(source_items, member_names, preds_src, timeout, log=print):
    """Per-model accuracy on the labeled SOURCE (train) split = the seen calibration.
    Real cross-domain prior (Spider train/dev share no databases -> genuine shift)."""
    M, Ns = len(member_names), len(source_items)
    acc = np.zeros(M)
    for mi, name in enumerate(member_names):
        c = 0
        for it in source_items:
            gold_key, _ = result_key(it["db_path"], it["gold_sql"], timeout)
            k, _ = result_key(it["db_path"], preds_src[name].get(it["id"], ""), timeout)
            if gold_key is not None and k is not None and k == gold_key:
                c += 1
        acc[mi] = c / max(1, Ns)
        log(f"  [prior] {name}: source EX={acc[mi]:.3f}")
    sigma = np.sqrt(np.clip(acc * (1 - acc), 1e-4, None) / max(1, Ns))
    return acc, np.maximum(sigma, 1e-3)


def build_poolrun(items, member_names, group_ids, preds_dev, preds_src,
                  source_items, timeout, log=print):
    tc, keys = true_class_matrix(items, member_names, preds_dev, timeout, log=log)
    true_acc = (tc == 0).mean(axis=1)
    vg, vc = execution_verifier(keys, tc)
    prior, prior_sigma = seen_prior(source_items, member_names, preds_src, timeout, log=log)
    M, N = tc.shape
    group = np.asarray(group_ids)
    n_groups = int(group.max()) + 1
    run = PoolRun(true_class=tc, true_acc=true_acc, group=group, prior=prior,
                  prior_sigma=prior_sigma, b=np.zeros(N), phi=np.zeros(N),
                  verifier_guess=vg, verifier_correct=vc, M=M, N=N, n_groups=n_groups)
    return run, keys
