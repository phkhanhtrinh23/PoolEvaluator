"""Evaluation metrics. Accuracy errors are in points (x100)."""
import numpy as np
from scipy.stats import kendalltau


def mae(acc_hat, acc_true):
    return float(np.mean(np.abs(np.asarray(acc_hat) - np.asarray(acc_true))) * 100)


def flip_rate(acc_hat, acc_true):
    a, t = np.asarray(acc_hat), np.asarray(acc_true)
    M = len(a); bad = tot = 0
    for i in range(M):
        for j in range(i + 1, M):
            if (a[i] - a[j]) * (t[i] - t[j]) < 0:
                bad += 1
            tot += 1
    return bad / tot if tot else 0.0


def kendall(acc_hat, acc_true):
    tau, _ = kendalltau(acc_hat, acc_true)
    return float(tau) if tau == tau else 0.0


def top1(acc_hat, acc_true):
    return float(np.argmax(acc_hat) == np.argmax(acc_true))


def topk(acc_hat, acc_true, k=3):
    a = set(np.argsort(acc_hat)[-k:])
    t = set(np.argsort(acc_true)[-k:])
    return len(a & t) / k


def all_metrics(acc_hat, acc_true):
    return dict(MAE=mae(acc_hat, acc_true), Flip=flip_rate(acc_hat, acc_true),
                Kendall=kendall(acc_hat, acc_true), Top1=top1(acc_hat, acc_true),
                Top3=topk(acc_hat, acc_true, 3))
