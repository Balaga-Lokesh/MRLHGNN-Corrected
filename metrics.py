import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def paper_metrics(y, s):
    """Metrics exactly as in the paper's utils.get_metrics (from LAGCN, Yu et al. 2021).

    999 score thresholds; AUC/AUPR by trapezoid; F1/Acc/Recall/Spec/Prec at the F1-optimal threshold.
    Vectorised with searchsorted instead of building a 999 x N matrix.
    """
    y = np.asarray(y, float).ravel(); s = np.asarray(s, float).ravel()
    u = np.unique(s)
    th = u[np.int32(len(u) * np.arange(1, 1000) / 1000)]
    s_all, s_pos = np.sort(s), np.sort(s[y == 1])
    TP = len(s_pos) - np.searchsorted(s_pos, th, 'left')
    PP = len(s_all) - np.searchsorted(s_all, th, 'left')
    FP = PP - TP; FN = y.sum() - TP; TN = len(y) - TP - FP - FN
    fpr, tpr = FP / (FP + TN), TP / (TP + FN)
    roc = np.array(sorted(np.column_stack((fpr, tpr)).tolist())); roc[0] = [0, 0]
    roc = np.vstack([roc, [1, 1]])
    auc = np.sum(0.5 * (roc[1:, 0] - roc[:-1, 0]) * (roc[:-1, 1] + roc[1:, 1]))
    prec = TP / (TP + FP)
    pr = np.array(sorted(np.column_stack((tpr, prec)).tolist())); pr[0] = [0, 1]
    pr = np.vstack([pr, [1, 0]])
    aupr = np.sum(0.5 * (pr[1:, 0] - pr[:-1, 0]) * (pr[:-1, 1] + pr[1:, 1]))
    f1 = 2 * TP / (len(y) + TP - TN)
    i = np.argmax(f1)
    return dict(AUC=auc, AUPR=aupr, F1=f1[i], Accuracy=((TP + TN) / len(y))[i], Recall=tpr[i],
                Specificity=(TN / (TN + FP))[i], Precision=prec[i])


def sk_metrics(y, s):
    """Standard sklearn ROC-AUC and average precision, reported alongside for transparency."""
    return dict(sk_AUC=roc_auc_score(y, s), sk_AP=average_precision_score(y, s))
