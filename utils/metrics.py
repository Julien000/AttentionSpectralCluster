from sklearn.metrics import pair_confusion_matrix
import torch
import numpy as np

def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    # ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    # p, r = tp / (tp + fp), tp / (tp + fn)
    # f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    # ARI 越大效果也就越好。
    # return ri, ari, f_beta
    return  ari

def purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        #TODO: 修改了类型
        labels_tmp = labels_true[idx, :].reshape(-1).astype('int32')
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]

def acc(y_true, y_pred):
    """
    https://github.com/XifengGuo/DEC-keras/blob/master/metrics.py
    
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = np.asarray(linear_assignment(w.max() - w)).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
