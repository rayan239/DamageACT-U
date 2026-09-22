from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


CLASS_IDS = [0,1,2,3]


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="macro", zero_division=0))


def per_class_f1(y_true, y_pred):
    vals = f1_score(y_true, y_pred, labels=CLASS_IDS, average=None, zero_division=0)
    return {k: float(vals[k]) for k in CLASS_IDS}


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def ordinal_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, int)-np.asarray(y_pred, int))))


def confidence(prob):
    return np.asarray(prob).max(1)


def entropy(prob):
    p = np.clip(np.asarray(prob, float), 1e-12, 1.0)
    return -(p*np.log(p)).sum(1)


def expected_severity(prob):
    return np.asarray(prob, float) @ np.arange(4, dtype=float)
