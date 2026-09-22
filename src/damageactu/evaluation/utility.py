from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score


def correctness_discordant_target(y, pred_post, pred_pair):
    y = np.asarray(y, int)
    pc = np.asarray(pred_post, int) == y
    tc = np.asarray(pred_pair, int) == y
    discordant = pc != tc
    q = tc[discordant].astype(int)
    return discordant, q


def delta_ce(y, post_prob, pair_prob):
    y = np.asarray(y, int)
    i = np.arange(len(y))
    pp = np.clip(np.asarray(post_prob, float)[i,y], 1e-12, 1.0)
    pq = np.clip(np.asarray(pair_prob, float)[i,y], 1e-12, 1.0)
    return np.log(pq)-np.log(pp)


def utility_scores(y, pred_post, pred_pair, gate, post_prob, pair_prob):
    disc, q = correctness_discordant_target(y, pred_post, pred_pair)
    g = np.asarray(gate, float)
    if len(np.unique(q)) != 2:
        raise RuntimeError("Utility AUROC target does not contain both classes.")
    dce = delta_ce(y, post_prob, pair_prob)
    return {
        "n_discordant": int(disc.sum()),
        "auroc": float(roc_auc_score(q, g[disc])),
        "auprc": float(average_precision_score(q, g[disc])),
        "spearman_gate_deltaCE": float(spearmanr(g, dce).statistic),
    }
