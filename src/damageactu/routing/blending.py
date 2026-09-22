from __future__ import annotations
import numpy as np


def route_logits(post_logits, pair_logits, gate):
    gate = np.asarray(gate, dtype=float).reshape(-1,1)
    return np.asarray(post_logits) + gate * (np.asarray(pair_logits)-np.asarray(post_logits))


def blend_predictions(post_logits, pair_logits, alpha):
    return route_logits(post_logits, pair_logits, np.full(len(post_logits), float(alpha))).argmax(1)


def max_confidence_selector(pred_post, pred_pair, conf_post, conf_pair):
    use_pair = np.asarray(conf_pair) > np.asarray(conf_post)
    return np.where(use_pair, np.asarray(pred_pair), np.asarray(pred_post))


def min_entropy_selector(pred_post, pred_pair, ent_post, ent_pair):
    use_pair = np.asarray(ent_pair) < np.asarray(ent_post)
    return np.where(use_pair, np.asarray(pred_pair), np.asarray(pred_post))
