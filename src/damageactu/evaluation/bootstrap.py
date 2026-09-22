from __future__ import annotations
import numpy as np
from .metrics import macro_f1


def paired_scene_bootstrap(frame, pred_a, pred_b, scene_col="scene_id", n_replicates=2000, seed=20260917):
    y = frame["true_label"].to_numpy(int)
    a = np.asarray(pred_a, int)
    b = np.asarray(pred_b, int)
    scenes = frame[scene_col].astype(str).to_numpy()
    uniq = np.unique(scenes)
    idx = {s: np.where(scenes == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(int(n_replicates)):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx[s] for s in draw])
        diffs.append(macro_f1(y[ix], b[ix]) - macro_f1(y[ix], a[ix]))
    arr = np.asarray(diffs, float)
    lo, hi = np.percentile(arr, [2.5,97.5])
    return {
        "n_replicates": int(len(arr)),
        "mean_gain": float(arr.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }
