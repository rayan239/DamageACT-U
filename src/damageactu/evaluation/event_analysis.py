from __future__ import annotations
import pandas as pd
from .metrics import macro_f1, accuracy, ordinal_mae, per_class_f1


def summarize_predictions(df, pred_col, event_col="disaster"):
    y = df["true_label"].to_numpy(int)
    p = df[pred_col].to_numpy(int)
    pooled = {
        "n": int(len(df)),
        "macro_f1": macro_f1(y,p),
        "accuracy": accuracy(y,p),
        "ordinal_mae": ordinal_mae(y,p),
        "per_class_f1": per_class_f1(y,p),
    }
    rows = []
    for event, g in df.groupby(event_col, sort=True):
        yy = g["true_label"].to_numpy(int)
        pp = g[pred_col].to_numpy(int)
        rows.append({"event":event, "n":len(g), "macro_f1":macro_f1(yy,pp)})
    per_event = pd.DataFrame(rows)
    pooled["unweighted_event_mean_macro_f1"] = float(per_event["macro_f1"].mean()) if len(per_event) else None
    return pooled, per_event
