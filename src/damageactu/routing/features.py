from __future__ import annotations
import numpy as np
import pandas as pd


FEATURE_NAMES = (
    [
        name
        for k in range(4)
        for name in (
            f"post_p{k}", f"pair_p{k}", f"diff_p{k}", f"absdiff_p{k}"
        )
    ] +
    ["confidence_post","confidence_pair","entropy_post","entropy_pair",
     "severity_post","severity_pair","conf_diff","entropy_diff","severity_diff","pred_disagree"]
)


def build_router_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for k in range(4):
        p = df[f"p{k}_post"].astype(float)
        q = df[f"p{k}_pair"].astype(float)
        x[f"post_p{k}"] = p
        x[f"pair_p{k}"] = q
        x[f"diff_p{k}"] = q - p
        x[f"absdiff_p{k}"] = (q - p).abs()
    for c in ["confidence_post","confidence_pair","entropy_post","entropy_pair","severity_post","severity_pair"]:
        x[c] = df[c].astype(float)
    x["conf_diff"] = x["confidence_pair"] - x["confidence_post"]
    x["entropy_diff"] = x["entropy_pair"] - x["entropy_post"]
    x["severity_diff"] = x["severity_pair"] - x["severity_post"]
    x["pred_disagree"] = (df["pred_pair"].to_numpy(int) != df["pred_post"].to_numpy(int)).astype(float)
    if x.shape[1] != 26:
        raise RuntimeError(f"Expected 26 router features, got {x.shape[1]}")
    if not np.isfinite(x.to_numpy(float)).all():
        raise RuntimeError("Non-finite router features.")
    return x
