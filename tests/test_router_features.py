import pandas as pd
from damageactu.routing.features import build_router_features


def test_router_feature_count_and_no_label_leak():
    row = {"pred_post":0,"pred_pair":1}
    for k in range(4):
        row[f"p{k}_post"] = 0.25
        row[f"p{k}_pair"] = 0.25
    row.update({
        "confidence_post":0.25,"confidence_pair":0.25,
        "entropy_post":1.386294,"entropy_pair":1.386294,
        "severity_post":1.5,"severity_pair":1.5,
        "true_label":3,"disaster":"should_not_enter_features",
    })
    X = build_router_features(pd.DataFrame([row]))
    assert X.shape[1] == 26
    assert "true_label" not in X.columns
    assert "disaster" not in X.columns
