from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from damageactu.evaluation.metrics import accuracy, macro_f1, ordinal_mae, per_class_f1
from damageactu.routing.blending import blend_predictions, max_confidence_selector, min_entropy_selector, route_logits
from damageactu.routing.features import FEATURE_NAMES, build_router_features
from damageactu.routing.neural_router import build_neural_router

CLEAN_PATH = ROOT / "predictions" / "heldout_events" / "event_test_clean_expert_predictions.csv.gz"
WRONG_PATH = ROOT / "predictions" / "heldout_events" / "event_test_wrongpre_siamese_predictions.csv.gz"
LOGISTIC_PATH = ROOT / "checkpoints" / "routers" / "final_logistic_soft_gate.joblib"
HGB_PATH = ROOT / "checkpoints" / "routers" / "final_hgb_hard_router.joblib"
NEURAL_PATH = ROOT / "checkpoints" / "routers" / "final_neural_safe_router.pt"
ALPHA_PATH = ROOT / "checkpoints" / "routers" / "final_static_alpha.json"
FROZEN_METRICS_PATH = ROOT / "results" / "heldout_events" / "test_metrics_all_conditions.csv"
FROZEN_CLASS_PATH = ROOT / "results" / "heldout_events" / "test_per_class_f1_all_conditions.csv"
FROZEN_EVENT_PATH = ROOT / "results" / "heldout_events" / "test_per_event_metrics.csv"
FROZEN_UTILITY_PATH = ROOT / "results" / "heldout_events" / "test_neural_utility.json"
FROZEN_INDEX_PATH = ROOT / "results" / "heldout_events" / "results_index.json"
OUT_PATH = ROOT / "results" / "reproduction_core_v2.json"

METHODS = [
    "POST-only",
    "Siamese",
    "Equal 0.5 logit blend",
    "Static alpha blend",
    "Max-confidence selector",
    "Min-entropy selector",
    "Logistic soft gate",
    "HGB hard router",
    "Neural Temporal Utility Router",
    "Oracle expert selector",
]

EXPECTED_SKLEARN = "1.6.1"
PRED_TOL = 0.0
METRIC_TOL = 5e-12
FLOAT_TOL = 2e-7


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required canonical artifact missing: {path}")


def logits(df: pd.DataFrame, suffix: str) -> np.ndarray:
    return np.column_stack([df[f"logit{k}_{suffix}"].to_numpy(float) for k in range(4)])


def probabilities(df: pd.DataFrame, suffix: str) -> np.ndarray:
    return np.column_stack([df[f"p{k}_{suffix}"].to_numpy(float) for k in range(4)])


def metric_row(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": macro_f1(y, p),
        "accuracy": accuracy(y, p),
        "ordinal_mae": ordinal_mae(y, p),
    }


def assert_close(name: str, got: float, expected: float, tol: float = METRIC_TOL) -> None:
    if not np.isfinite(got) or abs(float(got) - float(expected)) > tol:
        raise RuntimeError(f"PARITY FAIL {name}: got={got:.16g}, expected={expected:.16g}, tol={tol}")


def neural_gate(df: pd.DataFrame, artifact: dict, feature_frame: pd.DataFrame) -> np.ndarray:
    if list(feature_frame.columns) != list(artifact["feature_names"]):
        raise RuntimeError("Neural artifact feature order does not match build_router_features output.")
    x = feature_frame.to_numpy(np.float32)
    mean = np.asarray(artifact["mean"], dtype=np.float32)
    std = np.asarray(artifact["std"], dtype=np.float32)
    if mean.shape != (26,) or std.shape != (26,):
        raise RuntimeError("Unexpected neural normalization shape.")
    x = (x - mean) / std
    model = build_neural_router(n_features=26)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        gate = torch.sigmoid(model(torch.from_numpy(x))).cpu().numpy()
    return np.asarray(gate, dtype=float)


def build_methods(df: pd.DataFrame, alpha: float, logistic, hgb, neural_artifact: dict):
    y = df["true_label"].to_numpy(int)
    pred_post = df["pred_post"].to_numpy(int)
    pred_pair = df["pred_pair"].to_numpy(int)
    lp = logits(df, "post")
    lq = logits(df, "pair")
    x = build_router_features(df)

    artifact_features = list(neural_artifact["feature_names"])
    if list(x.columns) != artifact_features:
        raise RuntimeError("Router feature order mismatch against frozen neural artifact.")
    if hasattr(logistic, "feature_names_in_") and list(logistic.feature_names_in_) != artifact_features:
        raise RuntimeError("Logistic feature order mismatch against frozen neural artifact.")
    if hasattr(hgb, "feature_names_in_") and list(hgb.feature_names_in_) != artifact_features:
        raise RuntimeError("HGB feature order mismatch against frozen neural artifact.")

    logistic_gate = logistic.predict_proba(x)[:, 1]
    hgb_prob = hgb.predict_proba(x)[:, 1]
    neural_u = neural_gate(df, neural_artifact, x)

    out = {
        "POST-only": pred_post,
        "Siamese": pred_pair,
        "Equal 0.5 logit blend": blend_predictions(lp, lq, 0.5),
        "Static alpha blend": blend_predictions(lp, lq, alpha),
        "Max-confidence selector": max_confidence_selector(
            pred_post, pred_pair,
            df["confidence_post"].to_numpy(float),
            df["confidence_pair"].to_numpy(float),
        ),
        "Min-entropy selector": min_entropy_selector(
            pred_post, pred_pair,
            df["entropy_post"].to_numpy(float),
            df["entropy_pair"].to_numpy(float),
        ),
        "Logistic soft gate": route_logits(lp, lq, logistic_gate).argmax(1),
        "HGB hard router": np.where(hgb_prob >= 0.5, pred_pair, pred_post),
        "Neural Temporal Utility Router": route_logits(lp, lq, neural_u).argmax(1),
        "Oracle expert selector": np.where(pred_pair == y, pred_pair, pred_post),
    }
    return out, {"logistic": logistic_gate, "hgb": hgb_prob, "neural": neural_u}


def align_valid_and_wrong(clean: pd.DataFrame, wrong: pd.DataFrame):
    if not clean["building_id"].astype(str).is_unique:
        raise RuntimeError("Clean building IDs are not unique.")
    if not wrong["building_id"].astype(str).is_unique:
        raise RuntimeError("Wrong-PRE building IDs are not unique.")

    c = clean.copy()
    c["building_id"] = c["building_id"].astype(str)
    w = wrong.copy()
    w["building_id"] = w["building_id"].astype(str)
    missing = set(w["building_id"]) - set(c["building_id"])
    if missing:
        raise RuntimeError(f"Wrong-PRE targets missing from clean table: {len(missing)}")

    valid = c.set_index("building_id", drop=False).loc[w["building_id"].tolist()].reset_index(drop=True)
    for col in ["scene_id", "disaster", "true_label"]:
        if not np.array_equal(valid[col].astype(str).to_numpy(), w[col].astype(str).to_numpy()):
            raise RuntimeError(f"Wrong-PRE alignment mismatch in {col}.")

    post_cols = [
        "building_id", "scene_id", "disaster", "true_label", "pred_post",
        "confidence_post", "entropy_post", "severity_post",
    ]
    for k in range(4):
        post_cols += [f"p{k}_post", f"logit{k}_post"]
    pair_cols = ["building_id", "pred_pair", "confidence_pair", "entropy_pair", "severity_pair"]
    for k in range(4):
        pair_cols += [f"p{k}_pair", f"logit{k}_pair"]

    wrong_hybrid = valid[post_cols].merge(w[pair_cols], on="building_id", how="inner", validate="one_to_one", sort=False)
    if len(wrong_hybrid) != len(w):
        raise RuntimeError("Wrong-PRE hybrid population size changed during merge.")
    return valid, wrong_hybrid


def compare_metric_table(population: str, df: pd.DataFrame, predictions: dict, frozen: pd.DataFrame, checks: list):
    y = df["true_label"].to_numpy(int)
    ref = frozen[frozen["population"] == population].set_index("method")
    if set(ref.index) != set(METHODS):
        raise RuntimeError(f"Frozen metric methods mismatch for {population}.")
    for method in METHODS:
        got = metric_row(y, predictions[method])
        row = ref.loc[method]
        for metric in ["macro_f1", "accuracy", "ordinal_mae"]:
            assert_close(f"{population}/{method}/{metric}", got[metric], float(row[metric]))
        checks.append(f"PASS metrics {population}: {method}")


def compare_per_class(population: str, df: pd.DataFrame, predictions: dict, frozen: pd.DataFrame, checks: list):
    y = df["true_label"].to_numpy(int)
    ref = frozen[frozen["population"] == population]
    for method in METHODS:
        got = per_class_f1(y, predictions[method])
        sub = ref[ref["method"] == method].set_index("class_id")
        for k in range(4):
            assert_close(f"{population}/{method}/class{k}", got[k], float(sub.loc[k, "f1"]))
        checks.append(f"PASS per-class {population}: {method}")


def compare_per_event(clean: pd.DataFrame, predictions: dict, frozen: pd.DataFrame, checks: list):
    for method in METHODS:
        p = predictions[method]
        work = clean[["disaster", "true_label"]].copy()
        work["pred"] = p
        for event, g in work.groupby("disaster", sort=True):
            y = g["true_label"].to_numpy(int)
            q = g["pred"].to_numpy(int)
            got = metric_row(y, q)
            row = frozen[(frozen["event"] == event) & (frozen["method"] == method)]
            if len(row) != 1:
                raise RuntimeError(f"Frozen per-event row missing/duplicated: {event}/{method}")
            rr = row.iloc[0]
            assert_close(f"event/{event}/{method}/macro_f1", got["macro_f1"], float(rr["macro_f1_fixed4"]))
            assert_close(f"event/{event}/{method}/accuracy", got["accuracy"], float(rr["accuracy"]))
            assert_close(f"event/{event}/{method}/ordinal_mae", got["ordinal_mae"], float(rr["ordinal_mae"]))
    checks.append("PASS all clean per-event metrics")


def utility_point_estimates(clean: pd.DataFrame, neural_gate_values: np.ndarray) -> dict:
    y = clean["true_label"].to_numpy(int)
    pp = clean["pred_post"].to_numpy(int)
    pq = clean["pred_pair"].to_numpy(int)
    post_correct = pp == y
    pair_correct = pq == y
    discordant = post_correct != pair_correct
    q = pair_correct[discordant].astype(int)
    post_prob = probabilities(clean, "post")
    pair_prob = probabilities(clean, "pair")
    i = np.arange(len(y))
    p_post_y = np.clip(post_prob[i, y], 1e-12, 1.0)
    p_pair_y = np.clip(pair_prob[i, y], 1e-12, 1.0)
    dce = np.log(p_pair_y) - np.log(p_post_y)
    return {
        "n_correctness_discordant": int(discordant.sum()),
        "auroc": float(roc_auc_score(q, neural_gate_values[discordant])),
        "auprc": float(average_precision_score(q, neural_gate_values[discordant])),
        "spearman_gate_deltaCE": float(spearmanr(neural_gate_values, dce).statistic),
    }


def main():
    for p in [
        CLEAN_PATH, WRONG_PATH, LOGISTIC_PATH, HGB_PATH, NEURAL_PATH, ALPHA_PATH,
        FROZEN_METRICS_PATH, FROZEN_CLASS_PATH, FROZEN_EVENT_PATH,
        FROZEN_UTILITY_PATH, FROZEN_INDEX_PATH,
    ]:
        require(p)

    print("DamageACT-U statistical reproduction — deterministic core")
    print("Python:", sys.version.split()[0])
    print("scikit-learn:", sklearn.__version__)
    print("torch:", torch.__version__)
    if sklearn.__version__ != EXPECTED_SKLEARN:
        raise RuntimeError(
            f"Strict router deserialization requires scikit-learn {EXPECTED_SKLEARN}; "
            f"found {sklearn.__version__}."
        )

    clean = pd.read_csv(CLEAN_PATH, low_memory=False)
    wrong = pd.read_csv(WRONG_PATH, low_memory=False)
    if len(clean) != 115349:
        raise RuntimeError(f"Unexpected clean TEST size: {len(clean)}")
    if len(wrong) != 115269:
        raise RuntimeError(f"Unexpected wrong-PRE matched size: {len(wrong)}")

    alpha = float(json.loads(ALPHA_PATH.read_text(encoding="utf-8"))["alpha"])
    if alpha != 0.45:
        raise RuntimeError(f"Frozen static alpha changed: {alpha}")

    logistic = joblib.load(LOGISTIC_PATH)
    hgb = joblib.load(HGB_PATH)
    neural_artifact = torch.load(NEURAL_PATH, map_location="cpu", weights_only=False)
    if int(neural_artifact.get("best_epoch", -1)) != 1:
        raise RuntimeError("Frozen neural best_epoch is not 1.")
    if int(neural_artifact.get("expert_seed", -1)) != 42:
        raise RuntimeError("Frozen neural expert_seed is not 42.")

    # Important metadata audit: RC1's FEATURE_NAMES constant is not authoritative.
    # The authoritative order is the actual DataFrame order plus the frozen artifact names.
    probe = build_router_features(clean.head(2))
    source_feature_constant_matches = list(FEATURE_NAMES) == list(probe.columns)

    valid, wrong_hybrid = align_valid_and_wrong(clean, wrong)

    clean_pred, clean_gate = build_methods(clean, alpha, logistic, hgb, neural_artifact)
    valid_pred, valid_gate = build_methods(valid, alpha, logistic, hgb, neural_artifact)
    wrong_pred, wrong_gate = build_methods(wrong_hybrid, alpha, logistic, hgb, neural_artifact)

    frozen_metrics = pd.read_csv(FROZEN_METRICS_PATH)
    frozen_class = pd.read_csv(FROZEN_CLASS_PATH)
    frozen_event = pd.read_csv(FROZEN_EVENT_PATH)
    frozen_utility = json.loads(FROZEN_UTILITY_PATH.read_text(encoding="utf-8"))
    frozen_index = json.loads(FROZEN_INDEX_PATH.read_text(encoding="utf-8"))

    checks: list[str] = []
    compare_metric_table("clean_pooled", clean, clean_pred, frozen_metrics, checks)
    compare_metric_table("valid_matched", valid, valid_pred, frozen_metrics, checks)
    compare_metric_table("wrongpre_matched", wrong_hybrid, wrong_pred, frozen_metrics, checks)
    compare_per_class("clean_pooled", clean, clean_pred, frozen_class, checks)
    compare_per_class("wrongpre_matched", wrong_hybrid, wrong_pred, frozen_class, checks)
    compare_per_event(clean, clean_pred, frozen_event, checks)

    utility = utility_point_estimates(clean, clean_gate["neural"])
    if utility["n_correctness_discordant"] != int(frozen_utility["n_correctness_discordant"]):
        raise RuntimeError("Utility discordant population mismatch.")
    assert_close("utility/auroc", utility["auroc"], float(frozen_utility["auroc"]), FLOAT_TOL)
    assert_close("utility/auprc", utility["auprc"], float(frozen_utility["auprc"]), FLOAT_TOL)
    assert_close(
        "utility/spearman_gate_deltaCE",
        utility["spearman_gate_deltaCE"],
        float(frozen_utility["spearman_gate_deltaCE"]),
        FLOAT_TOL,
    )
    valid_gate_mean = float(np.mean(valid_gate["neural"]))
    wrong_gate_mean = float(np.mean(wrong_gate["neural"]))
    assert_close("gate/valid_matched_mean", valid_gate_mean, float(frozen_utility["valid_gate_mean_matched"]), FLOAT_TOL)
    assert_close("gate/wrong_matched_mean", wrong_gate_mean, float(frozen_utility["wrong_gate_mean_matched"]), FLOAT_TOL)
    checks.append("PASS temporal-utility point estimates and neural gate means")

    post_clean = metric_row(clean["true_label"].to_numpy(int), clean_pred["POST-only"])["macro_f1"]
    neural_clean = metric_row(clean["true_label"].to_numpy(int), clean_pred["Neural Temporal Utility Router"])["macro_f1"]
    primary_gain = neural_clean - post_clean
    assert_close("primary_clean_neural_minus_post", primary_gain, float(frozen_index["primary_clean_neural_minus_post"]))

    post_wrong = metric_row(wrong_hybrid["true_label"].to_numpy(int), wrong_pred["POST-only"])["macro_f1"]
    neural_wrong = metric_row(wrong_hybrid["true_label"].to_numpy(int), wrong_pred["Neural Temporal Utility Router"])["macro_f1"]
    wrong_gain = neural_wrong - post_wrong
    assert_close("wrongpre_neural_minus_post", wrong_gain, float(frozen_index["wrongpre_neural_minus_post"]))
    checks.append("PASS headline clean and wrong-PRE gains")

    report = {
        "status": "PASS",
        "scope": "deterministic statistical reproduction from frozen expert predictions and frozen routers; bootstrap CIs are a separate next-stage check",
        "environment": {
            "python": sys.version,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "artifact_facts": {
            "clean_test_n": int(len(clean)),
            "wrongpre_matched_n": int(len(wrong_hybrid)),
            "static_alpha": alpha,
            "neural_best_epoch": int(neural_artifact["best_epoch"]),
            "expert_seed": int(neural_artifact["expert_seed"]),
            "router_feature_count": len(neural_artifact["feature_names"]),
            "source_FEATURE_NAMES_constant_matches_authoritative_order": bool(source_feature_constant_matches),
        },
        "headline_reproduced": {
            "post_clean_macro_f1": post_clean,
            "neural_clean_macro_f1": neural_clean,
            "neural_minus_post_clean": primary_gain,
            "post_wrongpre_macro_f1": post_wrong,
            "neural_wrongpre_macro_f1": neural_wrong,
            "neural_minus_post_wrongpre": wrong_gain,
            "valid_gate_mean_matched": valid_gate_mean,
            "wrong_gate_mean_matched": wrong_gate_mean,
            **utility,
        },
        "frozen_decision_preserved": {
            "all_precommitted_criteria_pass": bool(frozen_index["all_precommitted_criteria_pass"]),
            "post_test_tuning_permitted": bool(frozen_index["post_test_tuning_permitted"]),
        },
        "checks_passed": checks,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nCORE REPRODUCTION PASS")
    print(f"POST clean Macro-F1     : {post_clean:.12f}")
    print(f"Neural clean Macro-F1   : {neural_clean:.12f}")
    print(f"Clean gain              : {primary_gain:+.12f}")
    print(f"Neural wrong-PRE F1     : {neural_wrong:.12f}")
    print(f"Wrong-PRE gain vs POST  : {wrong_gain:+.12f}")
    print(f"Utility AUROC           : {utility['auroc']:.12f}")
    print(f"Utility AUPRC           : {utility['auprc']:.12f}")
    print(f"Gate-deltaCE Spearman   : {utility['spearman_gate_deltaCE']:.12f}")
    print(f"Valid gate mean matched : {valid_gate_mean:.12f}")
    print(f"Wrong gate mean matched : {wrong_gate_mean:.12f}")
    print(f"ALL_PRECOMMITTED_CRITERIA_PASS preserved: {frozen_index['all_precommitted_criteria_pass']}")
    if not source_feature_constant_matches:
        print("NOTE: src/damageactu/routing/features.py FEATURE_NAMES constant order differs from the authoritative frozen feature order. build_router_features output is correct; fix the constant only after this parity checkpoint is committed.")
    print(f"REPORT: {OUT_PATH}")


if __name__ == "__main__":
    main()
