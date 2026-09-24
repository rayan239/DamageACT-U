#!/usr/bin/env python3
"""
DamageACT-U G4B deterministic router/intervention parity.

No raw xBD, no expert inference, no model fitting, no tuning.
The runner consumes only already-frozen expert prediction tables, router artifacts,
wrong-PRE mapping, and canonical result tables.
"""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn

EXPECTED_G4A_COMMIT = "9ca56cf"
EXPECTED_G4B_V1_PROTOCOL_COMMIT = "9d86349"
EXPECTED_RUNTIME_FIX_SHA256 = "0c66d95faadf85c990f2e771ac8c9697a7fb6e97edac7482639ad5a001eaf4c1"
EXPECTED_G4A_INVENTORY_SHA256 = "3543d47ffa26c1f1d043877b4b17870b7ed06031c5d41ef493ef44f2fa3f2d76"
EXPECTED_PROTOCOL_SHA256 = "3e99481439c22091e45f3f6c48c88ea7b6bef3736207de53812fdf42755e2e66"

FEATURE_NAMES = [
    "post_p0","pair_p0","diff_p0","absdiff_p0",
    "post_p1","pair_p1","diff_p1","absdiff_p1",
    "post_p2","pair_p2","diff_p2","absdiff_p2",
    "post_p3","pair_p3","diff_p3","absdiff_p3",
    "confidence_post","confidence_pair","entropy_post","entropy_pair",
    "severity_post","severity_pair","conf_diff","entropy_diff",
    "severity_diff","pred_disagree",
]

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

HEADLINE = {
    "post_clean_macro_f1": 0.456562921272,
    "neural_clean_macro_f1": 0.459673702532,
    "neural_clean_gain_vs_post": 0.003110781260,
    "neural_wrongpre_macro_f1": 0.453240019462,
    "neural_wrongpre_gain_vs_post": -0.003070980525,
    "utility_auroc": 0.772838348490,
    "utility_auprc": 0.719635385987,
    "gate_deltaCE_spearman": 0.103607093389,
    "valid_gate_mean_matched": 0.175213195858,
    "wrong_gate_mean_matched": 0.165126243399,
}

CANONICAL_METRIC_ATOL = 1e-10
HEADLINE_ATOL = 1e-9
SPEARMAN_ATOL = 1e-8

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def require(cond, message):
    if not cond:
        raise RuntimeError(message)

def close(a, b, atol):
    return abs(float(a) - float(b)) <= atol

class UtilityRouter(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def build_features(df):
    x = pd.DataFrame(index=df.index)
    for k in range(4):
        p = df[f"p{k}_post"].astype(float)
        q = df[f"p{k}_pair"].astype(float)
        x[f"post_p{k}"] = p
        x[f"pair_p{k}"] = q
        x[f"diff_p{k}"] = q - p
        x[f"absdiff_p{k}"] = (q - p).abs()
    for c in [
        "confidence_post","confidence_pair",
        "entropy_post","entropy_pair",
        "severity_post","severity_pair",
    ]:
        x[c] = df[c].astype(float)
    x["conf_diff"] = df["confidence_pair"].astype(float) - df["confidence_post"].astype(float)
    x["entropy_diff"] = df["entropy_pair"].astype(float) - df["entropy_post"].astype(float)
    x["severity_diff"] = df["severity_pair"].astype(float) - df["severity_post"].astype(float)
    x["pred_disagree"] = (
        df["pred_pair"].to_numpy(int) != df["pred_post"].to_numpy(int)
    ).astype(float)
    require(list(x.columns) == FEATURE_NAMES, "26-feature order changed.")
    require(np.isfinite(x.to_numpy(float)).all(), "Non-finite router features.")
    return x

def logits(df, branch):
    return df[[f"logit{k}_{branch}" for k in range(4)]].to_numpy(np.float32)

def prob_class1(model, X):
    pr = model.predict_proba(X)
    cls = np.asarray(model.classes_)
    pos = np.where(cls == 1)[0]
    require(len(pos) == 1, "Router classifier has no unique class-1 probability.")
    return pr[:, pos[0]]

def route_all(df, logistic, hgb, neural, mean, std, alpha):
    X = build_features(df)
    lp = logits(df, "post")
    lt = logits(df, "pair")

    ulog = prob_class1(logistic, X)
    uhgb = prob_class1(hgb, X)

    zn = (X.to_numpy(np.float32) - mean) / std
    require(np.isfinite(zn).all(), "Non-finite normalized neural-router features.")
    with torch.inference_mode():
        uneu = torch.sigmoid(neural(torch.tensor(zn, dtype=torch.float32))).cpu().numpy()

    y = df["true_label"].to_numpy(int)
    post = df["pred_post"].to_numpy(int)
    pair = df["pred_pair"].to_numpy(int)

    pred = {
        "POST-only": post,
        "Siamese": pair,
        "Equal 0.5 logit blend": (lp + 0.5 * (lt - lp)).argmax(1),
        "Static alpha blend": (lp + alpha * (lt - lp)).argmax(1),
        "Max-confidence selector": np.where(
            df["confidence_pair"].to_numpy(float) > df["confidence_post"].to_numpy(float),
            pair, post
        ),
        "Min-entropy selector": np.where(
            df["entropy_pair"].to_numpy(float) < df["entropy_post"].to_numpy(float),
            pair, post
        ),
        "Logistic soft gate": (lp + ulog[:, None] * (lt - lp)).argmax(1),
        "HGB hard router": np.where(uhgb >= 0.5, pair, post),
        "Neural Temporal Utility Router": (lp + uneu[:, None] * (lt - lp)).argmax(1),
        "Oracle expert selector": np.where((pair == y) & (post != y), pair, post),
    }
    require(list(pred.keys()) == METHODS, "Method roster/order changed.")
    return pred, {"logistic": ulog, "hgb": uhgb, "neural": uneu}

def metric_rows(frame, predictions, condition):
    y = frame["true_label"].to_numpy(int)
    out = []
    for method, p in predictions.items():
        p = np.asarray(p, int)
        out.append({
            "condition": condition,
            "method": method,
            "n": int(len(y)),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=[0,1,2,3], average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, p, labels=[0,1,2,3], average="weighted", zero_division=0)),
            "ordinal_mae": float(np.mean(np.abs(y - p))),
        })
    return pd.DataFrame(out)

def canonical_method_name(method, available):
    if method in available:
        return method
    aliases = {
        "Static alpha blend": [
            "Static alpha blend (a=0.45)",
            "Static alpha blend (a=0.45)",
            "Static alpha logit blend",
        ],
        "HGB hard router": ["Original-spec HGB hard router"],
        "Neural Temporal Utility Router": ["Frozen neural soft router", "Neural soft router"],
    }
    hits = [x for x in aliases.get(method, []) if x in available]
    require(len(hits) == 1, f"Could not uniquely map method {method!r}; available={sorted(available)}")
    return hits[0]

def resolve_grouping_column(canonical, conditions):
    """
    Resolve the historical grouping-field name without changing scientific content.

    Canonical DamageACT-U artifacts have used either:
      - condition
      - population

    We accept exactly one. If neither or both are present for a multi-population
    comparison, fail rather than guessing.
    """
    present = [c for c in ("condition", "population") if c in canonical.columns]

    if len(conditions) == 1 and not present:
        return None

    require(
        len(present) == 1,
        "Canonical multi-population metric table must contain exactly one grouping "
        f"column from {{'condition','population'}}; found {present}; "
        f"columns={list(canonical.columns)}"
    )

    col = present[0]
    observed = set(canonical[col].astype(str))
    missing = [c for c in conditions if c not in observed]
    require(
        not missing,
        f"Canonical grouping column {col!r} is missing required labels {missing}; "
        f"observed={sorted(observed)}"
    )
    return col


def compare_metric_table(regen, canonical, conditions):
    require("method" in canonical.columns, "Canonical metric table lacks method column.")
    require("macro_f1" in canonical.columns, "Canonical metric table lacks macro_f1.")
    require("accuracy" in canonical.columns, "Canonical metric table lacks accuracy.")

    grouping_col = resolve_grouping_column(canonical, conditions)
    comparisons = []

    for condition in conditions:
        rg = regen[regen["condition"] == condition]

        if grouping_col is None:
            cg = canonical
        else:
            cg = canonical[canonical[grouping_col].astype(str) == condition]

        require(len(cg) > 0, f"No canonical rows for population={condition}")
        available = set(cg["method"].astype(str))

        for _, row in rg.iterrows():
            method = str(row["method"])
            cm = canonical_method_name(method, available)
            hit = cg[cg["method"].astype(str) == cm]
            require(
                len(hit) == 1,
                f"Expected one canonical row for {condition}/{cm}; found {len(hit)}"
            )
            hit = hit.iloc[0]

            for metric in ("macro_f1", "accuracy"):
                got = float(row[metric])
                exp = float(hit[metric])
                require(
                    close(got, exp, CANONICAL_METRIC_ATOL),
                    f"Canonical metric mismatch {condition}/{method}/{metric}: "
                    f"got={got:.15f}, exp={exp:.15f}"
                )

            comparisons.append({
                "condition": condition,
                "canonical_grouping_column": grouping_col,
                "regenerated_method": method,
                "canonical_method": cm,
                "macro_f1": float(row["macro_f1"]),
                "accuracy": float(row["accuracy"]),
            })

    return comparisons

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    args = ap.parse_args()
    repo = args.repo_root.resolve()

    require((repo / ".git").exists(), f"Not a Git repository: {repo}")

    protocol_path = repo / "audit/g4b_router_intervention_parity_protocol.json"
    inventory_path = repo / "audit/g4_router_intervention_inventory.json"
    runtime_fix_path = repo / "audit/g4b_runtime_schema_fix_v2.json"

    require(protocol_path.is_file(), "Missing G4B frozen protocol.")
    require(runtime_fix_path.is_file(), "Missing G4B v2 runtime-fix addendum.")
    require(
        sha256_file(runtime_fix_path) == EXPECTED_RUNTIME_FIX_SHA256,
        "G4B v2 runtime-fix addendum SHA mismatch."
    )
    require(sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256, "G4B protocol SHA mismatch.")
    require(inventory_path.is_file(), "Missing frozen G4A inventory.")
    require(sha256_file(inventory_path) == EXPECTED_G4A_INVENTORY_SHA256, "G4A inventory SHA mismatch.")

    head = subprocess.check_output(["git","rev-parse","HEAD"], cwd=repo, text=True).strip()
    anc = subprocess.run(
        ["git","merge-base","--is-ancestor",EXPECTED_G4A_COMMIT,head],
        cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode
    require(anc == 0, f"Required G4A commit {EXPECTED_G4A_COMMIT} is not an ancestor of HEAD.")

    anc_v1 = subprocess.run(
        ["git","merge-base","--is-ancestor",EXPECTED_G4B_V1_PROTOCOL_COMMIT,head],
        cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode
    require(
        anc_v1 == 0,
        f"Original G4B protocol commit {EXPECTED_G4B_V1_PROTOCOL_COMMIT} is not an ancestor of HEAD."
    )

    require(subprocess.run(["git","diff","--quiet"], cwd=repo).returncode == 0,
            "Tracked unstaged changes exist.")
    require(subprocess.run(["git","diff","--cached","--quiet"], cwd=repo).returncode == 0,
            "Staged changes exist.")

    print("=== G4B DETERMINISTIC ROUTER / INTERVENTION PARITY — RUNTIME FIX v2 ===")
    print("HEAD:", head)
    print("Protocol SHA256:", EXPECTED_PROTOCOL_SHA256)
    print("Runtime-fix SHA256:", EXPECTED_RUNTIME_FIX_SHA256)
    print("Inventory SHA256:", EXPECTED_G4A_INVENTORY_SHA256)
    print("Python:", sys.version.split()[0])
    print("numpy:", np.__version__)
    print("pandas:", pd.__version__)
    print("scipy:", scipy.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("joblib:", joblib.__version__)
    print("torch:", torch.__version__)

    require(sys.version_info[:2] == (3,12), "Expected Python 3.12.x.")
    require(sklearn.__version__ == "1.6.1", f"Expected scikit-learn 1.6.1, got {sklearn.__version__}")
    require(joblib.__version__ == "1.6.0", f"Expected joblib 1.6.0, got {joblib.__version__}")
    require(torch.__version__.split("+")[0] == "2.10.0", f"Expected torch 2.10.0, got {torch.__version__}")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    require(inventory.get("status") == "PASS", "G4A inventory was not PASS.")
    require(inventory["feature_schema"]["names"] == FEATURE_NAMES, "G4A feature schema changed.")

    # Reverify every file identity recorded by G4A.
    for rel, rec in inventory["input_hashes"].items():
        p = repo / rel
        require(p.is_file(), f"Frozen G4 input missing: {rel}")
        got = sha256_file(p)
        require(got == rec["sha256"], f"Frozen G4 input hash changed: {rel}")
    print("Frozen G4A input identities: PASS")

    clean_path = repo / "predictions/heldout_events/event_test_clean_expert_predictions.csv.gz"
    wrong_path = repo / "predictions/heldout_events/event_test_wrongpre_siamese_predictions.csv.gz"
    donor_path = repo / "manifests/wrong_pre/test_hard_wrongpre_donor_map.csv.gz"

    clean = pd.read_csv(clean_path)
    wrong_pair = pd.read_csv(wrong_path)
    donor = pd.read_csv(donor_path)

    require(len(clean) == 115349, f"Clean TEST row count changed: {len(clean)}")
    require(clean["building_id"].astype(str).is_unique, "Clean TEST building IDs not unique.")
    require(len(wrong_pair) == 115269, f"Wrong-PRE row count changed: {len(wrong_pair)}")
    require(wrong_pair["building_id"].astype(str).is_unique, "Wrong-PRE building IDs not unique.")
    require(len(donor) == 115269, f"Wrong-PRE donor map row count changed: {len(donor)}")

    for col in ("target_building_id","donor_building_id"):
        require(col in donor.columns, f"Donor map missing {col}.")
    require(donor["target_building_id"].astype(str).is_unique, "Wrong-PRE donor targets not unique.")
    require(
        (donor["target_building_id"].astype(str) != donor["donor_building_id"].astype(str)).all(),
        "Wrong-PRE map contains self-donor."
    )

    clean_ids = set(clean["building_id"].astype(str))
    target_ids = donor["target_building_id"].astype(str)
    donor_ids = donor["donor_building_id"].astype(str)
    require(set(target_ids).issubset(clean_ids), "Wrong-PRE target outside clean TEST.")
    require(set(donor_ids).issubset(clean_ids), "Wrong-PRE donor outside clean TEST.")

    scene_by_id = clean.assign(building_id=clean["building_id"].astype(str)).set_index("building_id")["scene_id"].astype(str)
    require(
        (
            target_ids.map(scene_by_id).to_numpy()
            == donor_ids.map(scene_by_id).to_numpy()
        ).all(),
        "Wrong-PRE donor map contains cross-scene donor."
    )

    wrong_ids = wrong_pair["building_id"].astype(str).tolist()
    require(set(wrong_ids) == set(target_ids), "Wrong-PRE prediction IDs do not equal donor-map targets.")
    print("Wrong-PRE target/donor identity + same-scene safety: PASS")

    # Preserve prediction order as emitted by the frozen wrong-PRE inference.
    clean_by_id = clean.assign(building_id=clean["building_id"].astype(str)).set_index("building_id", drop=False)
    matched = clean_by_id.loc[wrong_ids].reset_index(drop=True)
    require(matched["building_id"].astype(str).tolist() == wrong_ids, "Matched-valid ordering failure.")

    wrong_frame = matched.copy()
    pair_cols = [
        "pred_pair","confidence_pair","entropy_pair","severity_pair",
        *[f"p{k}_pair" for k in range(4)],
        *[f"logit{k}_pair" for k in range(4)],
    ]
    for c in pair_cols:
        require(c in wrong_pair.columns, f"Wrong-PRE prediction table missing {c}")
        wrong_frame[c] = wrong_pair[c].to_numpy()

    logistic = joblib.load(repo / "checkpoints/routers/final_logistic_soft_gate.joblib")
    hgb = joblib.load(repo / "checkpoints/routers/final_hgb_hard_router.joblib")
    alpha_obj = json.loads((repo / "checkpoints/routers/final_static_alpha.json").read_text(encoding="utf-8"))
    require("alpha" in alpha_obj, "Static alpha artifact lacks 'alpha'.")
    alpha = float(alpha_obj["alpha"])
    require(close(alpha, 0.45, 1e-12), f"Static alpha changed: {alpha}")

    nb = torch.load(repo / "checkpoints/routers/final_neural_safe_router.pt", map_location="cpu", weights_only=False)
    require(int(nb.get("best_epoch",-1)) == 1, "Frozen neural best_epoch is not 1.")
    require(int(nb.get("expert_seed",-1)) == 42, "Frozen neural expert_seed is not 42.")
    require(list(nb["feature_names"]) == FEATURE_NAMES, "Frozen neural feature names changed.")

    neural = UtilityRouter(len(FEATURE_NAMES))
    neural.load_state_dict(nb["state_dict"], strict=True)
    neural.eval()
    mean = np.asarray(nb["mean"], np.float32)
    std = np.asarray(nb["std"], np.float32)
    require(mean.shape == (26,) and std.shape == (26,), "Neural normalization shape mismatch.")
    require((std > 0).all(), "Neural normalization std has non-positive values.")
    print("Frozen routers loaded without fitting: PASS")

    Pclean, Gclean = route_all(clean, logistic, hgb, neural, mean, std, alpha)
    Pvalid, Gvalid = route_all(matched, logistic, hgb, neural, mean, std, alpha)
    Pwrong, Gwrong = route_all(wrong_frame, logistic, hgb, neural, mean, std, alpha)

    regen_clean = metric_rows(clean, Pclean, "clean")
    regen_valid = metric_rows(matched, Pvalid, "valid_matched")
    regen_wrong = metric_rows(wrong_frame, Pwrong, "wrongpre_matched")
    regen_all = pd.concat([regen_clean, regen_valid, regen_wrong], ignore_index=True)

    canonical_clean = pd.read_csv(repo / "results/heldout_events/test_metrics_clean.csv")
    canonical_all = pd.read_csv(repo / "results/heldout_events/test_metrics_all_conditions.csv")

    clean_cmp = compare_metric_table(regen_clean, canonical_clean, ["clean"])
    matched_cmp = compare_metric_table(
        pd.concat([regen_valid, regen_wrong], ignore_index=True),
        canonical_all,
        ["valid_matched","wrongpre_matched"],
    )
    print("All 10 clean canonical route metrics: PASS")
    print("All 10 valid/wrong-PRE matched canonical route metrics: PASS")

    lookup_clean = regen_clean.set_index("method")
    lookup_wrong = regen_wrong.set_index("method")
    post_clean = float(lookup_clean.loc["POST-only","macro_f1"])
    neural_clean = float(lookup_clean.loc["Neural Temporal Utility Router","macro_f1"])
    neural_wrong = float(lookup_wrong.loc["Neural Temporal Utility Router","macro_f1"])

    require(close(post_clean, HEADLINE["post_clean_macro_f1"], HEADLINE_ATOL), "POST clean headline mismatch.")
    require(close(neural_clean, HEADLINE["neural_clean_macro_f1"], HEADLINE_ATOL), "Neural clean headline mismatch.")
    require(close(neural_clean-post_clean, HEADLINE["neural_clean_gain_vs_post"], HEADLINE_ATOL),
            "Neural clean gain headline mismatch.")
    require(close(neural_wrong, HEADLINE["neural_wrongpre_macro_f1"], HEADLINE_ATOL),
            "Neural wrong-PRE headline mismatch.")
    require(close(neural_wrong-post_clean, HEADLINE["neural_wrongpre_gain_vs_post"], HEADLINE_ATOL),
            "Neural wrong-PRE gain headline mismatch.")

    valid_gate_mean = float(np.mean(Gvalid["neural"]))
    wrong_gate_mean = float(np.mean(Gwrong["neural"]))
    require(close(valid_gate_mean, HEADLINE["valid_gate_mean_matched"], HEADLINE_ATOL),
            "Valid matched neural gate mean mismatch.")
    require(close(wrong_gate_mean, HEADLINE["wrong_gate_mean_matched"], HEADLINE_ATOL),
            "Wrong-PRE neural gate mean mismatch.")
    require(wrong_gate_mean < valid_gate_mean, "Gate suppression relation failed.")

    y = clean["true_label"].to_numpy(int)
    post_correct = clean["pred_post"].to_numpy(int) == y
    pair_correct = clean["pred_pair"].to_numpy(int) == y
    disc = post_correct != pair_correct
    q = pair_correct[disc].astype(int)
    gate_clean = np.asarray(Gclean["neural"], float)
    g = gate_clean[disc]

    pP = np.column_stack([clean[f"p{k}_post"].to_numpy(float) for k in range(4)])
    pT = np.column_stack([clean[f"p{k}_pair"].to_numpy(float) for k in range(4)])
    ii = np.arange(len(y))
    dce = np.log(np.clip(pT[ii,y],1e-12,1)) - np.log(np.clip(pP[ii,y],1e-12,1))

    utility_auc = float(roc_auc_score(q, g))
    utility_ap = float(average_precision_score(q, g))
    utility_rho = float(spearmanr(gate_clean, dce).statistic)

    require(close(utility_auc, HEADLINE["utility_auroc"], HEADLINE_ATOL), "Utility AUROC mismatch.")
    require(close(utility_ap, HEADLINE["utility_auprc"], HEADLINE_ATOL), "Utility AUPRC mismatch.")
    require(close(utility_rho, HEADLINE["gate_deltaCE_spearman"], SPEARMAN_ATOL), "Utility Spearman mismatch.")
    print("Frozen neural utility semantics: PASS")

    # Frozen paper conclusion must remain unchanged.
    results_index = json.loads((repo / "results/heldout_events/results_index.json").read_text(encoding="utf-8"))
    # Search recursively for an explicit boolean flag if present.
    def find_flag(obj, key):
        hits = []
        if isinstance(obj, dict):
            for k,v in obj.items():
                if str(k).lower() == key.lower():
                    hits.append(v)
                hits.extend(find_flag(v, key))
        elif isinstance(obj, list):
            for v in obj:
                hits.extend(find_flag(v, key))
        return hits
    flags = find_flag(results_index, "ALL_PRECOMMITTED_CRITERIA_PASS")
    if not flags:
        flags = find_flag(results_index, "all_precommitted_criteria_pass")
    require(flags, "Could not find frozen all-precommitted-criteria flag in results index.")
    require(all(v is False for v in flags), f"Frozen criteria flag changed: {flags}")

    report = {
        "stage": "G4B_deterministic_router_intervention_parity",
        "status": "PASS",
        "head": head,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "g4a_inventory_sha256": EXPECTED_G4A_INVENTORY_SHA256,
        "runtime_fix_sha256": EXPECTED_RUNTIME_FIX_SHA256,
        "canonical_metric_grouping": {
            "test_metrics_clean": (
                resolve_grouping_column(canonical_clean, ["clean"])
                if any(c in canonical_clean.columns for c in ("condition","population"))
                else None
            ),
            "test_metrics_all_conditions": resolve_grouping_column(
                canonical_all, ["valid_matched","wrongpre_matched"]
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "torch": torch.__version__,
        },
        "counts": {
            "clean_test": int(len(clean)),
            "wrongpre_matched": int(len(wrong_frame)),
            "correctness_discordant": int(disc.sum()),
        },
        "router_freeze": {
            "feature_count": 26,
            "feature_names": FEATURE_NAMES,
            "alpha": alpha,
            "neural_best_epoch": int(nb["best_epoch"]),
            "expert_seed": int(nb["expert_seed"]),
        },
        "headline_reproduced": {
            "post_clean_macro_f1": post_clean,
            "neural_clean_macro_f1": neural_clean,
            "neural_clean_gain_vs_post": neural_clean-post_clean,
            "neural_wrongpre_macro_f1": neural_wrong,
            "neural_wrongpre_gain_vs_post": neural_wrong-post_clean,
            "valid_gate_mean_matched": valid_gate_mean,
            "wrong_gate_mean_matched": wrong_gate_mean,
            "utility_auroc": utility_auc,
            "utility_auprc": utility_ap,
            "gate_deltaCE_spearman": utility_rho,
        },
        "canonical_metric_comparisons": clean_cmp + matched_cmp,
        "all_precommitted_criteria_pass": False,
        "training_performed": False,
        "router_fitting_performed": False,
        "expert_inference_performed": False,
        "raw_xbd_loaded": False,
        "post_test_tuning_performed": False,
        "next_gate": "G5 deterministic paper-artifact parity",
    }

    out_dir = repo / "audit/g4_router_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "g4b_router_intervention_parity_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    regen_path = out_dir / "g4b_regenerated_test_metrics.csv"
    regen_all.to_csv(regen_path, index=False)

    print("\nG4B ROUTER / INTERVENTION PARITY: PASS")
    print("POST clean Macro-F1     :", f"{post_clean:.12f}")
    print("Neural clean Macro-F1   :", f"{neural_clean:.12f}")
    print("Neural clean gain       :", f"{neural_clean-post_clean:+.12f}")
    print("Neural wrong-PRE F1     :", f"{neural_wrong:.12f}")
    print("Neural wrong-PRE gain   :", f"{neural_wrong-post_clean:+.12f}")
    print("Valid gate mean matched :", f"{valid_gate_mean:.12f}")
    print("Wrong gate mean matched :", f"{wrong_gate_mean:.12f}")
    print("Utility AUROC           :", f"{utility_auc:.12f}")
    print("Utility AUPRC           :", f"{utility_ap:.12f}")
    print("Gate-deltaCE Spearman   :", f"{utility_rho:.12f}")
    print("ALL_PRECOMMITTED_CRITERIA_PASS preserved: False")
    print("REPORT:", out)
    print("REGENERATED METRICS:", regen_path)
    print("REPORT SHA256:", sha256_file(out))

if __name__ == "__main__":
    main()
