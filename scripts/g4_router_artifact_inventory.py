#!/usr/bin/env python3
"""
DamageACT-U G4A router/intervention preflight.

Read-only scientific inventory. It deliberately does NOT:
- load xBD imagery,
- run expert inference,
- fit/refit routers,
- tune thresholds,
- modify canonical artifacts.

It writes only:
  audit/g4_router_intervention_inventory.json
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
import sklearn
import torch

EXPECTED_G3_CLOSEOUT_COMMIT = "8be7211"
EXPECTED_PROTOCOL_SHA256 = "4bd56f1a95fc0d7e0ac7259dac3c700a35a3faba2a48630b376010bd31e24f11"

EXPECTED_FEATURE_NAMES = [
    "post_p0","pair_p0","diff_p0","absdiff_p0",
    "post_p1","pair_p1","diff_p1","absdiff_p1",
    "post_p2","pair_p2","diff_p2","absdiff_p2",
    "post_p3","pair_p3","diff_p3","absdiff_p3",
    "confidence_post","confidence_pair","entropy_post","entropy_pair",
    "severity_post","severity_pair","conf_diff","entropy_diff",
    "severity_diff","pred_disagree",
]

REQUIRED = [
    "src/damageactu/routing/features.py",
    "checkpoints/routers/final_neural_safe_router.pt",
    "checkpoints/routers/final_logistic_soft_gate.joblib",
    "checkpoints/routers/final_hgb_hard_router.joblib",
    "checkpoints/routers/final_static_alpha.json",
    "manifests/wrong_pre/event_val_hard_wrongpre_donor_map.csv.gz",
    "manifests/wrong_pre/test_hard_wrongpre_donor_map.csv.gz",
    "predictions/development/event_val_post_only_predictions.csv.gz",
    "predictions/development/event_val_siamese_predictions.csv.gz",
    "predictions/development/event_val_siamese_wrongpre_predictions.csv.gz",
    "predictions/heldout_events/event_test_clean_expert_predictions.csv.gz",
    "predictions/heldout_events/event_test_wrongpre_siamese_predictions.csv.gz",
    "results/development/phase7d_c_results_index.json",
    "results/heldout_events/results_index.json",
    "results/heldout_events/test_neural_utility.json",
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()

def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)

def json_numeric_values(obj):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append((str(k), float(v)))
            else:
                for child_k, child_v in json_numeric_values(v):
                    out.append((f"{k}.{child_k}", child_v))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append((f"[{i}]", float(v)))
            else:
                for child_k, child_v in json_numeric_values(v):
                    out.append((f"[{i}].{child_k}", child_v))
    return out

def summarize_frame(path: Path):
    df = pd.read_csv(path)
    summary = {
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "column_count": int(len(df.columns)),
    }
    for candidate in ("building_id", "target_building_id", "target_id"):
        if candidate in df.columns:
            summary[f"{candidate}_unique"] = bool(df[candidate].astype(str).is_unique)
            summary[f"{candidate}_nunique"] = int(df[candidate].astype(str).nunique())
    for candidate in ("scene_id", "disaster", "true_label", "label_id"):
        if candidate in df.columns:
            summary[f"{candidate}_nunique"] = int(df[candidate].nunique(dropna=False))
    return summary

def public_object_metadata(obj):
    meta = {
        "python_type": f"{type(obj).__module__}.{type(obj).__name__}",
    }
    for attr in ("n_features_in_", "classes_", "feature_names_in_", "n_iter_"):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            meta[attr] = value
    params = None
    if hasattr(obj, "get_params"):
        try:
            params = obj.get_params(deep=False)
        except Exception:
            params = None
    if params is not None:
        safe = {}
        for k, v in params.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                safe[k] = v
            else:
                safe[k] = repr(v)
        meta["params"] = safe
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    require((repo / ".git").exists(), f"Not a Git repository root: {repo}")

    protocol_path = repo / "audit/g4_router_intervention_preflight_protocol.json"
    require(protocol_path.is_file(), f"Missing frozen G4A protocol: {protocol_path}")
    got_protocol_sha = sha256_file(protocol_path)
    require(
        got_protocol_sha == EXPECTED_PROTOCOL_SHA256,
        f"G4A protocol SHA mismatch: {got_protocol_sha}"
    )

    head = run_git(repo, "rev-parse", "HEAD")
    ancestor_rc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_G3_CLOSEOUT_COMMIT, head],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    require(ancestor_rc == 0, f"G3 closeout commit {EXPECTED_G3_CLOSEOUT_COMMIT} is not an ancestor of HEAD.")

    # Tracked-tree cleanliness only. Untracked restored evidence is intentionally allowed.
    require(subprocess.run(["git","diff","--quiet"], cwd=repo).returncode == 0,
            "Tracked unstaged modifications exist. Commit/revert them before G4A.")
    require(subprocess.run(["git","diff","--cached","--quiet"], cwd=repo).returncode == 0,
            "Staged modifications exist. Commit/unstage them before G4A.")

    print("=== G4A ROUTER / INTERVENTION PREFLIGHT ===")
    print("HEAD:", head)
    print("Protocol SHA256:", got_protocol_sha)
    print("Python:", sys.version.split()[0])
    print("numpy:", np.__version__)
    print("pandas:", pd.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("joblib:", joblib.__version__)
    print("torch:", torch.__version__)

    require(sys.version_info[:2] == (3, 12), "G4A requires Python 3.12.x.")
    require(sklearn.__version__ == "1.6.1", f"Expected scikit-learn 1.6.1; got {sklearn.__version__}")
    require(joblib.__version__ == "1.6.0", f"Expected joblib 1.6.0; got {joblib.__version__}")
    require(torch.__version__.split("+")[0] == "2.10.0", f"Expected torch 2.10.0; got {torch.__version__}")

    missing = [rel for rel in REQUIRED if not (repo / rel).is_file()]
    require(not missing, "Missing required G4A inputs:\n" + "\n".join(missing))
    print("Required inputs:", len(REQUIRED), "PASS")

    # Import the clean feature schema.
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from damageactu.routing.features import FEATURE_NAMES

    actual_features = list(FEATURE_NAMES)
    require(len(actual_features) == 26, f"FEATURE_NAMES count changed: {len(actual_features)}")
    require(actual_features == EXPECTED_FEATURE_NAMES, "FEATURE_NAMES order/content changed.")
    print("26-feature clean schema: PASS")

    neural_path = repo / "checkpoints/routers/final_neural_safe_router.pt"
    logistic_path = repo / "checkpoints/routers/final_logistic_soft_gate.joblib"
    hgb_path = repo / "checkpoints/routers/final_hgb_hard_router.joblib"
    alpha_path = repo / "checkpoints/routers/final_static_alpha.json"

    neural = torch.load(neural_path, map_location="cpu", weights_only=False)
    require(isinstance(neural, dict), "Neural router checkpoint is not a dict bundle.")
    require("feature_names" in neural, "Neural router checkpoint lacks feature_names.")
    require(list(neural["feature_names"]) == EXPECTED_FEATURE_NAMES,
            "Neural checkpoint feature_names mismatch.")
    print("Frozen neural feature schema: PASS")

    logistic = joblib.load(logistic_path)
    hgb = joblib.load(hgb_path)
    log_meta = public_object_metadata(logistic)
    hgb_meta = public_object_metadata(hgb)

    if "n_features_in_" in log_meta:
        require(int(log_meta["n_features_in_"]) == 26,
                f"Logistic router expects {log_meta['n_features_in_']} features, not 26.")
    if "n_features_in_" in hgb_meta:
        require(int(hgb_meta["n_features_in_"]) == 26,
                f"HGB router expects {hgb_meta['n_features_in_']} features, not 26.")
    print("Frozen logistic/HGB artifacts load: PASS")

    alpha_obj = json.loads(alpha_path.read_text(encoding="utf-8"))
    numeric_alpha = json_numeric_values(alpha_obj)
    alpha_hits = [(k, v) for k, v in numeric_alpha if abs(v - 0.45) <= 1e-12]
    require(alpha_hits, "Static-alpha artifact contains no numeric value equal to 0.45.")
    print("Frozen static alpha 0.45 present: PASS")

    tables = {}
    table_paths = [rel for rel in REQUIRED if rel.endswith(".csv.gz")]
    for rel in table_paths:
        p = repo / rel
        tables[rel] = summarize_frame(p)
        print(f"{rel}: rows={tables[rel]['rows']} cols={tables[rel]['column_count']}")

    val_donor = tables["manifests/wrong_pre/event_val_hard_wrongpre_donor_map.csv.gz"]
    require(val_donor["rows"] == 30698,
            f"Validation wrong-PRE donor rows changed: {val_donor['rows']}")
    print("Validation wrong-PRE donor count 30,698: PASS")

    hashes = {}
    for rel in REQUIRED:
        p = repo / rel
        hashes[rel] = {
            "sha256": sha256_file(p),
            "bytes": int(p.stat().st_size),
        }

    neural_meta = {
        "keys": sorted(map(str, neural.keys())),
        "feature_names": list(neural["feature_names"]),
    }
    for key in ("best_epoch", "epoch", "selected_epoch", "loss_weights"):
        if key in neural:
            value = neural[key]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            neural_meta[key] = value

    phase7c_index = json.loads((repo / "results/development/phase7d_c_results_index.json").read_text(encoding="utf-8"))
    test_index = json.loads((repo / "results/heldout_events/results_index.json").read_text(encoding="utf-8"))
    utility = json.loads((repo / "results/heldout_events/test_neural_utility.json").read_text(encoding="utf-8"))

    inventory = {
        "stage": "G4A_router_intervention_preflight",
        "status": "PASS",
        "head": head,
        "g3_closeout_commit_required": EXPECTED_G3_CLOSEOUT_COMMIT,
        "protocol_sha256": got_protocol_sha,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "torch": torch.__version__,
        },
        "feature_schema": {
            "count": len(actual_features),
            "names": actual_features,
        },
        "router_artifacts": {
            "neural": neural_meta,
            "logistic": log_meta,
            "hgb": hgb_meta,
            "static_alpha_numeric_values": numeric_alpha,
            "static_alpha_hits_0_45": alpha_hits,
        },
        "tables": tables,
        "input_hashes": hashes,
        "phase7d_c_results_index": phase7c_index,
        "heldout_results_index": test_index,
        "test_neural_utility": utility,
        "scientific_mutation_performed": False,
        "training_performed": False,
        "router_fitting_performed": False,
        "expert_inference_performed": False,
        "raw_xbd_loaded": False,
        "next_gate": "G4B deterministic router/intervention parity runner",
    }

    out = repo / "audit/g4_router_intervention_inventory.json"
    out.write_text(json.dumps(inventory, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("\nG4A PREFLIGHT: PASS")
    print("Inventory:", out)
    print("Inventory SHA256:", sha256_file(out))
    print("Next gate: freeze this inventory, then build G4B from the observed artifact schemas.")

if __name__ == "__main__":
    main()
