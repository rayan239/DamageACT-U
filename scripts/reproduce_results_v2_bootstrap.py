from __future__ import annotations

"""
DamageACT-U Phase7D-D statistical reproduction: scene-bootstrap + utility-bootstrap parity.

This is the second reproduction layer. It first requires the deterministic core
reproduction to pass, then independently regenerates the frozen bootstrap
statistics from the saved expert predictions and frozen routers.

Performance bootstrap:
  - same scene IDs, same NumPy RNG, same seeds, same number of replicates as
    the sealed Phase7D-D notebook;
  - uses per-scene 4x4 confusion matrices as an exact computational shortcut
    for repeatedly expanding 100k+ building rows. Macro-F1 is computed from
    the same integer counts, so this is mathematically identical to the sealed
    row-expansion calculation.

Utility bootstrap:
  - same scene draws and RNG seed as the sealed notebook;
  - scene multiplicities are represented as integer sample weights rather than
    physically duplicating rows for AUROC/AUPRC;
  - Spearman uses weighted average ranks exactly corresponding to the expanded
    multiset, up to ordinary floating-point roundoff.

No model fitting, threshold selection, TEST-dependent tuning, or artifact
mutation occurs in this script.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

# scripts/ is automatically on sys.path when this file is executed directly.
import reproduce_results_v2_core as core

ROOT = core.ROOT
FROZEN_BOOT_PATH = ROOT / "results" / "heldout_events" / "test_scene_bootstrap_all.csv"
FROZEN_UTILITY_PATH = ROOT / "results" / "heldout_events" / "test_neural_utility.json"
FROZEN_INDEX_PATH = ROOT / "results" / "heldout_events" / "results_index.json"
FROZEN_CLASS_PATH = ROOT / "results" / "heldout_events" / "test_per_class_f1_all_conditions.csv"

OUT_JSON = ROOT / "results" / "reproduction_bootstrap_v2.json"
OUT_BOOT_CSV = ROOT / "results" / "reproduction_scene_bootstrap_v2.csv"
OUT_UTILITY_JSON = ROOT / "results" / "reproduction_utility_bootstrap_v2.json"

EXPECTED_SKLEARN = "1.6.1"
PERF_TOL = 5e-12
UTILITY_TOL = 3e-7
CLASS_IDS = [0, 1, 2, 3]


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")


def assert_close(name: str, got: float, expected: float, tol: float) -> None:
    if not np.isfinite(got) or abs(float(got) - float(expected)) > tol:
        raise RuntimeError(
            f"BOOTSTRAP PARITY FAIL {name}: got={got:.16g}, "
            f"expected={expected:.16g}, tol={tol}"
        )


def _scene_confusions(
    y: np.ndarray,
    pred: np.ndarray,
    scene_codes: np.ndarray,
    n_scenes: int,
) -> np.ndarray:
    """Per-scene 4x4 confusion matrices flattened to 16 integer cells."""
    out = np.zeros((n_scenes, 16), dtype=np.int64)
    flat = y.astype(np.int64) * 4 + pred.astype(np.int64)
    np.add.at(out, (scene_codes, flat), 1)
    return out


def _macro_f1_from_flat_confusion(flat: np.ndarray) -> float:
    """Fixed-4-class Macro-F1 from one flattened 4x4 confusion matrix."""
    cm = np.asarray(flat, dtype=np.int64).reshape(4, 4)
    tp = np.diag(cm).astype(float)
    row_sum = cm.sum(axis=1).astype(float)
    col_sum = cm.sum(axis=0).astype(float)
    denom = row_sum + col_sum
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(4, dtype=float), where=denom != 0)
    return float(f1.mean())


def paired_scene_bootstrap_fast(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    reps: int,
    seed: int,
    condition: str,
) -> pd.DataFrame:
    """
    Exact scene-cluster bootstrap equivalent to the sealed notebook.

    The RNG calls intentionally match:
        rng.choice(unique_scenes, size=len(unique_scenes), replace=True)
    replicate-by-replicate.
    """
    y = frame["true_label"].to_numpy(int)
    scenes = frame["scene_id"].astype(str).to_numpy()
    unique_scenes = np.unique(scenes)
    scene_codes = np.searchsorted(unique_scenes, scenes)
    n_scenes = len(unique_scenes)

    methods = [
        m for m in predictions
        if m not in ("POST-only", "Oracle expert selector")
    ]

    scene_cm: dict[str, np.ndarray] = {
        "POST-only": _scene_confusions(
            y, np.asarray(predictions["POST-only"], dtype=int), scene_codes, n_scenes
        )
    }
    for method in methods:
        scene_cm[method] = _scene_confusions(
            y, np.asarray(predictions[method], dtype=int), scene_codes, n_scenes
        )

    values = {m: np.empty(reps, dtype=float) for m in methods}
    rng = np.random.default_rng(seed)

    for r in range(reps):
        # Same draw semantics and order as the sealed notebook.
        draw = rng.choice(unique_scenes, size=n_scenes, replace=True)
        counts = np.bincount(
            np.searchsorted(unique_scenes, draw), minlength=n_scenes
        ).astype(np.int64)

        post_f1 = _macro_f1_from_flat_confusion(counts @ scene_cm["POST-only"])
        for method in methods:
            method_f1 = _macro_f1_from_flat_confusion(counts @ scene_cm[method])
            values[method][r] = method_f1 - post_f1

    rows = []
    for method in methods:
        a = values[method]
        low, high = np.percentile(a, [2.5, 97.5])
        rows.append(
            {
                "method": method,
                "reps": int(len(a)),
                "mean_gain_vs_post": float(a.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "condition": condition,
            }
        )
    return pd.DataFrame(rows)


def _rank_group_ids(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Precompute fixed tie groups for weighted average-rank Spearman."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    start = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    gid_sorted = np.cumsum(start) - 1
    gid = np.empty(len(values), dtype=np.int64)
    gid[order] = gid_sorted
    return gid, int(gid_sorted[-1] + 1)


def _weighted_spearman_from_expansion(
    weights: np.ndarray,
    x_group: np.ndarray,
    n_x_groups: int,
    y_group: np.ndarray,
    n_y_groups: int,
) -> float:
    """
    Spearman correlation of an integer-weighted expanded multiset without
    physically duplicating observations.

    Integer weights are the number of times each scene was drawn.
    """
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 1:
        return float("nan")

    wx = np.bincount(x_group, weights=w, minlength=n_x_groups)
    wy = np.bincount(y_group, weights=w, minlength=n_y_groups)

    # Average 1-based rank of each tie group in the expanded multiset.
    rx_group = np.cumsum(wx) - wx + (wx + 1.0) / 2.0
    ry_group = np.cumsum(wy) - wy + (wy + 1.0) / 2.0
    rx = rx_group[x_group]
    ry = ry_group[y_group]

    mean_x = float(np.dot(w, rx) / total)
    mean_y = float(np.dot(w, ry) / total)
    dx = rx - mean_x
    dy = ry - mean_y
    cov = float(np.dot(w, dx * dy))
    var_x = float(np.dot(w, dx * dx))
    var_y = float(np.dot(w, dy * dy))
    if var_x <= 0.0 or var_y <= 0.0:
        return float("nan")
    return float(cov / np.sqrt(var_x * var_y))


def utility_scene_bootstrap_fast(
    clean: pd.DataFrame,
    neural_gate: np.ndarray,
    reps: int = 1000,
    seed: int = 20260917,
) -> dict:
    """Reproduce the sealed scene-bootstrap utility statistics."""
    y = clean["true_label"].to_numpy(int)
    post_pred = clean["pred_post"].to_numpy(int)
    pair_pred = clean["pred_pair"].to_numpy(int)
    post_correct = post_pred == y
    pair_correct = pair_pred == y
    discordant = post_correct != pair_correct
    q = pair_correct[discordant].astype(int)

    post_prob = core.probabilities(clean, "post")
    pair_prob = core.probabilities(clean, "pair")
    row = np.arange(len(y))
    dce = (
        np.log(np.clip(pair_prob[row, y], 1e-12, 1.0))
        - np.log(np.clip(post_prob[row, y], 1e-12, 1.0))
    )

    scenes = clean["scene_id"].astype(str).to_numpy()
    unique_scenes = np.unique(scenes)
    scene_codes = np.searchsorted(unique_scenes, scenes)
    n_scenes = len(unique_scenes)
    discordant_scene_codes = scene_codes[discordant]

    gate = np.asarray(neural_gate, dtype=float)
    x_group, n_x_groups = _rank_group_ids(gate)
    y_group, n_y_groups = _rank_group_ids(dce)

    auroc_vals: list[float] = []
    auprc_vals: list[float] = []
    spearman_vals: list[float] = []
    rng = np.random.default_rng(seed)

    for _ in range(reps):
        # Same scene draws, same RNG sequence as the sealed notebook.
        draw = rng.choice(unique_scenes, size=n_scenes, replace=True)
        scene_counts = np.bincount(
            np.searchsorted(unique_scenes, draw), minlength=n_scenes
        ).astype(np.int64)
        weights = scene_counts[scene_codes].astype(float)
        dweights = scene_counts[discordant_scene_codes].astype(float)

        # Duplicating scene rows is exactly equivalent to integer sample weights.
        # Keep a replicate only when both q classes have positive expanded weight.
        if dweights[q == 0].sum() > 0 and dweights[q == 1].sum() > 0:
            auroc_vals.append(
                float(
                    roc_auc_score(
                        q,
                        gate[discordant],
                        sample_weight=dweights,
                    )
                )
            )
            auprc_vals.append(
                float(
                    average_precision_score(
                        q,
                        gate[discordant],
                        sample_weight=dweights,
                    )
                )
            )

        rho = _weighted_spearman_from_expansion(
            weights, x_group, n_x_groups, y_group, n_y_groups
        )
        if np.isfinite(rho):
            spearman_vals.append(float(rho))

    def summary(values: list[float]) -> dict:
        a = np.asarray(values, dtype=float)
        if len(a) == 0:
            return {
                "mean": None,
                "ci95_low": None,
                "ci95_high": None,
                "valid_reps": 0,
            }
        low, high = np.percentile(a, [2.5, 97.5])
        return {
            "mean": float(a.mean()),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "valid_reps": int(len(a)),
        }

    return {
        "auroc_bootstrap": summary(auroc_vals),
        "auprc_bootstrap": summary(auprc_vals),
        "spearman_bootstrap": summary(spearman_vals),
    }


def _compare_bootstrap_table(got: pd.DataFrame, frozen: pd.DataFrame) -> None:
    key_cols = ["condition", "method"]
    got_idx = got.set_index(key_cols).sort_index()
    ref_idx = frozen.set_index(key_cols).sort_index()
    if list(got_idx.index) != list(ref_idx.index):
        raise RuntimeError("Frozen/reproduced performance-bootstrap row identities differ.")

    for key in got_idx.index:
        g = got_idx.loc[key]
        r = ref_idx.loc[key]
        if int(g["reps"]) != int(r["reps"]):
            raise RuntimeError(f"Bootstrap replicate count mismatch for {key}.")
        for col in ["mean_gain_vs_post", "ci95_low", "ci95_high"]:
            assert_close(
                f"performance/{key[0]}/{key[1]}/{col}",
                float(g[col]),
                float(r[col]),
                PERF_TOL,
            )


def _compare_utility(got: dict, frozen: dict) -> None:
    for block in ["auroc_bootstrap", "auprc_bootstrap", "spearman_bootstrap"]:
        g = got[block]
        r = frozen[block]
        if int(g["valid_reps"]) != int(r["valid_reps"]):
            raise RuntimeError(f"Utility bootstrap valid_reps mismatch: {block}")
        for col in ["mean", "ci95_low", "ci95_high"]:
            assert_close(
                f"utility/{block}/{col}",
                float(g[col]),
                float(r[col]),
                UTILITY_TOL,
            )


def main() -> None:
    print("DamageACT-U statistical reproduction — bootstrap layer")
    print("Python:", sys.version.split()[0])
    print("scikit-learn:", sklearn.__version__)
    print("torch:", torch.__version__)
    if sklearn.__version__ != EXPECTED_SKLEARN:
        raise RuntimeError(
            f"Strict frozen-router compatibility requires scikit-learn "
            f"{EXPECTED_SKLEARN}; found {sklearn.__version__}."
        )

    for p in [
        core.CLEAN_PATH,
        core.WRONG_PATH,
        core.LOGISTIC_PATH,
        core.HGB_PATH,
        core.NEURAL_PATH,
        core.ALPHA_PATH,
        FROZEN_BOOT_PATH,
        FROZEN_UTILITY_PATH,
        FROZEN_INDEX_PATH,
        FROZEN_CLASS_PATH,
    ]:
        require(p)

    # Gate 0: deterministic reconstruction must still pass before bootstrap work.
    print("\n[1/4] Re-running deterministic core parity gate...")
    core.main()

    print("\n[2/4] Reconstructing frozen TEST methods...")
    clean = pd.read_csv(core.CLEAN_PATH, low_memory=False)
    wrong = pd.read_csv(core.WRONG_PATH, low_memory=False)
    alpha = float(json.loads(core.ALPHA_PATH.read_text(encoding="utf-8"))["alpha"])
    logistic = joblib.load(core.LOGISTIC_PATH)
    hgb = joblib.load(core.HGB_PATH)
    neural_artifact = torch.load(core.NEURAL_PATH, map_location="cpu", weights_only=False)

    valid, wrong_hybrid = core.align_valid_and_wrong(clean, wrong)
    clean_pred, clean_gate = core.build_methods(
        clean, alpha, logistic, hgb, neural_artifact
    )
    valid_pred, valid_gate = core.build_methods(
        valid, alpha, logistic, hgb, neural_artifact
    )
    wrong_pred, wrong_gate = core.build_methods(
        wrong_hybrid, alpha, logistic, hgb, neural_artifact
    )

    print("[3/4] Reproducing 2,000-replicate clean/wrong-PRE scene bootstraps...")
    boot_clean = paired_scene_bootstrap_fast(
        clean, clean_pred, reps=2000, seed=20260917, condition="clean"
    )
    boot_wrong = paired_scene_bootstrap_fast(
        wrong_hybrid,
        wrong_pred,
        reps=2000,
        seed=20260918,
        condition="wrongpre",
    )
    boot_all = pd.concat([boot_clean, boot_wrong], ignore_index=True)
    frozen_boot = pd.read_csv(FROZEN_BOOT_PATH)
    _compare_bootstrap_table(boot_all, frozen_boot)

    print("[4/4] Reproducing 1,000-replicate temporal-utility scene bootstrap...")
    utility_boot = utility_scene_bootstrap_fast(
        clean, clean_gate["neural"], reps=1000, seed=20260917
    )
    frozen_utility = json.loads(FROZEN_UTILITY_PATH.read_text(encoding="utf-8"))
    _compare_utility(utility_boot, frozen_utility)

    # Re-evaluate the precommitted decision criteria from reproduced values.
    frozen_index = json.loads(FROZEN_INDEX_PATH.read_text(encoding="utf-8"))
    frozen_class = pd.read_csv(FROZEN_CLASS_PATH)

    neural_name = "Neural Temporal Utility Router"
    clean_metrics = {
        m: core.metric_row(clean["true_label"].to_numpy(int), p)["macro_f1"]
        for m, p in clean_pred.items()
    }
    wrong_metrics = {
        m: core.metric_row(wrong_hybrid["true_label"].to_numpy(int), p)["macro_f1"]
        for m, p in wrong_pred.items()
    }
    clean_gain = float(clean_metrics[neural_name] - clean_metrics["POST-only"])
    wrong_gain = float(wrong_metrics[neural_name] - wrong_metrics["POST-only"])

    bclean = boot_clean.set_index("method")
    bwrong = boot_wrong.set_index("method")
    clean_low = float(bclean.loc[neural_name, "ci95_low"])
    wrong_low = float(bwrong.loc[neural_name, "ci95_low"])

    pc = frozen_class[frozen_class["population"] == "clean_pooled"]
    fn = pc[pc["method"] == neural_name].set_index("class_id")["f1"]
    fp = pc[pc["method"] == "POST-only"].set_index("class_id")["f1"]
    worst_class_delta = float((fn - fp).min())

    valid_gate_mean = float(np.mean(valid_gate["neural"]))
    wrong_gate_mean = float(np.mean(wrong_gate["neural"]))

    clean_pass = bool(clean_gain >= 0.010 and clean_low > 0.0)
    class_pass = bool(worst_class_delta >= -0.020)
    wrong_pass = bool(wrong_gain >= -0.005 and wrong_low >= -0.010)
    gate_pass = bool(wrong_gate_mean < valid_gate_mean)
    utility_pass = bool(
        utility_boot["auroc_bootstrap"]["ci95_low"] > 0.5
        and utility_boot["spearman_bootstrap"]["ci95_low"] > 0.0
    )
    all_pass = bool(
        clean_pass and class_pass and wrong_pass and gate_pass and utility_pass
    )

    reproduced_decisions = {
        "clean_effectiveness_pass": clean_pass,
        "class_safety_pass": class_pass,
        "wrongpre_safety_pass": wrong_pass,
        "gate_suppression_pass": gate_pass,
        "utility_generalization_pass": utility_pass,
        "all_precommitted_criteria_pass": all_pass,
    }
    for key, got in reproduced_decisions.items():
        expected = bool(frozen_index[key])
        if bool(got) != expected:
            raise RuntimeError(
                f"Decision parity FAIL {key}: got={got}, frozen={expected}"
            )

    # Cross-check the headline CI values explicitly stored in results_index.json.
    assert_close(
        "decision/primary_clean_bootstrap_ci95_low",
        clean_low,
        float(frozen_index["primary_clean_bootstrap_ci95_low"]),
        PERF_TOL,
    )
    assert_close(
        "decision/wrongpre_bootstrap_ci95_low",
        wrong_low,
        float(frozen_index["wrongpre_bootstrap_ci95_low"]),
        PERF_TOL,
    )
    assert_close(
        "decision/utility_auroc_bootstrap_ci95_low",
        float(utility_boot["auroc_bootstrap"]["ci95_low"]),
        float(frozen_index["utility_auroc_bootstrap_ci95_low"]),
        UTILITY_TOL,
    )
    assert_close(
        "decision/utility_spearman_bootstrap_ci95_low",
        float(utility_boot["spearman_bootstrap"]["ci95_low"]),
        float(frozen_index["utility_spearman_bootstrap_ci95_low"]),
        UTILITY_TOL,
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    boot_all.to_csv(OUT_BOOT_CSV, index=False)
    OUT_UTILITY_JSON.write_text(
        json.dumps(utility_boot, indent=2), encoding="utf-8"
    )

    report = {
        "status": "PASS",
        "scope": (
            "full Phase7D-D statistical bootstrap reproduction from frozen expert "
            "predictions and frozen routers; no training or TEST-dependent tuning"
        ),
        "environment": {
            "python": sys.version,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "bootstrap_protocol": {
            "clean_performance_reps": 2000,
            "clean_performance_seed": 20260917,
            "wrongpre_performance_reps": 2000,
            "wrongpre_performance_seed": 20260918,
            "utility_reps": 1000,
            "utility_seed": 20260917,
            "cluster_unit": "scene_id",
        },
        "reproduced_headline_ci": {
            "clean_neural_minus_post": clean_gain,
            "clean_neural_minus_post_ci95_low": clean_low,
            "clean_neural_minus_post_ci95_high": float(
                bclean.loc[neural_name, "ci95_high"]
            ),
            "wrongpre_neural_minus_post": wrong_gain,
            "wrongpre_neural_minus_post_ci95_low": wrong_low,
            "wrongpre_neural_minus_post_ci95_high": float(
                bwrong.loc[neural_name, "ci95_high"]
            ),
            "utility_auroc_ci95_low": float(
                utility_boot["auroc_bootstrap"]["ci95_low"]
            ),
            "utility_auroc_ci95_high": float(
                utility_boot["auroc_bootstrap"]["ci95_high"]
            ),
            "utility_spearman_ci95_low": float(
                utility_boot["spearman_bootstrap"]["ci95_low"]
            ),
            "utility_spearman_ci95_high": float(
                utility_boot["spearman_bootstrap"]["ci95_high"]
            ),
        },
        "reproduced_decisions": reproduced_decisions,
        "guardrails": {
            "expert_seed": 42,
            "random_seed_robustness_evaluated": False,
            "post_test_tuning_permitted": False,
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFULL BOOTSTRAP REPRODUCTION PASS")
    print(
        "Clean neural-POST 95% CI : "
        f"[{bclean.loc[neural_name, 'ci95_low']:.12f}, "
        f"{bclean.loc[neural_name, 'ci95_high']:.12f}]"
    )
    print(
        "Wrong neural-POST 95% CI : "
        f"[{bwrong.loc[neural_name, 'ci95_low']:.12f}, "
        f"{bwrong.loc[neural_name, 'ci95_high']:.12f}]"
    )
    print(
        "Utility AUROC 95% CI      : "
        f"[{utility_boot['auroc_bootstrap']['ci95_low']:.12f}, "
        f"{utility_boot['auroc_bootstrap']['ci95_high']:.12f}]"
    )
    print(
        "Utility Spearman 95% CI   : "
        f"[{utility_boot['spearman_bootstrap']['ci95_low']:.12f}, "
        f"{utility_boot['spearman_bootstrap']['ci95_high']:.12f}]"
    )
    print("Criteria:")
    print("  clean effectiveness   :", "PASS" if clean_pass else "FAIL")
    print("  class safety          :", "PASS" if class_pass else "FAIL")
    print("  wrong-PRE safety      :", "PASS" if wrong_pass else "FAIL")
    print("  gate suppression      :", "PASS" if gate_pass else "FAIL")
    print("  utility generalization:", "PASS" if utility_pass else "FAIL")
    print("ALL_PRECOMMITTED_CRITERIA_PASS:", all_pass)
    print("REPORT:", OUT_JSON)
    print("BOOTSTRAP TABLE:", OUT_BOOT_CSV)
    print("UTILITY BOOTSTRAP:", OUT_UTILITY_JSON)


if __name__ == "__main__":
    main()
