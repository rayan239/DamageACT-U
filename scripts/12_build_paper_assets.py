from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import nn

FROZEN_COMMIT = "618ce84790c7e96efd0a7df3dea98f8b37b36cd9"
NEURAL = "Neural Temporal Utility Router"
LABELS = [0, 1, 2, 3]
EXPECTED = {
    "post_clean": 0.4565629212722525,
    "neural_clean": 0.4596737025324026,
    "neural_wrong": 0.45324001946153963,
    "clean_gain": 0.003110781260150075,
    "wrong_gain": -0.0030709805253712474,
    "auroc": 0.77283834848973,
    "auprc": 0.7196353859871121,
    "rho": 0.10360709395401074,
}

FEATURE_NAMES = [
    name
    for k in range(4)
    for name in (f"post_p{k}", f"pair_p{k}", f"diff_p{k}", f"absdiff_p{k}")
] + [
    "confidence_post","confidence_pair","entropy_post","entropy_pair",
    "severity_post","severity_pair","conf_diff","entropy_diff",
    "severity_diff","pred_disagree",
]

class TemporalUtilityRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(26, 32), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def close(name: str, got: float, expected: float, tol: float = 2e-7) -> None:
    if not np.isfinite(got) or abs(float(got)-float(expected)) > tol:
        raise RuntimeError(f"{name} drifted: got={got}, expected={expected}")

def features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for k in range(4):
        p = df[f"p{k}_post"].astype(float)
        q = df[f"p{k}_pair"].astype(float)
        x[f"post_p{k}"] = p
        x[f"pair_p{k}"] = q
        x[f"diff_p{k}"] = q-p
        x[f"absdiff_p{k}"] = (q-p).abs()
    for c in ["confidence_post","confidence_pair","entropy_post","entropy_pair",
              "severity_post","severity_pair"]:
        x[c] = df[c].astype(float)
    x["conf_diff"] = x["confidence_pair"]-x["confidence_post"]
    x["entropy_diff"] = x["entropy_pair"]-x["entropy_post"]
    x["severity_diff"] = x["severity_pair"]-x["severity_post"]
    x["pred_disagree"] = (
        df["pred_pair"].to_numpy(int) != df["pred_post"].to_numpy(int)
    ).astype(float)
    if list(x.columns) != FEATURE_NAMES:
        raise RuntimeError("26-feature schema drift.")
    return x

def gate(df: pd.DataFrame, artifact: dict) -> np.ndarray:
    x = features(df)
    if list(artifact["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("Frozen router feature order drift.")
    arr = x.to_numpy(np.float32)
    arr = (
        arr - np.asarray(artifact["mean"], dtype=np.float32)
    ) / np.asarray(artifact["std"], dtype=np.float32)
    model = TemporalUtilityRouter()
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(arr))).numpy()

def logits(df: pd.DataFrame, suffix: str) -> np.ndarray:
    return np.column_stack(
        [df[f"logit{k}_{suffix}"].to_numpy(float) for k in range(4)]
    )

def routed_pred(df: pd.DataFrame, u: np.ndarray) -> np.ndarray:
    lp, lt = logits(df, "post"), logits(df, "pair")
    return (lp + u[:,None]*(lt-lp)).argmax(axis=1)

def macro(y, pred) -> float:
    return float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0))

def savefig(fig, out: Path, name: str) -> None:
    for ext in ("png","pdf","svg"):
        fig.savefig(out/f"{name}.{ext}", dpi=320 if ext=="png" else None,
                    bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default="paper_assets_rendered")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    root = Path(a.repo_root).resolve()
    out = Path(a.out)
    if not out.is_absolute():
        out = root/out
    if out.exists():
        if not a.overwrite:
            raise RuntimeError(f"Output exists: {out}; use --overwrite.")
        shutil.rmtree(out)
    (out/"figures").mkdir(parents=True)
    (out/"tables").mkdir(parents=True)

    idx = json.loads(require(root/"results/heldout_events/results_index.json").read_text())
    if idx["post_test_tuning_permitted"] is not False:
        raise RuntimeError("post-TEST tuning guardrail changed")
    if idx["clean_effectiveness_pass"] is not False:
        raise RuntimeError("clean effectiveness must remain FAIL")
    if idx["all_precommitted_criteria_pass"] is not False:
        raise RuntimeError("ALL_PRECOMMITTED_CRITERIA_PASS must remain False")
    if int(idx["expert_seed"]) != 42:
        raise RuntimeError("final expert seed must remain 42")

    clean = pd.read_csv(require(root/"predictions/heldout_events/event_test_clean_expert_predictions.csv.gz"), low_memory=False)
    wrong = pd.read_csv(require(root/"predictions/heldout_events/event_test_wrongpre_siamese_predictions.csv.gz"), low_memory=False)
    artifact = torch.load(require(root/"checkpoints/routers/final_neural_safe_router.pt"), map_location="cpu", weights_only=False)
    if int(artifact["expert_seed"]) != 42 or int(artifact["best_epoch"]) != 1:
        raise RuntimeError("frozen neural-router identity drift")

    clean["building_id"] = clean["building_id"].astype(str)
    wrong["building_id"] = wrong["building_id"].astype(str)
    valid = clean.set_index("building_id", drop=False).loc[wrong["building_id"].tolist()].reset_index(drop=True)

    post_cols = ["building_id","scene_id","disaster","true_label","pred_post",
                 "confidence_post","entropy_post","severity_post"]
    pair_cols = ["building_id","pred_pair","confidence_pair","entropy_pair","severity_pair"]
    for k in range(4):
        post_cols += [f"p{k}_post",f"logit{k}_post"]
        pair_cols += [f"p{k}_pair",f"logit{k}_pair"]
    wrong_hybrid = valid[post_cols].merge(
        wrong[pair_cols], on="building_id", validate="one_to_one", sort=False
    )

    u_clean = gate(clean, artifact)
    u_valid = gate(valid, artifact)
    u_wrong = gate(wrong_hybrid, artifact)
    y = clean["true_label"].to_numpy(int)
    yv = valid["true_label"].to_numpy(int)

    post_clean = macro(y, clean["pred_post"].to_numpy(int))
    neural_clean = macro(y, routed_pred(clean, u_clean))
    post_wrong = macro(yv, wrong_hybrid["pred_post"].to_numpy(int))
    neural_wrong = macro(yv, routed_pred(wrong_hybrid, u_wrong))
    close("POST clean", post_clean, EXPECTED["post_clean"], 5e-12)
    close("neural clean", neural_clean, EXPECTED["neural_clean"], 5e-12)
    close("clean gain", neural_clean-post_clean, EXPECTED["clean_gain"], 5e-12)
    close("neural wrong-PRE", neural_wrong, EXPECTED["neural_wrong"], 5e-12)
    close("wrong-PRE gain", neural_wrong-post_wrong, EXPECTED["wrong_gain"], 5e-12)

    post_correct = clean["pred_post"].to_numpy(int) == y
    pair_correct = clean["pred_pair"].to_numpy(int) == y
    discordant = post_correct != pair_correct
    q = pair_correct[discordant].astype(int)
    pp = np.column_stack([clean[f"p{k}_post"].to_numpy(float) for k in range(4)])
    pt = np.column_stack([clean[f"p{k}_pair"].to_numpy(float) for k in range(4)])
    row = np.arange(len(y))
    delta_ce = np.log(np.clip(pt[row,y],1e-12,1.0)) - np.log(np.clip(pp[row,y],1e-12,1.0))
    auroc = float(roc_auc_score(q, u_clean[discordant]))
    auprc = float(average_precision_score(q, u_clean[discordant]))
    rho = float(spearmanr(u_clean, delta_ce).statistic)
    close("AUROC", auroc, EXPECTED["auroc"])
    close("AUPRC", auprc, EXPECTED["auprc"])
    close("Spearman", rho, EXPECTED["rho"])

    # Copy the committed source-first paper tables and diagrams.
    src = root/"paper_assets"
    shutil.copytree(src/"tables", out/"tables", dirs_exist_ok=True)
    for p in (src/"figures/main").glob("*.svg"):
        shutil.copy2(p, out/"figures"/p.name)

    # Regenerate exact utility ROC and PR curves from frozen per-building predictions.
    fpr,tpr,_ = roc_curve(q, u_clean[discordant])
    fig,ax=plt.subplots(figsize=(6.2,6.0))
    ax.plot(fpr,tpr,label=f"AUROC={auroc:.3f}")
    ax.plot([0,1],[0,1],"--",label="Random")
    ax.set(xlabel="False positive rate",ylabel="True positive rate",
           title="Temporal-utility ROC")
    ax.legend()
    savefig(fig,out/"figures","fig06a_temporal_utility_roc_exact")

    prec,rec,_ = precision_recall_curve(q, u_clean[discordant])
    fig,ax=plt.subplots(figsize=(6.2,6.0))
    ax.plot(rec,prec,label=f"AUPRC={auprc:.3f}")
    ax.axhline(q.mean(),ls="--",label=f"Prevalence={q.mean():.3f}")
    ax.set(xlabel="Recall",ylabel="Precision",
           title="Temporal-utility precision-recall")
    ax.legend()
    savefig(fig,out/"figures","fig06b_temporal_utility_pr_exact")

    # Exact normalized neural confusion matrices.
    for name,yy,pred in [
        ("figS06_neural_clean_confusion", y, routed_pred(clean,u_clean)),
        ("figS07_neural_wrongpre_confusion", yv, routed_pred(wrong_hybrid,u_wrong)),
    ]:
        cm=confusion_matrix(yy,pred,labels=LABELS,normalize="true")
        fig,ax=plt.subplots(figsize=(6.0,5.4))
        im=ax.imshow(cm,vmin=0,vmax=1)
        ax.set_xticks(range(4),["No","Minor","Major","Destroyed"])
        ax.set_yticks(range(4),["No","Minor","Major","Destroyed"])
        ax.set(xlabel="Predicted",ylabel="True",title=name.replace("_"," "))
        for i in range(4):
            for j in range(4):
                ax.text(j,i,f"{cm[i,j]:.2f}",ha="center",va="center",fontsize=8)
        fig.colorbar(im,ax=ax)
        savefig(fig,out/"figures",name)

    report={
        "status":"PASS",
        "scientific_source_commit":FROZEN_COMMIT,
        "expert_seed":42,
        "post_test_tuning_permitted":False,
        "clean_effectiveness_pass":False,
        "all_precommitted_criteria_pass":False,
        "recomputed":{
            "post_clean_macro_f1":post_clean,
            "neural_clean_macro_f1":neural_clean,
            "neural_minus_post_clean":neural_clean-post_clean,
            "neural_wrongpre_macro_f1":neural_wrong,
            "neural_minus_post_wrongpre":neural_wrong-post_wrong,
            "utility_auroc":auroc,
            "utility_auprc":auprc,
            "gate_deltaCE_spearman":rho,
            "valid_gate_mean":float(u_valid.mean(dtype=np.float64)),
            "wrong_gate_mean":float(u_wrong.mean(dtype=np.float64)),
        },
    }
    (out/"PAPER_ASSET_REBUILD_REPORT.json").write_text(json.dumps(report,indent=2))
    print("DAMAGEACT-U PAPER ASSET REBUILD: PASS")
    print(json.dumps(report["recomputed"],indent=2))

if __name__ == "__main__":
    main()
