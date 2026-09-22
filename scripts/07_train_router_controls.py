from __future__ import annotations
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from damageactu.routing.features import build_router_features
from damageactu.routing.logistic_router import fit_logistic_router
from damageactu.routing.hgb_router import fit_hgb_router


def mf1(y,p):
    return float(f1_score(y,p,labels=[0,1,2,3],average="macro",zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-development-table", required=True,
                    help="Table containing post/pair probabilities, predictions and derived confidence/entropy/severity columns.")
    ap.add_argument("--out", default="results/development/router_controls")
    args = ap.parse_args()


    df = pd.read_csv(args.aligned_development_table, low_memory=False)
    X = build_router_features(df)
    y = df.true_label.to_numpy(int)
    pc = df.pred_post.to_numpy(int) == y
    qc = df.pred_pair.to_numpy(int) == y
    disc = pc != qc
    target = qc[disc].astype(int)
    if len(np.unique(target)) != 2:
        raise RuntimeError("Discordant utility target must contain both classes.")


    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logistic = fit_logistic_router(X.loc[disc], target)
    hgb = fit_hgb_router(X.loc[disc], target)
    joblib.dump(logistic, out/"logistic.joblib")
    joblib.dump(hgb, out/"hgb.joblib")
    (out/"frozen_choices.json").write_text(json.dumps({"static_alpha":0.45,"neural_selected_epoch":1},indent=2))
    print("PASS: classical router controls fit with frozen hyperparameters.")
    print("NOTE: canonical neural-router training remains represented by the frozen Phase7D-C artifact.")


if __name__ == "__main__":
    main()
