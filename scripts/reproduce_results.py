from pathlib import Path
import json
import pandas as pd
from damageactu.evaluation.metrics import macro_f1


EXPECTED = {
    "post_macro_f1": 0.456563,
    "neural_macro_f1": 0.459674,
}


def find_candidate(root, names):
    root = Path(root)
    hits = []
    for name in names:
        hits.extend(root.rglob(name))
    hits = [p for p in hits if p.is_file()]
    return sorted(set(hits), key=str)


def main():
    pred_root = Path("predictions/heldout_events")
    combined = find_candidate(pred_root, ["heldout_predictions.csv.gz","heldout_predictions.csv"])
    report = {"status":"INCOMPLETE","checks":{},"expected":EXPECTED}
    if not combined:
        report["message"] = (
            "Restore the canonical held-out prediction table under predictions/heldout_events. "
            "The repository intentionally does not fabricate missing paper artifacts."
        )
        Path("results").mkdir(exist_ok=True)
        Path("results/reproduction_report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report, indent=2))
        return
    if len(combined) != 1:
        raise RuntimeError("Multiple held-out prediction tables found; remove duplicates.")
    df = pd.read_csv(combined[0], low_memory=False)
    required = {"true_label","pred_post","pred_neural"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError("Prediction table missing: " + str(sorted(missing)))
    got = {
        "post_macro_f1": macro_f1(df.true_label, df.pred_post),
        "neural_macro_f1": macro_f1(df.true_label, df.pred_neural),
    }
    checks = {k: abs(got[k]-EXPECTED[k]) <= 5e-6 for k in EXPECTED}
    report = {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"expected":EXPECTED,"reproduced":got}
    Path("results").mkdir(exist_ok=True)
    Path("results/reproduction_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
