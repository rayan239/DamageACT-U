from pathlib import Path
import json
import pandas as pd


PUBLISHED = {
    "post_macro_f1": 0.456563,
    "neural_macro_f1": 0.459674,
    "neural_minus_post": 0.003111,
    "utility_auroc": 0.7728,
    "utility_spearman": 0.1036,
}


def main():
    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([PUBLISHED]).to_csv(out/"published_headline_results.csv", index=False)
    Path("results/reproduction_report.json").write_text(
        json.dumps({
            "status":"REFERENCE_VALUES_WRITTEN",
            "note":"Exact regenerated equality requires restored final prediction artifacts.",
            "published":PUBLISHED,
        }, indent=2),
        encoding="utf-8"
    )
    print("WROTE paper reference table and reproduction-report template.")


if __name__ == "__main__":
    main()
