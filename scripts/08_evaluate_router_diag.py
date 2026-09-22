import argparse
import pandas as pd
from damageactu.evaluation.event_analysis import summarize_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--pred-col", default="pred_neural")
    args = ap.parse_args()
    df = pd.read_csv(args.predictions, low_memory=False)
    pooled, per_event = summarize_predictions(df, args.pred_col)
    print(pooled)
    print(per_event.to_string(index=False))


if __name__ == "__main__":
    main()
