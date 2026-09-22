import argparse
import pandas as pd
from damageactu.interventions.wrong_pre import validate_donor_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", required=True)
    ap.add_argument("--donor-map", required=True)
    args = ap.parse_args()
    b = pd.read_csv(args.buildings, low_memory=False)
    d = pd.read_csv(args.donor_map, low_memory=False)
    validate_donor_map(d,b)
    print("PASS: donor map is same-scene and target != donor for all rows.")
    print("Rows:", len(d))


if __name__ == "__main__":
    main()
