import argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd-root", required=True)
    args = ap.parse_args()
    print("xBD root:", args.xbd_root)
    raise SystemExit(
        "Full retraining is intentionally guarded in RC1. Exact paper reproduction should use frozen predictions. "
        "Full replication should run only after the archived Phase7D-B trainer has been parity-ported and verified."
    )
if __name__ == "__main__":
    main()
