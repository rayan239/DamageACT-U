import argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd-root", required=True)
    args = ap.parse_args()
    print("xBD root:", args.xbd_root)
    raise SystemExit(
        "Inference reproduction requires the exact canonical expert/router checkpoints and frozen source snapshot. "
        "Restore them using scripts/restore_canonical_artifacts.py, then parity-port the archived inference runner."
    )
if __name__ == "__main__":
    main()
