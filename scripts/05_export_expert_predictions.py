import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    for p in [args.checkpoint, args.manifest]:
        if not Path(p).exists():
            raise FileNotFoundError(p)
    raise SystemExit(
        "Inference export requires the exact restored frozen dataset/crop source and checkpoint schema. "
        "Use the archived Phase7D-B exporter until parity migration is completed."
    )


if __name__ == "__main__":
    main()
