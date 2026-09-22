import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd-root", required=True)
    args = ap.parse_args()
    root = Path(args.xbd_root)
    if not root.exists():
        raise FileNotFoundError(root)
    png = sum(1 for _ in root.rglob("*.png"))
    js = sum(1 for _ in root.rglob("*.json"))
    print("xBD root:", root.resolve())
    print("PNG files:", png)
    print("JSON files:", js)
    if png == 0 or js == 0:
        raise RuntimeError("xBD audit found no PNG or JSON files.")
    print("PASS: basic xBD presence audit. For paper release, also preserve the original Phase7C v3 integrity audit.")


if __name__ == "__main__":
    main()
