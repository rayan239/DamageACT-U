from pathlib import Path
import csv
from damageactu.utils.reproducibility import build_artifact_manifest


def main():
    root = Path(".")
    rows = build_artifact_manifest(root)
    out = Path("audit/artifact_manifest.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path","size_bytes","raw_sha256"])
        w.writeheader()
        w.writerows(rows)
    print("WROTE", out, "rows=", len(rows))


if __name__ == "__main__":
    main()
