import argparse
from pathlib import Path
from damageactu.utils.frozen_source import verify_frozen_source


MESSAGE = """
Exact Phase7D-A building-manifest generation must use the hash-verified original
build_manifest.py/crop_utils.py snapshot from the evidence vault. This wrapper
verifies that snapshot before any rebuild. The public release should preserve
the frozen manifests themselves so paper-number reproduction never depends on
silently reimplementing geometry.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-project-root", required=True)
    args = ap.parse_args()
    mapping = verify_frozen_source(args.frozen_project_root)
    print(MESSAGE.strip())
    print("PASS: exact frozen source snapshot verified:")
    for k,v in mapping.items():
        print(" ", k, "->", v)
    print("Next: execute the archived Phase7D-A build entry point or migrate it only after parity tests are available.")


if __name__ == "__main__":
    main()
