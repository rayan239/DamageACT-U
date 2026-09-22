import argparse
from pathlib import Path
from damageactu.data.xbd import flatten_hafner_metadata
from damageactu.data.event_split import build_event_split
from damageactu.utils.io import write_deterministic_gzip_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-json", required=True)
    ap.add_argument("--output", default="manifests/event_split/patch_roles_reconstructed.csv.gz")
    args = ap.parse_args()
    meta = flatten_hafner_metadata(args.metadata_json)
    split = build_event_split(meta)
    out = Path(args.output)
    write_deterministic_gzip_csv(split, out)
    print(split["event_role"].value_counts().to_string())
    print("WROTE:", out)


if __name__ == "__main__":
    main()
