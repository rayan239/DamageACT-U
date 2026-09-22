from __future__ import annotations
from pathlib import Path
import argparse
import shutil
from damageactu.utils.hashing import raw_file_sha256


KNOWN = {
    "phase7d_frozen_unique_patch_roles.csv.gz": (
        "manifests/event_split/phase7d_frozen_unique_patch_roles.csv.gz",
        "7a396d0c61c6a78151a58b2c62dd016786ab1c4f9dd85f05eed5bde430ee4f05",
    ),
    "event_train_buildings.csv.gz": (
        "manifests/buildings/event_train_buildings.csv.gz",
        "c82b2743f857a11ed7b390a06b4b2b24e6bebdbd7e925585221c66ba5a60e9d4",
    ),
    "event_val_buildings.csv.gz": (
        "manifests/buildings/event_val_buildings.csv.gz",
        "44adcd20edcff627ef36ae9fbb7d259f0ca404ee86f343cea2313c6d3fefc35c",
    ),
    "event_val_hard_wrongpre_donor_map.csv.gz": (
        "manifests/wrong_pre/event_val_hard_wrongpre_donor_map.csv.gz",
        "4bfb7d73e5cec58a70977ce719cb32f33e15b6e8f06edf917392f218801905a6",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-vault", required=True)
    args = ap.parse_args()
    vault = Path(args.evidence_vault)
    if not vault.exists():
        raise FileNotFoundError(vault)
    for name, (dest, expected) in KNOWN.items():
        hits = [p for p in vault.rglob(name) if p.is_file()]
        if not hits:
            print("MISSING:", name)
            continue
        valid = [p for p in hits if raw_file_sha256(p) == expected]
        if len(valid) != 1:
            print("NOT RESTORED:", name, "valid_exact_copies=", len(valid))
            continue
        dst = Path(dest)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(valid[0], dst)
        print("RESTORED:", name, "->", dst)


if __name__ == "__main__":
    main()
