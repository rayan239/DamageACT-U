from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


ORIGINAL_SUBSETS = ["train","tier3","test","hold"]


def flatten_hafner_metadata(metadata_json):
    obj = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    rows = []
    for subset in ORIGINAL_SUBSETS:
        part = obj.get(subset, {})
        patches = part.get("patches", [])
        for p in patches:
            if not isinstance(p, dict):
                continue
            row = dict(p)
            row["subset"] = subset
            event = str(row.get("event", ""))
            patch_id = str(row.get("patch_id", ""))
            row["pair_key"] = event + "_" + patch_id
            rows.append(row)
    return pd.DataFrame(rows)


def find_xbd_root(root):
    root = Path(root)
    candidates = [root, root/"xbd_full", root/"xbd"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(root)
