from __future__ import annotations
from pathlib import Path
import gzip
import hashlib
import io
import pandas as pd


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw_file_sha256(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def logical_csv_bytes(path) -> bytes:
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    return raw


def logical_csv_sha256(path) -> str:
    return sha256_bytes(logical_csv_bytes(path))


def semantic_manifest_signature(df: pd.DataFrame, columns, sort_by=None) -> str:
    cols = list(columns)
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError("Missing semantic columns: " + str(sorted(missing)))
    x = df[cols].copy()
    for c in cols:
        x[c] = x[c].astype(str)
    keys = list(sort_by or cols)
    x = x.sort_values(keys, kind="stable").reset_index(drop=True)
    payload = x.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256_bytes(payload)
