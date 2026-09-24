from __future__ import annotations
import pandas as pd


EXPECTED_TRAIN_ROWS = 265273
EXPECTED_VAL_ROWS = 30735
EXPECTED_CLASS_COUNTS = {0:212466, 1:15518, 2:19528, 3:17761}


def validate_building_manifest(df: pd.DataFrame, role=None):
    if "building_id" not in df.columns:
        raise ValueError("Missing building_id.")
    if df["building_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate building_id.")
    if "scene_id" not in df.columns:
        raise ValueError("Missing scene_id.")
    if role == "train" and len(df) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(f"Train building count changed: {len(df)}")
    if role == "val" and len(df) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Validation building count changed: {len(df)}")
    return True


def validate_train_val_no_overlap(train_df, val_df):
    a = set(train_df["building_id"].astype(str))
    b = set(val_df["building_id"].astype(str))
    if a & b:
        raise RuntimeError("Train/val building overlap.")
    sa = set(train_df["scene_id"].astype(str))
    sb = set(val_df["scene_id"].astype(str))
    if sa & sb:
        raise RuntimeError("Train/val scene overlap.")
    return True
