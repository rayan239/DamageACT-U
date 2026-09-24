from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TRAIN_EVENTS = ['lower-puna-volcano', 'palu-tsunami', 'mexico-earthquake', 'socal-fire', 'woolsey-fire', 'portugal-wildfire', 'pinery-bushfire', 'midwest-flooding', 'moore-tornado', 'joplin-tornado', 'hurricane-harvey', 'hurricane-michael', 'hurricane-florence']
TEST_EVENTS = ['nepal-flooding', 'guatemala-volcano', 'sunda-tsunami', 'santa-rosa-wildfire', 'hurricane-matthew', 'tuscaloosa-tornado']
EXPECTED = {"total":11034, "train":8202, "val":912, "test":1920}


def build_event_split(df: pd.DataFrame, event_col="event", seed=321, val_fraction=0.10):
    if event_col not in df.columns:
        raise ValueError("Missing event column.")
    x = df.copy().reset_index(drop=True)
    events = set(x[event_col].astype(str))
    unknown = events - set(TRAIN_EVENTS) - set(TEST_EVENTS)
    if unknown:
        raise ValueError("Unknown events: " + str(sorted(unknown)))


    trainval = x[x[event_col].isin(TRAIN_EVENTS)].copy().reset_index(drop=True)
    testdf = x[x[event_col].isin(TEST_EVENTS)].copy().reset_index(drop=True)


    idx = np.arange(len(trainval))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=trainval[event_col].to_numpy(),
    )
    role = np.full(len(trainval), "", dtype=object)
    role[tr_idx] = "train"
    role[va_idx] = "val"
    trainval["event_role"] = role
    testdf["event_role"] = "test"
    out = pd.concat([trainval, testdf], ignore_index=True)
    validate_event_split(out, event_col=event_col)
    return out


def validate_event_split(df, event_col="event"):
    dev = set(df.loc[df.event_role.isin(["train","val"]), event_col].astype(str))
    tst = set(df.loc[df.event_role.eq("test"), event_col].astype(str))
    if dev & tst:
        raise RuntimeError("Event leakage between development and held-out TEST.")
    if dev != set(TRAIN_EVENTS):
        raise RuntimeError("Development event set mismatch.")
    if tst != set(TEST_EVENTS):
        raise RuntimeError("TEST event set mismatch.")
    if "pair_key" in df.columns and df["pair_key"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate pair_key.")
    return True
