from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
from torch.utils.data import DataLoader, Subset


EXPECTED = {
    "audit/g3_clean_parity_panel_23.csv":
        "2924bc88beeda3507c8cf2178db53c2d17eca7483c483a61ca59087a87636c43",
    "audit/g3_clean_parity_selector_96.csv":
        "131ede6d2c4bf3efbd0e62a219994bc8228457aeaeb6022e5068d1586ee97f03",
    "audit/g3_parity_selection_lock.json":
        "cf463ca5d680ba7ede33d498350692e2782f5d716bc527629dd07a77e210883c",
    "audit/g3_parity_execution_protocol.json":
        "112ade4ac863dd7bc7df03a6c2544be6b3757ca0034f864556738851b2d2409c",
    "manifests/buildings/event_test_buildings_rebuilt.csv.gz":
        "1dda199c0faa444e955ecf7b131872dc5c5a713482914e821cd916aea4b37a2a",
    "predictions/heldout_events/event_test_clean_expert_predictions.csv.gz":
        "f0657fc72629acb8730464897c44040cde65942966a81f47c956f3c5b1bb09f2",
    "checkpoints/experts/post_seed42_best_state_dict.pt":
        "df6b607973ea74e1a6aa7eb640459b65c7a4e1bdaabc7fce956d2c757374967c",
    "checkpoints/experts/siamese_seed42_best_state_dict.pt":
        "3d6466108cf44e994c0c7b7bb1745e549807d89601e45898911de34a455d73a5",
    "vendor/frozen_phase7d_source/src/data/dataset.py":
        "165a71c4ebf6adea14f20f735bc487e883c5448e5ccfaf4fe8b74e49a3a77044",
    "vendor/frozen_phase7d_source/src/data/crop_utils.py":
        "8c34139a0f96eaf8246dd228cf27c83386d6c1ca049426c25d28666576589593",
    "vendor/frozen_phase7d_source/src/models/damage_models.py":
        "9f9ee3382902f339309ddf03a203b882d42e8674de7bf8ee3d18b14863478f2d",
}

ATOL = 1e-5
RTOL = 1e-5
BATCH_SIZE = 96
NUM_WORKERS = 2


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def subset_dir(root, subset):
    aliases = {
        "train": ["train", "tier1"],
        "tier1": ["tier1", "train"],
        "tier3": ["tier3"],
        "test": ["test"],
        "hold": ["hold", "holdout"],
        "holdout": ["holdout", "hold"],
    }.get(str(subset), [str(subset)])

    good = []
    for a in aliases:
        q = root / a
        if (q / "images").is_dir() and (q / "labels").is_dir():
            good.append(q)

    if not good:
        raise RuntimeError(
            f"No physical xBD subset for manifest subset={subset}"
        )

    return good[0]


def compare(new, frozen):
    new = np.asarray(new, dtype=np.float64)
    frozen = np.asarray(frozen, dtype=np.float64)

    diff = np.abs(new - frozen)

    return {
        "max_absolute_difference": float(diff.max()),
        "mean_absolute_difference": float(diff.mean()),
        "median_absolute_difference": float(np.median(diff)),
        "allclose": bool(
            np.allclose(
                new,
                frozen,
                atol=ATOL,
                rtol=RTOL,
            )
        ),
    }


def four_columns(df, pattern):
    return df[
        [pattern.format(k=k) for k in range(4)]
    ].to_numpy(dtype=np.float64)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--bundle-root",
    required=True,
)

parser.add_argument(
    "--xbd-root",
    required=True,
)

parser.add_argument(
    "--output-root",
    default="/kaggle/working/damageactu_g3_checkpoint_parity",
)

args = parser.parse_args()

BUNDLE = Path(args.bundle_root).resolve()
XBD_ROOT = Path(args.xbd_root).resolve()
OUT = Path(args.output_root).resolve()
STAGE = OUT.parent / "damageactu_g3_parity_stage"


# ============================================================
# 1. Frozen bundle integrity
# ============================================================

for rel, expected in EXPECTED.items():
    p = BUNDLE / rel

    if not p.is_file():
        raise FileNotFoundError(p)

    got = sha256_file(p)

    if got != expected:
        raise RuntimeError(
            f"Hash mismatch: {rel}\n"
            f"got={got}\n"
            f"expected={expected}"
        )

print("PASS: all 11 frozen input hashes")


sums_path = BUNDLE / "SHA256SUMS.json"

if not sums_path.is_file():
    raise FileNotFoundError(sums_path)

sums = json.loads(
    sums_path.read_text(encoding="utf-8")
)

if len(sums) != 11:
    raise RuntimeError(
        f"Expected 11 SHA256SUMS entries; found {len(sums)}"
    )

for rel, expected in sums.items():
    p = BUNDLE / rel

    if sha256_file(p) != expected:
        raise RuntimeError(
            f"Internal SHA256SUMS failure: {rel}"
        )

print("PASS: internal SHA256SUMS")


# ============================================================
# 2. Environment gate
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable. Authoritative G3 run requires Tesla T4."
    )

environment = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "pillow": PIL.__version__,
}

print(json.dumps(environment, indent=2))


errors = []

if sys.version_info[:2] != (3, 12):
    errors.append(
        f"Python={sys.version_info[:3]} expected 3.12.x"
    )

if torch.__version__ != "2.10.0+cu128":
    errors.append(
        f"torch={torch.__version__} expected 2.10.0+cu128"
    )

if torchvision.__version__ != "0.25.0+cu128":
    errors.append(
        f"torchvision={torchvision.__version__} "
        "expected 0.25.0+cu128"
    )

if str(torch.version.cuda) != "12.8":
    errors.append(
        f"CUDA={torch.version.cuda} expected 12.8"
    )

if "T4" not in torch.cuda.get_device_name(0):
    errors.append(
        f"GPU={torch.cuda.get_device_name(0)} expected Tesla T4"
    )

if errors:
    raise RuntimeError(
        "AUTHORITATIVE ENVIRONMENT GATE FAILED:\n"
        + "\n".join(errors)
    )

print("PASS: authoritative T4 environment gate")
print(
    "NOTE: historical Pillow version was not recorded; "
    "runtime Pillow is reported, not claimed canonical."
)


# ============================================================
# 3. Frozen execution rule
# ============================================================

protocol = json.loads(
    (
        BUNDLE /
        "audit/g3_parity_execution_protocol.json"
    ).read_text(encoding="utf-8")
)

rule = protocol[
    "predeclared_parity_rules"
]["numerical"]

if float(rule["atol"]) != ATOL:
    raise RuntimeError("ATOL differs from frozen protocol")

if float(rule["rtol"]) != RTOL:
    raise RuntimeError("RTOL differs from frozen protocol")

print("PASS: predeclared numerical rule")


# ============================================================
# 4. Canonical manifest + selector
# ============================================================

manifest_path = (
    BUNDLE /
    "manifests/buildings/event_test_buildings_rebuilt.csv.gz"
)

selector_path = (
    BUNDLE /
    "audit/g3_clean_parity_selector_96.csv"
)

anchors_path = (
    BUNDLE /
    "audit/g3_clean_parity_panel_23.csv"
)

frozen_predictions_path = (
    BUNDLE /
    "predictions/heldout_events/"
    "event_test_clean_expert_predictions.csv.gz"
)

manifest = pd.read_csv(
    manifest_path,
    low_memory=False,
)

selector = pd.read_csv(
    selector_path
)

anchors = pd.read_csv(
    anchors_path
)

if len(manifest) != 115349:
    raise RuntimeError(
        f"Unexpected canonical manifest rows: {len(manifest)}"
    )

if len(selector) != 96:
    raise RuntimeError(
        f"Unexpected selector rows: {len(selector)}"
    )

if selector.building_id.astype(str).nunique() != 96:
    raise RuntimeError(
        "Selector building IDs not unique"
    )

if int(selector.is_anchor.sum()) != 23:
    raise RuntimeError(
        "Selector does not contain exactly 23 anchors"
    )

if not selector.manifest_row.astype(int).is_monotonic_increasing:
    raise RuntimeError(
        "Selector no longer follows canonical manifest order"
    )


indices = selector.manifest_row.astype(int).tolist()

selected = (
    manifest
    .iloc[indices]
    .reset_index(drop=True)
)

if (
    selected.building_id.astype(str).tolist()
    != selector.building_id.astype(str).tolist()
):
    raise RuntimeError(
        "Selector -> canonical manifest identity failure"
    )


anchor_ids = set(
    anchors.building_id.astype(str)
)

if not anchor_ids.issubset(
    set(selector.building_id.astype(str))
):
    raise RuntimeError(
        "23-anchor panel is not contained in selector"
    )

print("PASS: canonical manifest + selector identity")


# ============================================================
# 5. Stage exactly the 77 selected scenes
# ============================================================

scene_check = (
    selected
    .groupby("scene_id")
    .agg(
        n_subset=("source_subset", "nunique"),
        n_pre=("pre_image", "nunique"),
        n_post=("post_image", "nunique"),
    )
)

if not (
    (scene_check.n_subset == 1)
    & (scene_check.n_pre == 1)
    & (scene_check.n_post == 1)
).all():
    raise RuntimeError(
        "Inconsistent selected scene metadata"
    )


scenes = (
    selected[
        [
            "scene_id",
            "source_subset",
            "pre_image",
            "post_image",
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

if len(scenes) != 77:
    raise RuntimeError(
        f"Expected 77 scenes; found {len(scenes)}"
    )


if STAGE.exists():
    shutil.rmtree(STAGE)

(STAGE / "images").mkdir(
    parents=True
)

(STAGE / "labels").mkdir(
    parents=True
)


created = set()

for r in scenes.itertuples(index=False):

    src = subset_dir(
        XBD_ROOT,
        r.source_subset,
    )

    stem = str(r.scene_id)

    required = [
        (
            src / str(r.pre_image),
            STAGE / str(r.pre_image),
        ),
        (
            src / str(r.post_image),
            STAGE / str(r.post_image),
        ),
        (
            src / "labels" /
            f"{stem}_pre_disaster.json",
            STAGE / "labels" /
            f"{stem}_pre_disaster.json",
        ),
        (
            src / "labels" /
            f"{stem}_post_disaster.json",
            STAGE / "labels" /
            f"{stem}_post_disaster.json",
        ),
    ]

    for source, destination in required:

        if not source.is_file():
            raise FileNotFoundError(source)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination not in created:
            os.symlink(
                source,
                destination,
            )
            created.add(destination)


if len(created) != 77 * 4:
    raise RuntimeError(
        f"Unexpected staged artifact count: {len(created)}"
    )

print("PASS: 77 selected scenes staged")


# ============================================================
# 6. Frozen source imports
# ============================================================

FROZEN = (
    BUNDLE /
    "vendor/frozen_phase7d_source"
)

sys.path.insert(
    0,
    str(FROZEN),
)

sys.path.insert(
    0,
    str(FROZEN / "src/data"),
)

from src.data.crop_utils import CropConfig
from src.data.dataset import PairedXBDDataset
from src.models.damage_models import DamageACTClassifier


CROP = CropConfig(
    context_scale=2.0,
    min_side_px=32,
    output_size=192,
)


# ============================================================
# 7. Canonical dataset + 96-row Subset
# ============================================================

full_ds = PairedXBDDataset(
    manifest_path=manifest_path,
    dataset_root=STAGE,
    crop_config=CROP,
    training=False,
    augment_geometry=False,
    return_metadata=True,
)

parity_ds = Subset(
    full_ds,
    indices,
)

loader = DataLoader(
    parity_ds,
    batch_size=96,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
    persistent_workers=False,
)

if len(loader) != 1:
    raise RuntimeError(
        f"Expected one 96-building batch; got {len(loader)}"
    )


# ============================================================
# 8. Frozen expert checkpoints
# ============================================================

DEVICE = torch.device(
    "cuda:0"
)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

try:
    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )
except Exception:
    pass


post_model = DamageACTClassifier(
    "post_only",
    pretrained=False,
).to(DEVICE)

pair_model = DamageACTClassifier(
    "siamese",
    pretrained=False,
).to(DEVICE)


post_model.load_state_dict(
    torch.load(
        BUNDLE /
        "checkpoints/experts/"
        "post_seed42_best_state_dict.pt",
        map_location="cpu",
        weights_only=True,
    ),
    strict=True,
)

pair_model.load_state_dict(
    torch.load(
        BUNDLE /
        "checkpoints/experts/"
        "siamese_seed42_best_state_dict.pt",
        map_location="cpu",
        weights_only=True,
    ),
    strict=True,
)

post_model.eval()
pair_model.eval()

for model in [
    post_model,
    pair_model,
]:
    for p in model.parameters():
        p.requires_grad_(False)

print("PASS: frozen checkpoints loaded strict=True")


# ============================================================
# 9. One canonical-size FP32 inference batch
# ============================================================

rows = []

with torch.inference_mode():

    for batch in loader:

        pre = batch["pre"].to(
            DEVICE,
            non_blocking=True,
        )

        post = batch["post"].to(
            DEVICE,
            non_blocking=True,
        )

        # IMPORTANT:
        # no autocast; canonical final TEST used direct FP32 forwards.
        lp = post_model(
            pre,
            post,
        )

        lt = pair_model(
            pre,
            post,
        )

        pp = torch.softmax(
            lp,
            dim=1,
        ).cpu().numpy()

        pt = torch.softmax(
            lt,
            dim=1,
        ).cpu().numpy()

        lpn = lp.cpu().numpy()
        ltn = lt.cpu().numpy()

        for i, bid in enumerate(
            batch["building_id"]
        ):

            rec = {
                "building_id": str(bid),
                "scene_id":
                    str(batch["scene_id"][i]),
                "disaster":
                    str(batch["disaster"][i]),
                "true_label":
                    int(batch["label"][i]),
                "new_pred_post":
                    int(pp[i].argmax()),
                "new_pred_pair":
                    int(pt[i].argmax()),
            }

            for k in range(4):

                rec[f"new_p{k}_post"] = (
                    float(pp[i, k])
                )

                rec[f"new_p{k}_pair"] = (
                    float(pt[i, k])
                )

                rec[f"new_logit{k}_post"] = (
                    float(lpn[i, k])
                )

                rec[f"new_logit{k}_pair"] = (
                    float(ltn[i, k])
                )

            rows.append(rec)


new = pd.DataFrame(
    rows
)

if len(new) != 96:
    raise RuntimeError(
        f"Expected 96 outputs; got {len(new)}"
    )

if not new.building_id.is_unique:
    raise RuntimeError(
        "Regenerated building IDs not unique"
    )

if (
    new.building_id.astype(str).tolist()
    != selector.building_id.astype(str).tolist()
):
    raise RuntimeError(
        "Regenerated output ordering changed"
    )

print("PASS: regenerated 96 clean expert outputs")


# ============================================================
# 10. Align sealed Phase7D-D outputs
# ============================================================

frozen = pd.read_csv(
    frozen_predictions_path
)

if frozen.building_id.astype(str).duplicated().any():
    raise RuntimeError(
        "Frozen prediction IDs not unique"
    )

frozen["_id"] = (
    frozen.building_id.astype(str)
)

frozen = (
    frozen
    .set_index("_id")
    .loc[
        new.building_id.astype(str).tolist()
    ]
    .reset_index(drop=True)
)

if (
    new.scene_id.astype(str).tolist()
    != frozen.scene_id.astype(str).tolist()
):
    raise RuntimeError(
        "Scene identity mismatch"
    )

if (
    new.true_label.astype(int).tolist()
    != frozen.true_label.astype(int).tolist()
):
    raise RuntimeError(
        "Label identity mismatch"
    )


combined = new.copy()

for c in frozen.columns:

    if c not in {
        "building_id",
        "scene_id",
        "disaster",
        "true_label",
        "_id",
    }:
        combined[
            f"frozen_{c}"
        ] = frozen[c].to_numpy()


# ============================================================
# 11. Predeclared parity tests
# ============================================================

post_new_logits = four_columns(
    combined,
    "new_logit{k}_post",
)

post_old_logits = four_columns(
    combined,
    "frozen_logit{k}_post",
)

pair_new_logits = four_columns(
    combined,
    "new_logit{k}_pair",
)

pair_old_logits = four_columns(
    combined,
    "frozen_logit{k}_pair",
)

post_new_prob = four_columns(
    combined,
    "new_p{k}_post",
)

post_old_prob = four_columns(
    combined,
    "frozen_p{k}_post",
)

pair_new_prob = four_columns(
    combined,
    "new_p{k}_pair",
)

pair_old_prob = four_columns(
    combined,
    "frozen_p{k}_pair",
)


post_match = (
    combined.new_pred_post.to_numpy(int)
    ==
    combined.frozen_pred_post.to_numpy(int)
)

pair_match = (
    combined.new_pred_pair.to_numpy(int)
    ==
    combined.frozen_pred_pair.to_numpy(int)
)


anchor_mask = (
    combined
    .building_id
    .astype(str)
    .isin(anchor_ids)
    .to_numpy()
)

if int(anchor_mask.sum()) != 23:
    raise RuntimeError(
        "Anchor identity failure"
    )


numerical = {
    "post_logits":
        compare(
            post_new_logits,
            post_old_logits,
        ),

    "post_probabilities":
        compare(
            post_new_prob,
            post_old_prob,
        ),

    "siamese_logits":
        compare(
            pair_new_logits,
            pair_old_logits,
        ),

    "siamese_probabilities":
        compare(
            pair_new_prob,
            pair_old_prob,
        ),
}


anchor_numerical = {
    "post_logits":
        compare(
            post_new_logits[anchor_mask],
            post_old_logits[anchor_mask],
        ),

    "post_probabilities":
        compare(
            post_new_prob[anchor_mask],
            post_old_prob[anchor_mask],
        ),

    "siamese_logits":
        compare(
            pair_new_logits[anchor_mask],
            pair_old_logits[anchor_mask],
        ),

    "siamese_probabilities":
        compare(
            pair_new_prob[anchor_mask],
            pair_old_prob[anchor_mask],
        ),
}


categorical_pass = bool(
    post_match.all()
    and pair_match.all()
)

numerical_pass = bool(
    all(
        x["allclose"]
        for x in numerical.values()
    )
)

strict_pass = bool(
    categorical_pass
    and numerical_pass
)


# ============================================================
# 12. Save audit results
# ============================================================

if OUT.exists():
    shutil.rmtree(
        OUT
    )

OUT.mkdir(
    parents=True
)


report = {
    "stage":
        "G3_checkpoint_inference_parity",

    "status":
        (
            "PASS"
            if strict_pass
            else "MISMATCH_REQUIRES_INVESTIGATION"
        ),

    "frozen_commits": {
        "selection":
            "1ae5e4f",

        "execution_protocol":
            "398d913",

        "input_bundle":
            "99ededc",
    },

    "environment":
        environment,

    "counts": {
        "manifest_rows":
            len(manifest),

        "selector_rows":
            len(selector),

        "anchors":
            int(anchor_mask.sum()),

        "selected_scenes":
            len(scenes),
    },

    "predeclared_parity_rule": {
        "atol":
            ATOL,

        "rtol":
            RTOL,

        "exact_argmax_required":
            True,
    },

    "categorical": {
        "post_matches":
            int(post_match.sum()),

        "post_total":
            96,

        "siamese_matches":
            int(pair_match.sum()),

        "siamese_total":
            96,

        "anchor_post_matches":
            int(
                post_match[
                    anchor_mask
                ].sum()
            ),

        "anchor_siamese_matches":
            int(
                pair_match[
                    anchor_mask
                ].sum()
            ),

        "all_predictions_match":
            categorical_pass,
    },

    "numerical":
        numerical,

    "anchor_numerical":
        anchor_numerical,

    "strict_parity_pass":
        strict_pass,

    "runner_sha256":
        sha256_file(
            Path(__file__).resolve()
        ),

    "limitations": [
        (
            "Checkpoint inference parity only; "
            "not training-from-scratch reproduction."
        ),
        (
            "Historical Pillow version was not recorded."
        ),
        (
            "Event-track expert evidence remains "
            "single-initialization Seed 42."
        ),
    ],
}


combined.to_csv(
    OUT /
    "g3_regenerated_vs_frozen_96.csv",
    index=False,
)

(
    OUT /
    "g3_checkpoint_parity_report.json"
).write_text(
    json.dumps(
        report,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

(
    OUT /
    "g3_environment.json"
).write_text(
    json.dumps(
        environment,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print()
print(
    "=== G3 CHECKPOINT PARITY SUMMARY ==="
)

print(
    "POST prediction matches    :",
    f"{post_match.sum()}/96",
)

print(
    "Siamese prediction matches :",
    f"{pair_match.sum()}/96",
)

for name, x in numerical.items():

    print(
        name,
        "max_abs=",
        x["max_absolute_difference"],
        "mean_abs=",
        x["mean_absolute_difference"],
        "median_abs=",
        x["median_absolute_difference"],
        "allclose=",
        x["allclose"],
    )

print(
    "STRICT_PARITY_PASS:",
    strict_pass,
)


# Remove staged raw-data links before packaging results.
if STAGE.exists():
    shutil.rmtree(
        STAGE
    )


zip_base = (
    OUT.parent /
    "DamageACTU_G3_Parity_Results"
)

shutil.make_archive(
    str(zip_base),
    "zip",
    root_dir=OUT,
)

print(
    "RESULT_ZIP:",
    str(zip_base) + ".zip",
)