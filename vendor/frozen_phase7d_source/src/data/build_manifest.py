
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from shapely import wkt
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union
from tqdm import tqdm

try:
    from shapely.validation import make_valid  # Shapely >= 2
except ImportError:
    make_valid = None


LABEL_TO_ID = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

UNCLASSIFIED_LABELS = {
    "un-classified",
    "unclassified",
    "un_classified",
}

PRE_SUFFIX = "_pre_disaster"
POST_SUFFIX = "_post_disaster"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a research-auditable building-level manifest from the xBD Challenge training set."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/xbd/train"),
        help="xBD train directory containing images/, labels/, and targets/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests"),
        help="Directory in which manifest outputs will be written.",
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Optional number of scenes to process for a pilot/debug run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing manifest files to be replaced.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "manifest_builder.log"
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def scene_id_from_stem(stem: str, suffix: str) -> str | None:
    if not stem.endswith(suffix):
        return None
    return stem[: -len(suffix)]


def index_files(directory: Path, suffix: str, extension: str) -> dict[str, Path]:
    """Map scene_id -> path for files with a known phase suffix."""
    result: dict[str, Path] = {}
    pattern = f"*{suffix}{extension}"

    for path in directory.glob(pattern):
        scene_id = scene_id_from_stem(path.stem, suffix)
        if scene_id is None:
            continue
        if scene_id in result:
            raise RuntimeError(
                f"Duplicate file for scene '{scene_id}' in {directory}: "
                f"{result[scene_id]} and {path}"
            )
        result[scene_id] = path

    return result


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalized_label(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def get_building_features(
    data: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[str], int]:
    """
    Return:
      uid -> feature
      duplicated UIDs
      number of malformed building features without a UID
    """
    features = data.get("features", {}).get("xy", [])
    if not isinstance(features, list):
        return {}, set(), 0

    by_uid: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    missing_uid = 0

    for feature in features:
        props = feature.get("properties", {}) or {}
        feature_type = normalized_label(props.get("feature_type"))

        # xBD xy features are building polygons; keep explicit filtering defensive.
        if feature_type and feature_type != "building":
            continue

        uid = str(props.get("uid", "")).strip()
        if not uid:
            missing_uid += 1
            continue

        if uid in by_uid:
            duplicates.add(uid)
            continue

        by_uid[uid] = feature

    return by_uid, duplicates, missing_uid


def keep_polygonal_parts(geom):
    """Keep only polygonal content while preserving MultiPolygons."""
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom

    if isinstance(geom, GeometryCollection):
        polygonal = [
            g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty
        ]
        if not polygonal:
            return None
        return unary_union(polygonal)

    return None


def parse_polygon(feature: dict[str, Any]):
    """
    Parse xBD pixel-space WKT robustly.

    Returns:
        geom, repaired, error
    """
    raw_wkt = feature.get("wkt")
    if not raw_wkt:
        return None, False, "missing_wkt"

    try:
        geom = wkt.loads(raw_wkt)
    except Exception:
        return None, False, "wkt_parse_error"

    if geom is None or geom.is_empty:
        return None, False, "empty_geometry"

    repaired = False
    geom = keep_polygonal_parts(geom)
    if geom is None:
        return None, False, "non_polygon_geometry"

    if not geom.is_valid:
        repaired = True
        try:
            geom = make_valid(geom) if make_valid is not None else geom.buffer(0)
            geom = keep_polygonal_parts(geom)
        except Exception:
            geom = None

        if geom is None or geom.is_empty:
            return None, True, "geometry_repair_failed"

        # A second inexpensive fallback if make_valid still returns invalid polygonal data.
        if not geom.is_valid:
            try:
                geom = geom.buffer(0)
                geom = keep_polygonal_parts(geom)
            except Exception:
                geom = None

    if geom is None or geom.is_empty or not isinstance(geom, (Polygon, MultiPolygon)):
        return None, repaired, "invalid_polygon_after_repair"

    if not geom.is_valid:
        return None, repaired, "still_invalid_after_repair"

    if geom.area <= 0:
        return None, repaired, "zero_area_geometry"

    return geom, repaired, None


def image_size(path: Path) -> tuple[int, int]:
    """Pillow reads the image header without fully decoding the image."""
    with Image.open(path) as img:
        return img.size  # width, height


def visible_fraction(geom, width: int, height: int) -> float:
    canvas = box(0, 0, width, height)
    if geom.area <= 0:
        return 0.0
    intersection_area = geom.intersection(canvas).area
    return float(max(0.0, min(1.0, intersection_area / geom.area)))


def geometry_fields(prefix: str, geom, repaired: bool, width: int, height: int) -> dict[str, Any]:
    minx, miny, maxx, maxy = geom.bounds
    centroid = geom.centroid
    vis = visible_fraction(geom, width, height)

    return {
        f"{prefix}_wkt": geom.wkt,
        f"{prefix}_geom_type": geom.geom_type,
        f"{prefix}_geometry_repaired": repaired,
        f"{prefix}_area_px2": float(geom.area),
        f"{prefix}_minx": float(minx),
        f"{prefix}_miny": float(miny),
        f"{prefix}_maxx": float(maxx),
        f"{prefix}_maxy": float(maxy),
        f"{prefix}_bbox_width_px": float(maxx - minx),
        f"{prefix}_bbox_height_px": float(maxy - miny),
        f"{prefix}_centroid_x": float(centroid.x),
        f"{prefix}_centroid_y": float(centroid.y),
        f"{prefix}_visible_fraction": vis,
        f"{prefix}_touches_or_crosses_border": bool(vis < 0.999999),
    }


def metadata_fields(pre_meta: dict[str, Any], post_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "disaster": post_meta.get("disaster", pre_meta.get("disaster")),
        "disaster_type": post_meta.get("disaster_type", pre_meta.get("disaster_type")),
        "pre_capture_date": pre_meta.get("capture_date"),
        "post_capture_date": post_meta.get("capture_date"),
        "pre_catalog_id": pre_meta.get("catalog_id"),
        "post_catalog_id": post_meta.get("catalog_id"),
        "pre_sensor": pre_meta.get("sensor"),
        "post_sensor": post_meta.get("sensor"),
        "pre_gsd": pre_meta.get("gsd"),
        "post_gsd": post_meta.get("gsd"),
        "pre_off_nadir_angle": pre_meta.get("off_nadir_angle"),
        "post_off_nadir_angle": post_meta.get("off_nadir_angle"),
    }


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def excluded_record(
    scene_id: str,
    uid: str | None,
    reason: str,
    raw_label: str | None = None,
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "uid": uid,
        "reason": reason,
        "raw_label": raw_label,
    }


def save_dataframe(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Re-run with --overwrite if replacement is intentional."
        )

    if path.suffix == ".gz":
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    setup_logging(output)

    images_dir = root / "images"
    labels_dir = root / "labels"
    targets_dir = root / "targets"

    for required in [images_dir, labels_dir]:
        if not required.exists():
            raise FileNotFoundError(f"Required directory not found: {required}")

    if not targets_dir.exists():
        logging.warning("targets/ not found. This is okay for the manifest builder.")

    logging.info("Dataset root: %s", root)
    logging.info("Output directory: %s", output)

    pre_images = index_files(images_dir, PRE_SUFFIX, ".png")
    post_images = index_files(images_dir, POST_SUFFIX, ".png")
    pre_jsons = index_files(labels_dir, PRE_SUFFIX, ".json")
    post_jsons = index_files(labels_dir, POST_SUFFIX, ".json")

    all_scene_ids = sorted(
        set(pre_images)
        | set(post_images)
        | set(pre_jsons)
        | set(post_jsons)
    )

    if args.limit_scenes is not None:
        all_scene_ids = all_scene_ids[: args.limit_scenes]

    logging.info("Scenes discovered: %d", len(all_scene_ids))

    manifest_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for scene_id in tqdm(all_scene_ids, desc="Building manifest"):
        scene_excluded_before = len(excluded_rows)
        required_paths = {
            "pre_image": pre_images.get(scene_id),
            "post_image": post_images.get(scene_id),
            "pre_json": pre_jsons.get(scene_id),
            "post_json": post_jsons.get(scene_id),
        }

        missing = [name for name, path in required_paths.items() if path is None]

        scene_summary: dict[str, Any] = {
            "scene_id": scene_id,
            "status": "ok",
            "missing_resources": ";".join(missing),
            "n_pre_buildings": 0,
            "n_post_buildings": 0,
            "n_matched_uids": 0,
            "n_usable": 0,
            "n_excluded": 0,
            "n_no_damage": 0,
            "n_minor_damage": 0,
            "n_major_damage": 0,
            "n_destroyed": 0,
        }

        if missing:
            scene_summary["status"] = "missing_resources"
            reason_counts["scene_missing_resources"] += 1
            scene_rows.append(scene_summary)
            continue

        pre_img_path = required_paths["pre_image"]
        post_img_path = required_paths["post_image"]
        pre_json_path = required_paths["pre_json"]
        post_json_path = required_paths["post_json"]

        assert pre_img_path is not None
        assert post_img_path is not None
        assert pre_json_path is not None
        assert post_json_path is not None

        try:
            pre_data = read_json(pre_json_path)
            post_data = read_json(post_json_path)
        except Exception as exc:
            scene_summary["status"] = "json_read_error"
            scene_summary["error"] = repr(exc)
            reason_counts["scene_json_read_error"] += 1
            scene_rows.append(scene_summary)
            continue

        try:
            pre_width, pre_height = image_size(pre_img_path)
            post_width, post_height = image_size(post_img_path)
        except Exception as exc:
            scene_summary["status"] = "image_header_error"
            scene_summary["error"] = repr(exc)
            reason_counts["scene_image_header_error"] += 1
            scene_rows.append(scene_summary)
            continue

        pre_meta = pre_data.get("metadata", {}) or {}
        post_meta = post_data.get("metadata", {}) or {}

        pre_by_uid, pre_duplicates, pre_missing_uid = get_building_features(pre_data)
        post_by_uid, post_duplicates, post_missing_uid = get_building_features(post_data)

        scene_summary["n_pre_buildings"] = len(pre_by_uid)
        scene_summary["n_post_buildings"] = len(post_by_uid)
        scene_summary["n_pre_duplicate_uids"] = len(pre_duplicates)
        scene_summary["n_post_duplicate_uids"] = len(post_duplicates)
        scene_summary["n_pre_features_missing_uid"] = pre_missing_uid
        scene_summary["n_post_features_missing_uid"] = post_missing_uid
        scene_summary["pre_width"] = pre_width
        scene_summary["pre_height"] = pre_height
        scene_summary["post_width"] = post_width
        scene_summary["post_height"] = post_height

        # Keep scene metadata in the scene-level audit table too.
        scene_summary.update(metadata_fields(pre_meta, post_meta))

        pre_uids = set(pre_by_uid)
        post_uids = set(post_by_uid)
        matched_uids = pre_uids & post_uids
        scene_summary["n_matched_uids"] = len(matched_uids)
        scene_summary["n_pre_only_uids"] = len(pre_uids - post_uids)
        scene_summary["n_post_only_uids"] = len(post_uids - pre_uids)

        # Record unmatched buildings explicitly.
        for uid in sorted(pre_uids - post_uids):
            excluded_rows.append(excluded_record(scene_id, uid, "uid_missing_in_post"))
            reason_counts["uid_missing_in_post"] += 1

        for uid in sorted(post_uids - pre_uids):
            props = post_by_uid[uid].get("properties", {}) or {}
            raw_label = normalized_label(props.get("subtype"))
            excluded_rows.append(
                excluded_record(scene_id, uid, "uid_missing_in_pre", raw_label)
            )
            reason_counts["uid_missing_in_pre"] += 1

        duplicate_uids = pre_duplicates | post_duplicates

        class_counter: Counter[str] = Counter()

        for uid in sorted(matched_uids):
            if uid in duplicate_uids:
                excluded_rows.append(excluded_record(scene_id, uid, "duplicate_uid"))
                reason_counts["duplicate_uid"] += 1
                continue

            pre_feature = pre_by_uid[uid]
            post_feature = post_by_uid[uid]
            post_props = post_feature.get("properties", {}) or {}
            raw_label = normalized_label(post_props.get("subtype"))

            if raw_label in UNCLASSIFIED_LABELS:
                excluded_rows.append(
                    excluded_record(scene_id, uid, "unclassified", raw_label)
                )
                reason_counts["unclassified"] += 1
                continue

            if raw_label not in LABEL_TO_ID:
                excluded_rows.append(
                    excluded_record(scene_id, uid, "unknown_or_missing_label", raw_label)
                )
                reason_counts["unknown_or_missing_label"] += 1
                continue

            pre_geom, pre_repaired, pre_error = parse_polygon(pre_feature)
            if pre_error is not None:
                excluded_rows.append(
                    excluded_record(scene_id, uid, f"pre_{pre_error}", raw_label)
                )
                reason_counts[f"pre_{pre_error}"] += 1
                continue

            post_geom, post_repaired, post_error = parse_polygon(post_feature)
            if post_error is not None:
                excluded_rows.append(
                    excluded_record(scene_id, uid, f"post_{post_error}", raw_label)
                )
                reason_counts[f"post_{post_error}"] += 1
                continue

            assert pre_geom is not None
            assert post_geom is not None

            pre_vis = visible_fraction(pre_geom, pre_width, pre_height)
            post_vis = visible_fraction(post_geom, post_width, post_height)

            if pre_vis <= 0:
                excluded_rows.append(
                    excluded_record(scene_id, uid, "pre_geometry_outside_image", raw_label)
                )
                reason_counts["pre_geometry_outside_image"] += 1
                continue

            if post_vis <= 0:
                excluded_rows.append(
                    excluded_record(scene_id, uid, "post_geometry_outside_image", raw_label)
                )
                reason_counts["post_geometry_outside_image"] += 1
                continue

            pre_centroid = pre_geom.centroid
            post_centroid = post_geom.centroid
            centroid_shift = math.hypot(
                post_centroid.x - pre_centroid.x,
                post_centroid.y - pre_centroid.y,
            )

            row: dict[str, Any] = {
                "building_id": f"{scene_id}__{uid}",
                "scene_id": scene_id,
                "uid": uid,
                "source_split": "challenge_train",
                "label_name": raw_label,
                "label_id": LABEL_TO_ID[raw_label],
                "pre_image": relative_posix(pre_img_path, root),
                "post_image": relative_posix(post_img_path, root),
                "pre_json": relative_posix(pre_json_path, root),
                "post_json": relative_posix(post_json_path, root),
                "pre_image_width": pre_width,
                "pre_image_height": pre_height,
                "post_image_width": post_width,
                "post_image_height": post_height,
                "same_image_dimensions": (
                    pre_width == post_width and pre_height == post_height
                ),
                "centroid_shift_px": float(centroid_shift),
                "post_to_pre_area_ratio": (
                    float(post_geom.area / pre_geom.area)
                    if pre_geom.area > 0
                    else None
                ),
            }

            row.update(metadata_fields(pre_meta, post_meta))
            row.update(
                geometry_fields(
                    "pre", pre_geom, pre_repaired, pre_width, pre_height
                )
            )
            row.update(
                geometry_fields(
                    "post", post_geom, post_repaired, post_width, post_height
                )
            )

            manifest_rows.append(row)
            class_counter[raw_label] += 1

        scene_summary["n_usable"] = sum(class_counter.values())
        scene_summary["n_no_damage"] = class_counter["no-damage"]
        scene_summary["n_minor_damage"] = class_counter["minor-damage"]
        scene_summary["n_major_damage"] = class_counter["major-damage"]
        scene_summary["n_destroyed"] = class_counter["destroyed"]

        scene_summary["n_excluded"] = len(excluded_rows) - scene_excluded_before

        scene_rows.append(scene_summary)

    manifest_df = pd.DataFrame(manifest_rows)
    excluded_df = pd.DataFrame(excluded_rows)
    scenes_df = pd.DataFrame(scene_rows)

    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(
            ["scene_id", "uid"], kind="stable"
        ).reset_index(drop=True)

        if manifest_df["building_id"].duplicated().any():
            duplicated = manifest_df.loc[
                manifest_df["building_id"].duplicated(keep=False), "building_id"
            ].tolist()
            raise RuntimeError(
                "Duplicate building_id values found in final manifest. "
                f"Examples: {duplicated[:10]}"
            )

    output.mkdir(parents=True, exist_ok=True)

    manifest_path = output / "all_buildings.csv.gz"
    preview_path = output / "all_buildings_preview.csv"
    scene_path = output / "scene_summary.csv"
    excluded_path = output / "excluded_buildings.csv.gz"
    report_path = output / "manifest_report.json"

    save_dataframe(manifest_df, manifest_path, args.overwrite)
    save_dataframe(manifest_df.head(200), preview_path, args.overwrite)
    save_dataframe(scenes_df, scene_path, args.overwrite)
    save_dataframe(excluded_df, excluded_path, args.overwrite)

    label_counts = (
        manifest_df["label_name"].value_counts().to_dict()
        if not manifest_df.empty
        else {}
    )
    disaster_counts = (
        manifest_df["disaster"].fillna("UNKNOWN").value_counts().to_dict()
        if not manifest_df.empty and "disaster" in manifest_df.columns
        else {}
    )

    report = {
        "dataset_root": str(root),
        "scenes_processed": len(all_scene_ids),
        "usable_buildings": int(len(manifest_df)),
        "excluded_records": int(len(excluded_df)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "disaster_counts": {str(k): int(v) for k, v in disaster_counts.items()},
        "exclusion_reason_counts": {
            str(k): int(v) for k, v in reason_counts.most_common()
        },
        "scene_status_counts": {
            str(k): int(v)
            for k, v in scenes_df["status"].value_counts().to_dict().items()
        }
        if not scenes_df.empty
        else {},
        "label_mapping": LABEL_TO_ID,
        "notes": {
            "canonical_manifest": "all_buildings.csv.gz",
            "paths_are_relative_to_dataset_root": True,
            "unclassified_buildings_are_excluded": True,
            "split_has_not_been_created_yet": True,
        },
    }

    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{report_path} already exists. Use --overwrite if intentional."
        )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info("Done.")
    logging.info("Usable buildings: %d", len(manifest_df))
    logging.info("Excluded records: %d", len(excluded_df))
    logging.info("Class counts: %s", label_counts)
    logging.info("Manifest: %s", manifest_path)
    logging.info("Scene summary: %s", scene_path)
    logging.info("Audit report: %s", report_path)


if __name__ == "__main__":
    main()
