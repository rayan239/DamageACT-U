"""
DamageACT-U crop utilities.

Design principles
-----------------
1) Define the spatial crop ONLY from PRE-disaster building geometry.
2) Apply the EXACT SAME pixel window to PRE and POST images.
3) Shift the square window inside the image instead of padding whenever possible.
4) Use a minimum raw crop side so extremely tiny buildings are not enlarged from
   only a handful of pixels to the full network input.
5) Never use the post-disaster damage label or post polygon to choose the crop.

This keeps the classifier conditional on a known PRE-event building footprint
without leaking post-event annotation geometry into model input construction.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
from PIL import Image, ImageDraw
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon


@dataclass(frozen=True)
class CropConfig:
    context_scale: float = 2.0
    min_side_px: int = 32
    output_size: int = 192
    interpolation: str = "bilinear"

    def validate(self) -> None:
        if self.context_scale <= 1.0:
            raise ValueError("context_scale should be > 1.0 so the crop includes context.")
        if self.min_side_px < 4:
            raise ValueError("min_side_px is unrealistically small.")
        if self.output_size < 32:
            raise ValueError("output_size is too small.")


def _get(row: Mapping[str, Any], key: str) -> Any:
    """Works with dict-like rows and pandas Series."""
    return row[key]


def _resample(name: str):
    name = name.lower()
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    if name == "nearest":
        return Image.Resampling.NEAREST
    raise ValueError(f"Unknown interpolation: {name}")


def resolve_path(dataset_root: str | Path, relative_path: str) -> Path:
    p = Path(dataset_root) / str(relative_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return p


def compute_pre_reference_crop_box(
    row: Mapping[str, Any],
    cfg: CropConfig,
) -> dict:
    """
    Compute one square crop box from PRE geometry only.

    Returns integer [x0, y0, x1, y1] with x1-x0 == y1-y0 == side.
    The box is shifted to remain inside the image. If the requested side exceeds
    the image dimensions, it is capped to min(image_width, image_height).
    """
    cfg.validate()

    width = int(_get(row, "pre_image_width"))
    height = int(_get(row, "pre_image_height"))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    # DamageACT-U v1 assumes the PRE and POST coordinate systems share dimensions.
    post_width = int(_get(row, "post_image_width"))
    post_height = int(_get(row, "post_image_height"))
    if (width, height) != (post_width, post_height):
        raise ValueError(
            "PRE/POST dimensions differ. Do not silently reuse the PRE crop box "
            f"across coordinate systems: PRE={width}x{height}, "
            f"POST={post_width}x{post_height}"
        )

    bbox_w = float(_get(row, "pre_bbox_width_px"))
    bbox_h = float(_get(row, "pre_bbox_height_px"))
    cx = float(_get(row, "pre_centroid_x"))
    cy = float(_get(row, "pre_centroid_y"))

    if not np.isfinite([bbox_w, bbox_h, cx, cy]).all():
        raise ValueError("Non-finite PRE geometry values.")
    if bbox_w <= 0 or bbox_h <= 0:
        raise ValueError(f"Invalid PRE bbox: width={bbox_w}, height={bbox_h}")

    max_dim = max(bbox_w, bbox_h)
    requested_side = max(float(cfg.min_side_px), cfg.context_scale * max_dim)

    max_legal_side = float(min(width, height))
    side_was_capped = requested_side > max_legal_side
    side = int(np.ceil(min(requested_side, max_legal_side)))
    side = max(1, min(side, width, height))

    # Desired centered crop.
    desired_x0 = cx - side / 2.0
    desired_y0 = cy - side / 2.0

    # Shift (rather than pad) to keep natural imagery inside the crop.
    max_x0 = width - side
    max_y0 = height - side
    x0 = int(round(np.clip(desired_x0, 0, max_x0)))
    y0 = int(round(np.clip(desired_y0, 0, max_y0)))
    x1 = x0 + side
    y1 = y0 + side

    # How far the crop center had to be moved because the building is near an edge.
    actual_cx = x0 + side / 2.0
    actual_cy = y0 + side / 2.0
    center_shift = float(np.hypot(actual_cx - cx, actual_cy - cy))

    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "side_px": side,
        "requested_side_px": float(requested_side),
        "side_was_capped": bool(side_was_capped),
        "used_min_side": bool(cfg.min_side_px > cfg.context_scale * max_dim),
        "crop_center_shift_px": center_shift,
        "building_max_dim_px": float(max_dim),
        "building_fraction_of_crop": float(max_dim / side),
        "image_width": width,
        "image_height": height,
    }


def crop_pair(
    row: Mapping[str, Any],
    dataset_root: str | Path,
    cfg: CropConfig,
) -> Tuple[Image.Image, Image.Image, dict]:
    """
    Load and crop PRE/POST using the same PRE-derived spatial box.

    The returned images are RGB PIL images resized to cfg.output_size.
    """
    box = compute_pre_reference_crop_box(row, cfg)

    pre_path = resolve_path(dataset_root, _get(row, "pre_image"))
    post_path = resolve_path(dataset_root, _get(row, "post_image"))

    with Image.open(pre_path) as im:
        pre = im.convert("RGB").crop((box["x0"], box["y0"], box["x1"], box["y1"]))
    with Image.open(post_path) as im:
        post = im.convert("RGB").crop((box["x0"], box["y0"], box["x1"], box["y1"]))

    expected = (box["side_px"], box["side_px"])
    if pre.size != expected or post.size != expected:
        raise RuntimeError(
            f"Unexpected crop size. expected={expected}, pre={pre.size}, post={post.size}"
        )

    target = (cfg.output_size, cfg.output_size)
    interp = _resample(cfg.interpolation)
    pre = pre.resize(target, interp)
    post = post.resize(target, interp)

    meta = dict(box)
    meta.update({
        "building_id": str(_get(row, "building_id")),
        "scene_id": str(_get(row, "scene_id")),
        "label_name": str(_get(row, "label_name")),
        "label_id": int(_get(row, "label_id")),
        "context_scale": float(cfg.context_scale),
        "min_side_px": int(cfg.min_side_px),
        "output_size": int(cfg.output_size),
    })
    return pre, post, meta


def _geometry_parts(geom):
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def draw_polygon_overlay(
    image: Image.Image,
    wkt_string: str,
    crop_meta: Mapping[str, Any],
    outline: tuple[int, int, int] = (255, 255, 0),
    width: int = 2,
) -> Image.Image:
    """
    Draw an annotation polygon on an already-resized crop for QA ONLY.

    This overlay must never be passed to the model.
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)

    try:
        geom = wkt.loads(str(wkt_string))
    except Exception:
        return out

    x0 = float(crop_meta["x0"])
    y0 = float(crop_meta["y0"])
    side = float(crop_meta["side_px"])
    output_size = int(crop_meta["output_size"])

    def map_xy(x, y):
        ox = (float(x) - x0) / side * output_size
        oy = (float(y) - y0) / side * output_size
        return (ox, oy)

    for poly in _geometry_parts(geom):
        pts = [map_xy(x, y) for x, y in poly.exterior.coords]
        if len(pts) >= 2:
            draw.line(pts, fill=outline, width=width, joint="curve")

    return out
