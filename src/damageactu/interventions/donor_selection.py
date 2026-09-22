from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DonorCandidate:
    building_id: str
    scene_id: str
    area: float
    crop_box: tuple[float,float,float,float]


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = map(float, a)
    bx0, by0, bx1, by1 = map(float, b)
    ix0, iy0 = max(ax0,bx0), max(ay0,by0)
    ix1, iy1 = min(ax1,bx1), min(ay1,by1)
    iw, ih = max(0.0, ix1-ix0), max(0.0, iy1-iy0)
    inter = iw*ih
    aa = max(0.0, ax1-ax0)*max(0.0, ay1-ay0)
    bb = max(0.0, bx1-bx0)*max(0.0, by1-by0)
    den = aa+bb-inter
    return 0.0 if den <= 0 else inter/den


def choose_wrongpre_donor(target: DonorCandidate, candidates, preferred_iou=0.01):
    allowed = [
        c for c in candidates
        if c.scene_id == target.scene_id and c.building_id != target.building_id
    ]
    if not allowed:
        return None
    def area_gap(c):
        return abs(math.log(max(target.area,1e-12))-math.log(max(c.area,1e-12)))
    preferred = [c for c in allowed if box_iou(target.crop_box, c.crop_box) <= preferred_iou]
    if preferred:
        return min(preferred, key=lambda c: (area_gap(c), box_iou(target.crop_box,c.crop_box), str(c.building_id)))
    return min(allowed, key=lambda c: (box_iou(target.crop_box,c.crop_box), area_gap(c), str(c.building_id)))
