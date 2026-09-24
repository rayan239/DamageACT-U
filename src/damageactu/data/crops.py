from __future__ import annotations
from pathlib import Path
from damageactu.utils.frozen_source import verify_frozen_source, import_from_path


def load_canonical_crop_module(frozen_project_root):
    mapping = verify_frozen_source(frozen_project_root)
    return import_from_path("damageactu_frozen_crop_utils", mapping["crop_utils.py"])


def canonical_crop_pair(row, xbd_root, frozen_project_root, crop_config):
    mod = load_canonical_crop_module(frozen_project_root)
    return mod.crop_pair(row, Path(xbd_root), crop_config)
