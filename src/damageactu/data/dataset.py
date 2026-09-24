from __future__ import annotations
from damageactu.utils.frozen_source import verify_frozen_source, import_from_path


def load_canonical_dataset_module(frozen_project_root):
    mapping = verify_frozen_source(frozen_project_root)
    return import_from_path("damageactu_frozen_dataset", mapping["dataset.py"])
