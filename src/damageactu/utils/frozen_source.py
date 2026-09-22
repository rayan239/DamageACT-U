from __future__ import annotations
from pathlib import Path
import importlib.util
from .hashing import raw_file_sha256


EXPECTED = {
    "build_manifest.py": "4337d8ec54d2fdaa5361e4c30e7544e16f0edf306f517dcff612b7fe12d7807a",
    "dataset.py": "165a71c4ebf6adea14f20f735bc487e883c5448e5ccfaf4fe8b74e49a3a77044",
    "crop_utils.py": "8c34139a0f96eaf8246dd228cf27c83386d6c1ca049426c25d28666576589593",
    "damage_models.py": "9f9ee3382902f339309ddf03a203b882d42e8674de7bf8ee3d18b14863478f2d",
}


def verify_frozen_source(root):
    root = Path(root)
    mapping = {
        "build_manifest.py": root / "src/data/build_manifest.py",
        "dataset.py": root / "src/data/dataset.py",
        "crop_utils.py": root / "src/data/crop_utils.py",
        "damage_models.py": root / "src/models/damage_models.py",
    }
    for name, path in mapping.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        got = raw_file_sha256(path)
        if got != EXPECTED[name]:
            raise RuntimeError(f"Frozen source hash mismatch for {name}: expected {EXPECTED[name]}, got {got}")
    return mapping


def import_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
