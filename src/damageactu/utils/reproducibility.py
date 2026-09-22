from __future__ import annotations
from pathlib import Path
import json
import os
import platform
import random
import subprocess
import sys
import numpy as np
from .hashing import raw_file_sha256


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def capture_environment() -> dict:
    out = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import numpy, pandas, sklearn, scipy
        out.update({
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
        })
    except Exception:
        pass
    try:
        import torch, torchvision
        out.update({
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception:
        pass
    return out


def assert_no_test_rows(df, role_col="event_role"):
    if role_col in df.columns and df[role_col].astype(str).str.lower().eq("test").any():
        raise RuntimeError("Held-out TEST rows are forbidden in this development operation.")


def build_artifact_manifest(root):
    root = Path(root)
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rows.append({
            "path": p.relative_to(root).as_posix(),
            "size_bytes": p.stat().st_size,
            "raw_sha256": raw_file_sha256(p),
        })
    return rows


def current_git_commit(root="."):
    try:
        return subprocess.check_output(
            ["git","rev-parse","HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None
