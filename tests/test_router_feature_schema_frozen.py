from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from damageactu.routing.features import FEATURE_NAMES, build_router_features


EXPECTED_FEATURE_NAMES = [
    "post_p0", "pair_p0", "diff_p0", "absdiff_p0",
    "post_p1", "pair_p1", "diff_p1", "absdiff_p1",
    "post_p2", "pair_p2", "diff_p2", "absdiff_p2",
    "post_p3", "pair_p3", "diff_p3", "absdiff_p3",
    "confidence_post", "confidence_pair",
    "entropy_post", "entropy_pair",
    "severity_post", "severity_pair",
    "conf_diff", "entropy_diff", "severity_diff",
    "pred_disagree",
]


def _example_frame() -> pd.DataFrame:
    post = np.array(
        [
            [0.70, 0.20, 0.08, 0.02],
            [0.10, 0.20, 0.30, 0.40],
        ],
        dtype=float,
    )
    pair = np.array(
        [
            [0.60, 0.25, 0.10, 0.05],
            [0.15, 0.25, 0.35, 0.25],
        ],
        dtype=float,
    )

    data = {}
    for k in range(4):
        data[f"p{k}_post"] = post[:, k]
        data[f"p{k}_pair"] = pair[:, k]

    severity_axis = np.arange(4, dtype=float)
    data["confidence_post"] = post.max(axis=1)
    data["confidence_pair"] = pair.max(axis=1)
    data["entropy_post"] = -(post * np.log(post)).sum(axis=1)
    data["entropy_pair"] = -(pair * np.log(pair)).sum(axis=1)
    data["severity_post"] = post @ severity_axis
    data["severity_pair"] = pair @ severity_axis
    data["pred_post"] = post.argmax(axis=1)
    data["pred_pair"] = pair.argmax(axis=1)
    return pd.DataFrame(data)


def test_feature_names_are_exact_frozen_schema():
    assert len(FEATURE_NAMES) == 26
    assert list(FEATURE_NAMES) == EXPECTED_FEATURE_NAMES


def test_builder_emits_exact_feature_order():
    features = build_router_features(_example_frame())
    assert list(features.columns) == EXPECTED_FEATURE_NAMES
    assert features.shape == (2, 26)
    assert np.isfinite(features.to_numpy(float)).all()


def test_router_schema_contains_no_label_or_event_metadata():
    forbidden = {
        "true_label",
        "label",
        "label_id",
        "target",
        "building_id",
        "scene_id",
        "disaster",
        "event",
        "pair_key",
        "patch_id",
    }
    assert forbidden.isdisjoint(FEATURE_NAMES)


def test_schema_matches_frozen_neural_router_when_artifact_is_present():
    repo_root = Path(__file__).resolve().parents[1]
    artifact_path = (
        repo_root
        / "checkpoints"
        / "routers"
        / "final_neural_safe_router.pt"
    )

    if not artifact_path.exists():
        pytest.skip("Frozen neural-router artifact is not present.")

    torch = pytest.importorskip("torch")
    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=False,
    )

    assert "feature_names" in artifact
    assert list(artifact["feature_names"]) == EXPECTED_FEATURE_NAMES
    assert list(FEATURE_NAMES) == list(artifact["feature_names"])
