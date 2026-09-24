"""
PyTorch Dataset for DamageACT-U paired building classification.

This version supports both:
- direct execution from src/data/
- package import from src.training / src.evaluation

Core design:
- PRE geometry defines the crop.
- the same pixel window is applied to PRE and POST.
- only paired geometric augmentation is allowed during main training.
- ImageNet normalization is used for the pretrained ResNet-18 backbone.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image

try:
    from .crop_utils import CropConfig, crop_pair
except ImportError:
    from crop_utils import CropConfig, crop_pair


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_paired_geometry_augmentation(
    pre: Image.Image,
    post: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    k = random.randint(0, 3)
    rotate_ops = [
        None,
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
    ]
    op = rotate_ops[k]
    if op is not None:
        pre = pre.transpose(op)
        post = post.transpose(op)

    if random.random() < 0.5:
        pre = pre.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        post = post.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        pre = pre.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        post = post.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    return pre, post


def to_model_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor


class PairedXBDDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        crop_config: Optional[CropConfig] = None,
        training: bool = False,
        augment_geometry: bool = False,
        return_metadata: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.crop_config = crop_config or CropConfig()
        self.training = bool(training)
        self.augment_geometry = bool(augment_geometry)
        self.return_metadata = bool(return_metadata)

        if self.augment_geometry and not self.training:
            raise ValueError("augment_geometry=True is allowed only for training data.")

        self.df = pd.read_csv(self.manifest_path)

        required = {
            "building_id", "scene_id", "disaster",
            "label_name", "label_id",
            "pre_image", "post_image",
            "pre_image_width", "pre_image_height",
            "post_image_width", "post_image_height",
            "pre_bbox_width_px", "pre_bbox_height_px",
            "pre_centroid_x", "pre_centroid_y",
            "centroid_shift_px",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        if not self.df["label_id"].isin([0, 1, 2, 3]).all():
            bad = sorted(
                self.df.loc[
                    ~self.df["label_id"].isin([0, 1, 2, 3]), "label_id"
                ].unique()
            )
            raise ValueError(f"Unexpected label IDs: {bad}")

        if self.df["building_id"].duplicated().any():
            raise ValueError("Duplicate building_id values found in manifest.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        pre, post, crop_meta = crop_pair(
            row=row,
            dataset_root=self.dataset_root,
            cfg=self.crop_config,
        )

        if self.training and self.augment_geometry:
            pre, post = apply_paired_geometry_augmentation(pre, post)

        pre_tensor = to_model_tensor(pre)
        post_tensor = to_model_tensor(post)
        label = torch.tensor(int(row["label_id"]), dtype=torch.long)

        sample = {
            "pre": pre_tensor,
            "post": post_tensor,
            "label": label,
        }

        if self.return_metadata:
            sample.update(
                {
                    "building_id": str(row["building_id"]),
                    "scene_id": str(row["scene_id"]),
                    "disaster": str(row["disaster"]),
                    "label_name": str(row["label_name"]),
                    "centroid_shift_px": torch.tensor(
                        float(row["centroid_shift_px"]), dtype=torch.float32
                    ),
                    "raw_building_max_dim_px": torch.tensor(
                        float(max(row["pre_bbox_width_px"], row["pre_bbox_height_px"])),
                        dtype=torch.float32,
                    ),
                    "crop_side_px": torch.tensor(
                        float(crop_meta["side_px"]), dtype=torch.float32
                    ),
                    "crop_center_shift_px": torch.tensor(
                        float(crop_meta["crop_center_shift_px"]), dtype=torch.float32
                    ),
                }
            )

        return sample
