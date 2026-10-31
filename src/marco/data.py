"""Dataset, transforms, and class weights for the ANN.

Every augmentation choice here is justified by something specific: drop orientation carries
no signal (flips, 90-degree rotation), source institutions have different imaging
setups/colour casts per the EDA (colour jitter, random greyscale), and image quality varies
(mild blur) - see notebooks/01_eda.ipynb Findings 2 and 3.
"""

import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

CLASS_ORDER = ["Clear", "Crystals", "Other", "Precipitate"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomRightAngleRotation:
    """Rotate by a random multiple of 90 degrees.

    Unlike arbitrary-angle rotation, this needs no padding/interpolation at the borders
    (a square image rotated by a right angle is still the same square), and matches the
    justification for the flips: drop orientation under the microscope carries no signal.
    """

    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle)


def build_transforms(train: bool) -> transforms.Compose:
    """Direct resize to IMAGE_SIZE (no crop) - crystals can sit at the drop edge, so cropping
    risks cutting out the decision-relevant signal."""
    resize_and_normalize = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if not train:
        return transforms.Compose(resize_and_normalize + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose(resize_and_normalize + [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRightAngleRotation(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MarcoDataset(Dataset):
    """Loads images and labels from a data/splits.csv subset. split is required (not
    defaulted) so callers must deliberately choose train/val/test - test is read only by
    scripts/final_eval.py, per CLAUDE.md rule 2."""

    def __init__(self, splits_df: pd.DataFrame, data_dir: Path, train: bool):
        self.df = splits_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = build_transforms(train)
        self.label_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.data_dir / row["path"]).convert("RGB")
        img = self.transform(img)
        label = self.label_to_idx[row["label"]]
        return img, label


def load_split(splits_path: Path, split: str) -> pd.DataFrame:
    """Load one named split (train/val/test) from data/splits.csv."""
    splits = pd.read_csv(splits_path)
    return splits[splits["split"] == split].reset_index(drop=True)


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency class weights from the actual train-split counts, matching the
    class_weight='balanced' approach already used for the classical baseline."""
    counts = train_df["label"].value_counts().reindex(CLASS_ORDER)
    weights = len(train_df) / (len(CLASS_ORDER) * counts)
    return torch.tensor(weights.to_numpy(), dtype=torch.float32)
